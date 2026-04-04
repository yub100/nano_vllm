
import torch
import pickle
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from config import Config
from models.qwen3 import Qwen3ForCausalLM
from utils.loader import load_model
from layers.sampler import Sampler
from engine.sequence import Sequence
from utils.context import set_context, get_context, reset_context

# Each process holds a ModelRunner instance, and bind a gpu
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.dist_initialized = False

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This project currently requires at least one NVIDIA GPU. "
                "If this machine should have a GPU, please check nvidia-container-runtime / CUDA visibility."
            )

        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
            self.dist_initialized = True
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        if self.world_size > 1:
            if rank == 0:
                # create shared memory
                self.shm = SharedMemory(name="vllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                # connect to shared memory
                self.shm = SharedMemory(name="vllm")
                self.loop()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batch_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batch_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # free memory size and total memory size.
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:4+n])
        self.event.clear()
        return method_name, args
        
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0: 4] = n.to_bytes(4, "little")
        self.shm.buf[4: 4+n] = data
        for event in self.event:
            event.set()


    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)
    
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.dist_initialized:
                dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if self.dist_initialized:
            dist.destroy_process_group()


    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def prepare_block_tables(self, scheduledBatch: list[Sequence]):
        max_block_table_len = max(len(seq.block_table) for seq in scheduledBatch)
        block_tables = [seq.block_table + [-1] * (max_block_table_len - len(seq.block_table)) for seq in scheduledBatch]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def prepare(self, scheduledBatch: list[Sequence]):
        # prepare decode
        # prepare prefill
        # generate tokens mask
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = []
        is_decodes = []
        context_lens = []

        for seq in scheduledBatch:
            num_tokens = seq.chunk_size

            # tokens in [start, end) need to import modelruner
            chunk_start_idx = seq.chunk_start_idx
            end = chunk_start_idx + num_tokens

            context_lens.append(end)
            input_ids.extend(seq.token_ids[chunk_start_idx:end])
            positions.extend(range(chunk_start_idx, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + num_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, num_tokens)
            max_seqlen_k = max(max_seqlen_k, end)
            
            # decode: num_tokens == 1
            if num_tokens == 1:
                token_idx = len(seq) - 1
                block_idx = token_idx // self.block_size
                block_offset = token_idx % self.block_size
                slot_mapping.append(seq.block_table[block_idx] * self.block_size + block_offset)
            # prefill: num_tokens > 1
            elif num_tokens > 1:
                start_block_id = seq.chunk_start_block_idx
                # the fist chunk's first token relative position in block
                position_within_start_block = seq.position_within_start_block

                slot_mapping.extend(seq.block_table[start_block_id] * self.block_size + i for i in range(position_within_start_block, self.block_size))

                num_remain_tokens = num_tokens - (self.block_size - position_within_start_block)
                if num_remain_tokens > 0:
                    blockid_start = start_block_id + 1
                    blockid_end = seq.block_table[-1]
                    start_block_id + (num_remain_tokens + self.block_size - 1) % self.block_size
                    for i in range(blockid_start, blockid_end + 1):
                        chunked_start = seq.block_table[i] * self.block_size

                        if (i == blockid_end):
                            chunked_end = chunked_start + num_remain_tokens % self.block_size
                        else:
                            chunked_end = chunked_start + self.block_size

                        slot_mapping.extend(range(chunked_start, chunked_end))
            
            is_decodes.append(seq.is_decode)
            
            
        block_tables = self.prepare_block_tables(scheduledBatch)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        is_decodes = torch.tensor(is_decodes, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        
        set_context(max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k, context_lens, slot_mapping, block_tables, is_decodes)
        return input_ids, positions

    @torch.inference_mode
    def run_model(self, input_ids: torch.Tensor, position: torch.Tensor):
        if self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, position))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # Write data to gpu
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = position
            # since this dict used to write kvcache, should drop invalid value.
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, scheduledBatch: list[Sequence])->list[int]:
        input_ids, position = self.prepare(scheduledBatch)
        temperatures = self.prepare_sample(scheduledBatch) if self.rank == 0 else None
        logits = self.run_model(input_ids, position)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # graph and it's variable likes "input_ids", "positions" and so on can use in Multi-round batch
    @torch.inference_mode() # 
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_batched_tokens, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)

        context_lens = torch.zeros(config.max_num_seqs, dtype=torch.int32)
        block_tables = torch.zeros(config.max_num_seqs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(config.max_num_seqs, hf_config.hidden_size)

        # [1, 2, 4, 8, 16, 32, 48...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # dictory
        self.graphs = {}

        self.graph_pool = None
        for bs in reversed(self.graph_bs):
            # create a new graph
            graph = torch.cuda.CUDAGraph()
            set_context(slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # warmup
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # capture graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            # Prevent next captrue begin before this captrue havn't over
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids = input_ids,
            positions = positions,
            slot_mapping = slot_mapping,
            block_tables = block_tables,
            context_lens = context_lens,
            outputs = outputs
        )
