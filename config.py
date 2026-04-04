import os
from dataclasses import dataclass
from transformers import AutoConfig

# used for simplify data definition
@dataclass
class Config:
    model: str
    
    # the maximum number of tokens enter one step, accumulate last step
    max_num_batched_tokens: int = 16384
    
    # the maximum number of sequences that one batch
    max_num_seqs: int = 512
    
    # the maximum number of (tokens of prompt + already generated tokens). for one prompt, not for a batch.
    max_model_len: int = 4096

    # 90% memory for engine, 10% for model weights and others
    gpu_memory_utilization: float = 0.9

    # 1 gpu
    tensor_parallel_size: int = 1

    enforce_eager: bool = False

    # load model's config
    hf_config: AutoConfig | None = None

    # updated in LLMEngine.__init__()
    eos: int = -1

    # can store 256 tokens' kv
    kvcache_block_size: int = 256

    # calculate in ModelRunner.allocate_kv_cache()
    num_kvcache_blocks: int = -1

    enable_chunked_prefill = True

    # prefill chunk's size
    max_num_chunk_tokens = 256

    def __post_init__(self):
        assert os.path.isdir(self.model)

        # load model's config，refer to Qwen3DecoderLayer
        self.hf_config = AutoConfig.from_pretrained(self.model)

        # max_position_embeddings is the maximum length of rotary_emb
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
