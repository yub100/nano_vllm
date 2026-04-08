from collections import deque
from typing import Tuple

from config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager

# scheduler职责：
# 1.处理上一个step生成的token，依据块是否满来分配新物理kvcache block给逻辑块（将token加入seq中就已经是写入块了，
# 只不过这里写入的是逻辑块，在gpu前向计算时会依据这个toke所在逻辑块找到对应的kvcache物理块，然后计算这个token的kv写入cache
# 而其本身不需要存储在cache，推理时会直接拿临时变量存储q）
# 2.负责生成需要prefill或decode的seqs
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.max_chunk_size = config.max_num_chunk_tokens
        Sequence.max_chunk_size = config.max_num_chunk_tokens

        # The further to the left, the higher the priority.      
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def is_finished(self):
        return not (self.waiting or self.running)
    
    # When seq is seized, this function is called.
    # Using waiting.appendleft ensures that seq will be processed first when signal arrives.
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_decode = False
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def schedule_running(self, num_seqs: int, num_batched_tokens: int):
        seqs = []
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running[0]
            chunk_size = seq.chunk_size
            # prefill
            if chunk_size > 1:
                if chunk_size + num_batched_tokens > self.max_num_batched_tokens or not self.block_manager.can_append(seq):
                    break

                self.block_manager._append(seq)
            # decode
            else:
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    self.block_manager.may_append(seq)

            num_batched_tokens += chunk_size
            num_seqs += 1
            seqs.append(seq)
            self.running.popleft()
        
        self.running.extendleft(reversed(seqs))
        return seqs, num_seqs, num_batched_tokens

    def schedule_waiting(self, num_seqs: int, num_batched_tokens: int):
        seqs = []
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            self.block_manager.allocate(seq)
            chunk_size = seq.chunk_size
            if (chunk_size + num_batched_tokens > self.max_num_batched_tokens) or not self.block_manager.can_append(seq):
                self.block_manager.deallocate(seq)
                break
            self.block_manager._append(seq)
            seqs.append(seq)
            num_batched_tokens += chunk_size
            num_seqs += 1
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()

        self.running.extend(seqs)
        return seqs, num_seqs, num_batched_tokens

    def schedule(self) -> Tuple[list[Sequence], int]:
        scheduled_seqs = []

        # The number of seqence that scheduled_seqs get.
        num_seqs = 0
        num_batched_tokens = 0

        # 
        sch1, num_seqs, num_batched_tokens = self.schedule_running(num_seqs, num_batched_tokens)
        scheduled_seqs.extend(sch1)
        
        # 
        sch2, num_seqs, num_batched_tokens = self.schedule_waiting(num_seqs, num_batched_tokens)
        scheduled_seqs.extend(sch2)
        
        assert scheduled_seqs is not None

        return scheduled_seqs, num_batched_tokens

    def postprocess(self, scheduledBatch: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(scheduledBatch, token_ids):
            chunk_size = seq.chunk_size
            finished_prefill = (not seq.is_decode) and (seq.num_computed_tokens + chunk_size == seq.num_tokens)
            seq.num_computed_tokens += chunk_size
            if finished_prefill:
                seq.is_decode = True
            if seq.is_decode:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
