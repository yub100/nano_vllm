from copy import copy
from enum import Enum, auto
from itertools import count

from sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()
    max_chunk_size = 256

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        # token_ids just is length of prompt, self.token_ids's length can grow
        self.token_ids = copy(token_ids)

        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_token
        self.ignore_eos = sampling_params.ignore_eos


        self.num_computed_tokens = 0
        self.is_decode = False

    @property
    def chunk_size(self):
        if self.is_decode:
            return 1
        return (self.num_tokens - self.num_computed_tokens) if (self.num_tokens - self.num_computed_tokens) < self.max_chunk_size else self.max_chunk_size
    
    @property
    def chunk_start_idx(self):
        return self.num_computed_tokens 
    
    @property
    def chunk_start_block_idx(self):
        return self.num_computed_tokens // self.block_size

    @property
    def position_within_start_block(self):
        return self.num_computed_tokens % self.block_size
    
    @property
    def num_need_append_block(self):
        return (self.num_computed_tokens + self.chunk_size + self.block_size - 1) // self.block_size - len(self.block_table)
    
        
    def __len__(self):
        return self.num_tokens
    
    def __getitem__(self, key):
        return self.token_ids[key]
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
    
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]
    
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
    
    # can not use len(block_table), since block_table may be include pre-allocated empty block
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size
    
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        return self.num_computed_tokens - (len(self.block_table) - 1) * self.block_size
    
    @property
    def last_block_tokens(self):
        return self.token_ids[self.block_size * self.num_blocks : self.num_computed_tokens]

    def block(self, i):
        return self.token_ids[self.block_size * i : self.block_size * (i + 1)] if i < self.num_blocks - 1 else self.token_ids[self.block_size * i : self.num_tokens]

    # update dataset but not update block, it happens in decode
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
