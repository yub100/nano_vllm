import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    # get idx=n
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # the relative address of key[n, 1...i]
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # self.kv_cache = torch.empty(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
    # D = num_kv_heads * head_dim = key_stride
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # kvcache: [max_num_slots, D]
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # store one k and one v per block
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # if there not warm_up, it can be None and bind into physical address after ModelRunner.allocate_kv_cache()
        self.k_cache = self.v_cache = torch.tensor([])
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # unsure warmup operate rightly
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # block_tables have all seqs' block_table. if it is None, means it is the first round of prefill and no prefix.
            # Because if not prefix cache(block_tables is None), the key and value are stored discretely in the kvcache,
            # but block_tables is None and we can'y find all kv in cache by block_tables, so we should use k, v as parameters of fa
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                        max_seqlen_v=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                        softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens,
                                        block_table=context.block_tables, softmax_scale=self.scale, causal=True)
        return o            
