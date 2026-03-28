import torch
from torch import nn
from utils.context import get_context

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.is_contiguous() and value.is_contiguous()
    assert slot_mapping.numel() == N

    slot_mapping = slot_mapping.to(torch.long)
    valid = slot_mapping != -1
    if not torch.any(valid):
        return

    flat_key = key.view(N, D)
    flat_value = value.view(N, D)
    flat_k_cache = k_cache.view(-1, D)
    flat_v_cache = v_cache.view(-1, D)
    flat_k_cache[slot_mapping[valid]] = flat_key[valid]
    flat_v_cache[slot_mapping[valid]] = flat_value[valid]

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
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return x
        assert self.num_heads % self.num_kv_heads == 0
        repeat = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix_len: int = 0,
        causal: bool = True,
    ) -> torch.Tensor:
        out_dtype = q.dtype
        q = q.float()
        k = self._expand_kv_heads(k.float())
        v = self._expand_kv_heads(v.float())

        # scores: [num_heads, q_len, k_len]
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale
        if causal:
            q_len = q.size(0)
            k_len = k.size(0)
            q_pos = prefix_len + torch.arange(q_len, device=q.device)
            k_pos = torch.arange(k_len, device=q.device)
            mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, v)
        return out.to(out_dtype)

    def _gather_from_cache(self, block_table: torch.Tensor, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        # cache shape: [num_kvcache_blocks, block_size, num_kv_heads, head_dim]
        block_size = self.k_cache.size(1)
        full_blocks, tail = divmod(seqlen, block_size)
        num_blocks = full_blocks + (1 if tail > 0 else 0)
        block_ids = block_table[:num_blocks].to(torch.long)

        k = self.k_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen]
        v = self.v_cache[block_ids].reshape(-1, self.num_kv_heads, self.head_dim)[:seqlen]
        return k, v
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            outputs = []
            batch = context.cu_seqlens_q.numel() - 1
            for i in range(batch):
                q_start = int(context.cu_seqlens_q[i].item())
                q_end = int(context.cu_seqlens_q[i + 1].item())
                k_start = int(context.cu_seqlens_k[i].item())
                k_end = int(context.cu_seqlens_k[i + 1].item())

                q_i = q[q_start:q_end]
                k_len = k_end - k_start
                if context.block_tables is None:
                    k_i = k[k_start:k_end]
                    v_i = v[k_start:k_end]
                else:
                    k_i, v_i = self._gather_from_cache(context.block_tables[i], k_len)

                prefix_len = k_len - q_i.size(0)
                outputs.append(self._attention(q_i, k_i, v_i, prefix_len=prefix_len, causal=True))
            return torch.cat(outputs, dim=0)

        outputs = []
        for i in range(q.size(0)):
            q_i = q[i:i + 1]
            k_len = int(context.context_lens[i].item())
            k_i, v_i = self._gather_from_cache(context.block_tables[i], k_len)
            outputs.append(self._attention(q_i, k_i, v_i, causal=False))
        return torch.cat(outputs, dim=0)
