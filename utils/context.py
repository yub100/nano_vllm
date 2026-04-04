from dataclasses import dataclass
import torch

@dataclass
class Context:
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_decodes: torch.Tensor | None = None


_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(max_seqlen_q=0, max_seqlen_k=0,
                cu_seqlens_q=None, cu_seqlens_k=None, context_lens=None,
                slot_mapping=None, block_tables=None, is_decodes=None):
    global _CONTEXT
    _CONTEXT = Context(max_seqlen_q, max_seqlen_k,
                       cu_seqlens_q, cu_seqlens_k, context_lens,
                       slot_mapping, block_tables, is_decodes)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

