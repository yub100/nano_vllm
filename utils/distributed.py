import torch.distributed as dist


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_tp_rank() -> int:
    return dist.get_rank() if is_distributed_initialized() else 0


def get_tp_world_size() -> int:
    return dist.get_world_size() if is_distributed_initialized() else 1

