import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

def divide(x, y):
    assert x % y == 0
    return x // y

class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError
    

class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size,
        output_size,
        bias = False
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shared_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shared_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shared_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight, self.bias)
    
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size: int, output_sizes: list[int], bias=False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        start = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, start, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# caculate q,k,v.
# self.weight.shape is [q.size + k.size + v.size, hidden_size]
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False    
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) *head_size
        super().__init__(hidden_size, output_size, bias)

    # loaded_shard_id = 'q' or 'k' or 'v'
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        if loaded_shard_id == 'q':
            start = 0
            shard_size = self.num_heads * self.head_size
        elif loaded_shard_id == 'k':
            start = self.num_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        else:
            start = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            shard_size = self.num_kv_heads * self.head_size
        # Locate the position of loaded_shard_id in self.weight
        param_data = param_data.narrow(self.tp_dim, start, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias = False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)
    
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
