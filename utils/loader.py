import os
import torch
from torch import nn
from glob import glob
from safetensors import safe_open

# Those key need have their own weight_loader()
# packed_modules_mapping = {
#     "q_proj": ("qkv_proj", "q"),
#     "k_proj": ("qkv_proj", "k"),
#     "v_proj": ("qkv_proj", "v"),
#     "gate_proj": ("gate_up_proj", 0),
#     "up_proj": ("gate_up_proj", 1),
# }

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # Obtain all file path whose suffix is .safetensors
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # f.key() return path of all weights.
            # eg. weight_name = "model.layers.0.self_attn.k_proj.weight"
            for weight_name in f.keys():
                # used to load weights in packed_modules_mapping
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        # eg. "model.layers.0.self_attn.k_proj.weight"->"model.layers.0.self_attn.qkv_proj.weight"
                        param_name = weight_name.replace(k, v)
                        # obtain weight instance by param_name
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # used to weights sunch as rmsnorm...
                # If param carries user-defined weight_loader, use it;
                # otherwise, use default_weight_loader.
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


        


