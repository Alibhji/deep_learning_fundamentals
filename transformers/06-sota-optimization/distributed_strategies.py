"""
Distributed Strategies for Transformers
- FSDP setup helpers
- DDP scaffold
- Tensor/sequence parallel placeholders
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    FSDP = None


def setup_dist(backend='nccl'):
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))


def wrap_ddp(model: nn.Module) -> nn.Module:
    device_id = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    return model


def wrap_fsdp(model: nn.Module) -> nn.Module:
    if FSDP is None:
        return model
    device_id = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(device_id)
    return FSDP(model)


class TensorParallelPlaceholder(nn.Module):
    """Placeholder class to indicate where tensor parallel would be applied.
    For real TP, see Megatron-LM or Tensor Parallel libraries.
    """
    def __init__(self, module: nn.Module, world_size: int):
        super().__init__()
        self.module = module
        self.world_size = world_size

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
