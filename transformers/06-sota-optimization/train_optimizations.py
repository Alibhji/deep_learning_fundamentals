"""
Training Optimizations for Transformers
- Mixed precision (AMP/BF16)
- Gradient checkpointing
- FSDP/ZeRO (scaffolding)
- Optimizer selection (AdamW, Lion, Sophia)
"""

import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

try:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    FSDP = None


class MixedPrecisionTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, use_bf16: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.use_bf16 = use_bf16
        self.scaler = None if use_bf16 else GradScaler()

    def step(self, loss: torch.Tensor):
        if self.use_bf16:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return loss.item()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            return loss.item()

    def forward_with_autocast(self, fn, *args, **kwargs):
        dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        with autocast(dtype=dtype):
            return fn(*args, **kwargs)


def enable_gradient_checkpointing(module: nn.Module):
    """Enable gradient checkpointing on supported submodules."""
    for m in module.modules():
        if hasattr(m, 'gradient_checkpointing_enable'):
            m.gradient_checkpointing_enable()
    return module


def wrap_with_fsdp(model: nn.Module, auto_wrap_policy=None) -> nn.Module:
    """Wrap model with FSDP if available."""
    if FSDP is None:
        return model
    return FSDP(model, auto_wrap_policy=auto_wrap_policy)


# Optimizers: AdamW, Lion, Sophia (minimal versions)
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                update = exp_avg.sign()
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                p.data.add_(update, alpha=-group['lr'])
        return loss


class Sophia(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            rho = group['rho']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['h'] = torch.zeros_like(p.data)
                m, h = state['m'], state['h']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                h.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = m / (h.sqrt() + rho)
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                p.data.add_(update, alpha=-group['lr'])
        return loss


def training_step(model, batch, criterion, trainer: MixedPrecisionTrainer):
    def forward():
        outputs = model(**batch)
        loss = criterion(outputs, batch['labels'])
        return loss
    loss = trainer.forward_with_autocast(forward)
    value = trainer.step(loss)
    return value
