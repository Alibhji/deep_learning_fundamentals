"""
Quantization Toolkit for Transformers
- Post-training quantization (PTQ) scaffolding
- Quantization-aware training (QAT) stubs
- Calibration utilities
"""

import torch
import torch.nn as nn


def ptq_dynamic_linear_only(model: nn.Module) -> nn.Module:
    """Apply dynamic PTQ to Linear layers (CPU-friendly)."""
    import torch.quantization as quant
    qmodel = quant.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return qmodel


def qat_prepare(model: nn.Module) -> nn.Module:
    """Prepare model for QAT (requires layer fusions & observers)."""
    try:
        import torch.ao.quantization as aq
    except Exception:
        return model
    model.train()
    model.qconfig = aq.get_default_qat_qconfig('fbgemm')
    aq.prepare_qat(model, inplace=True)
    return model


def qat_convert(model: nn.Module) -> nn.Module:
    try:
        import torch.ao.quantization as aq
    except Exception:
        return model
    model.eval()
    aq.convert(model, inplace=True)
    return model


def calibrate(model: nn.Module, dataloader, num_batches: int = 10, device='cpu'):
    """Run a few batches through model to collect quant stats (observers)."""
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(**batch)


def export_onnx(model: nn.Module, sample_inputs: dict, path: str = 'model.onnx') -> str:
    model.eval()
    dummy = tuple(sample_inputs.values())
    torch.onnx.export(
        model,
        dummy if len(dummy) > 1 else dummy[0],
        path,
        opset_version=17,
        input_names=list(sample_inputs.keys()),
        output_names=['output'],
        dynamic_axes={name: {0: 'batch'} for name in sample_inputs.keys()}
    )
    return path
