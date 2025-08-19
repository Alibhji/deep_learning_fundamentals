"""
Tiny ViT-like model for 32x32 RGB -> binary human/not-human
- Super minimal for readability
- Dummy weights for demonstration
- Exporters for ONNX (and notes for TFLite)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TinyViT32(nn.Module):
    """Minimal ViT-style classifier for 32x32 RGB image.
    Patches: 4x4 => 8x8 = 64 tokens. Single attention-lite + MLP.
    Output: probability (sigmoid).
    """
    def __init__(self, embed_dim=64, patch=4, num_heads=1):
        super().__init__()
        self.patch = patch
        self.embed_dim = embed_dim

        # Patch embedding: linear over flattened 4x4x3
        self.patch_embed = nn.Linear(3 * patch * patch, embed_dim)

        # Positional embedding (learned, tiny)
        self.pos = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Attention-lite (single head via MHA)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 1)
        )

        self.reset_dummy()

    def reset_dummy(self):
        # Initialize small weights for a stable dummy forward
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        B, C, H, W = x.shape  # (B, 3, 32, 32)
        assert H == 32 and W == 32, "Expected 32x32 input"
        # Extract non-overlapping 4x4 patches: (B, 3, 32, 32) -> (B, 8, 8, 3, 4, 4)
        patches = x.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, 8, 8, 3, 4, 4)
        patches = patches.contiguous().view(B, 64, 3 * self.patch * self.patch)  # (B, 64, 48)
        tokens = self.patch_embed(patches)  # (B, 64, D)

        # Add positional embedding
        tokens = tokens + self.pos  # (B, 64, D)

        # Attention-lite
        tokens_norm = self.norm1(tokens)  # (B, 64, D)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)  # (B, 64, D)
        tokens = tokens + attn_out  # (B, 64, D)

        # Global average pooling over tokens
        pooled = tokens.mean(dim=1)  # (B, D)

        # Binary logit
        logit = self.mlp(pooled)  # (B, 1)
        prob = torch.sigmoid(logit)  # (B, 1)
        return prob


def export_onnx(model: nn.Module, path='tiny_vit32.onnx'):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy, path, opset_version=17,
        input_names=['input'], output_names=['prob'],
        dynamic_axes={'input': {0: 'batch'}, 'prob': {0: 'batch'}}
    )
    return path


def demo_inference():
    model = TinyViT32()
    x = torch.randn(1, 3, 32, 32)
    prob = model(x)
    print('Probability of human:', float(prob.item()))
    print('Predicted label:', bool(prob.item() > 0.5))


if __name__ == '__main__':
    demo_inference()
    path = export_onnx(TinyViT32())
    print('Exported ONNX to:', path)
