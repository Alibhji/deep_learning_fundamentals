# Low-level Compute: From Pixel to Logit on CPU/GPU

This section explains in detail how a tiny transformer-like vision model (e.g., `TinyViT32`) is materialized in memory, how parameters are placed and accessed on GPU, and how a single pixel flows through the compute stack, including registers, caches, and matrix-multiply pipelines.

## Memory Hierarchy Overview
- CPU: Registers → L1/L2/L3 caches → DRAM
- GPU: Registers (per-thread) → Shared memory/L1 → L2 → HBM/GDDR
- DMA/PCIe/NVLink: Host-device transfers
- Tensors in frameworks (PyTorch) are views over contiguous or strided buffers with dtypes (fp32, fp16, bf16, int8)

## Parameter Placement (GPU)
When `model.to('cuda')` is called:
- Each `nn.Parameter` is allocated on device memory (HBM). Typical layout is row-major contiguous for `nn.Linear.weight` of shape (out_features, in_features).
- Optimizer states (e.g., Adam moments) are separate device buffers.
- During forward:
  - Weights stream from HBM → L2 → L1/SMEM → registers as needed
  - Activations are maintained in device memory between layers; fused kernels reduce round-trips

### Example: `TinyViT32`
- `patch_embed.weight`: shape [D, 3*4*4]
- `pos`: shape [1, 64, D]
- `attn.qkv` (packed inside MHA): internal projections [D, 3D]
- `mlp.*`: linear layers [D, 2D] and [2D, 1]

All of these tensors reside in GPU device memory once moved to CUDA.

## Tensor Layout & Access Patterns
- NCHW for images: (batch, channels, height, width) for PyTorch CUDA ops
- Tokens: (batch, seq, dim) for batch-first MHA; kernels often transform to optimal tile sizes
- MatMul implements GEMM: C = A×B with tiling, using shared memory and registers

## Registers and MatMul Execution (GPU)
- Threads are grouped into warps (e.g., 32 threads). A block cooperatively loads tiles of A and B into shared memory.
- Each thread accumulates partial sums in registers for a C-tile.
- Tensor cores (on modern GPUs) perform MMA instructions on fp16/bf16/INT8 tiles.
- Minimal register operands per MMA depend on tile size (e.g., 16×16 fragments). Register pressure impacts occupancy.

## Minimal Register/Unit Requirements (Conceptual)
- Scalar FMA: needs registers for A[i,k], B[k,j], accumulator C[i,j]
- Tile MMA: fragments of A, B, C per thread → several registers (dozens), plus shared memory for the tile
- For INT8 kernels: additional scale/zero-point registers/buffers

## Pixel-to-Logit Flow (One Pixel)
1. Host loads image (CPU RAM) → optional preproc → transfer to GPU via PCIe/NVLink
2. On GPU: image in device buffer (NCHW). A 4×4 patch containing the pixel is extracted via strided loads
3. Patch flatten → `patch_embed` GEMM multiplies patch vector by weight matrix
4. Token + positional embedding → MHA: Q,K,V projections (GEMMs) → attention scores (Q×K^T) → softmax → weight×V
5. Residual add & LayerNorm → MLP (two GEMMs + activation)
6. Pool tokens → final linear → sigmoid → probability

## Diagram: Memory/Compute Flow
```
CPU DRAM (image) → PCIe/NVLink → GPU HBM
  ↓                           ↓
CPU registers            L2 cache → SMEM → Registers → Tensor Cores
                                   ↓            ↓
                              MatMul tiles   Accumulators
                                   ↓
                            Activations in HBM
```

## Exact Parameter Locations (Framework View)
- `model.patch_embed.weight.device` → `cuda:0`
- `model.pos.device` → `cuda:0`
- `for n,p in model.named_parameters(): print(n, p.shape, p.device)`
- Physical HBM address is abstracted; kernels receive device pointers for buffers

## Example: Dumping Parameter Pointers (Advanced)
In CUDA/C++ extensions, you can inspect raw device pointers (`void*`). In Python/PyTorch, access is abstracted; but you can log `tensor.data_ptr()` which returns the device address offset (not stable across runs).

```python
for n, p in model.named_parameters():
    print(n, hex(p.data_ptr()), p.shape, p.dtype, p.device)
```

## Single-Pixel Walkthrough (Detail)
- Pixel at (h,w) contributes to its 4×4 patch. The patch-flatten vector multiplies `patch_embed.weight.T` (GEMM). Each output dim element is dot(product) over 48 inputs.
- In MHA, the pixel's token affects Q/K/V for its token; attention includes interactions with all tokens (global context). Q×K^T accumulates over D dims; softmax normalized over sequence.
- Final logit aggregates contributions from all tokens; gradients propagate back similarly in training.

## CPU Path (Fallback)
- Same conceptual flow; BLAS (MKL/OpenBLAS) handles GEMMs; vectorization via AVX/NEON; caches feed registers; no tensor cores.

## Tools to Observe/Verify
- Nsight Systems/Compute: kernel traces, memory throughput
- PyTorch profiler: op-level traces and tensor shapes
- CUPTI/rocprof: low-level counters

---

See the companion notebook for annotated code, parameter dumps, and profiling tips.

