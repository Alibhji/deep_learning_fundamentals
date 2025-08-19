# SOTA Optimization for Transformer Training & Inference

This module covers state-of-the-art techniques to optimize transformer models for training and inference. It spans algorithmic improvements, systems-level acceleration, quantization and sparsity, parameter-efficient finetuning (PEFT), compilation stacks, and serving.

## 🎯 Goals
- Reduce training and inference cost (time, memory, and energy)
- Maintain or improve model quality
- Enable deployment at scale with predictable latency and throughput

## 🔍 Topics Covered

### 1) Attention & Architecture Efficiency
- FlashAttention v1/v2: IO-aware exact attention kernels for large sequence lengths
- Multi-Query & Grouped-Query Attention (MQA/GQA): Shared K/V for faster decode
- Speculative Decoding (Medusa, Lookahead): Generate candidates with a small draft model
- Dynamic NTK & Positional Tweaks: RoPE scaling, ALiBi for extrapolation
- Sparse/Local Attention Variants: Long sequence efficiency
- KV Cache Optimizations: Paged KV cache, blockwise kv-cache

### 2) Training Optimizations
- Mixed Precision: BF16/FP16 with loss-scaling
- ZeRO & Sharded Optimizers: DeepSpeed ZeRO stages 1–3
- Gradient Checkpointing & Activation Recomputation
- FSDP & Tensor/Sequence Parallelism: PyTorch FSDP, Megatron-LM style parallelism
- Optimizer Choices: AdamW → Lion/Sophia for improved stability/throughput
- Data/Batching: Bucketing, packing, efficient dataloaders (streaming datasets)
- Curriculum & Continual Learning: Efficient adaptation

### 3) Inference Optimizations
- TensorRT-LLM / BetterTransformer / Inductor: Graph capture and kernel fusion
- VLLM & PagedAttention: High-throughput serving with efficient KV cache memory
- Quantization: 8-bit (LLM.int8), 4-bit (QLoRA, GPTQ, AWQ), 2-bit (NF4 variants)
- Speculative Decoding & Cascaded Models: Faster generation without quality drop
- Sliding Window / Attention Sinks for long-context decoding
- Operator-level Fusions: RMSNorm+MatMul, SiLU/SwiGLU fusions

### 4) Parameter-Efficient Finetuning (PEFT)
- LoRA / QLoRA / AdaLoRA: Low-rank adapters; QLoRA enables 4-bit finetuning
- Prefix/Prompt Tuning & P-Tuning v2
- Adapters & IA3: Inserted lightweight modules with frozen base
- Transfer & Merging: Adapter merging & model surgery

### 5) Quantization & Sparsity
- Post-Training Quantization (PTQ): GPTQ, AWQ
- Quantization-Aware Training (QAT): Learned step sizes, LLM-aware QAT
- Blockwise Pruning & Movement Pruning
- Structured Sparsity (2:4 NVIDIA) & N:M sparsity kernels

### 6) Compilers & Runtimes
- PyTorch 2.x: torch.compile (AOTAutograd, Inductor)
- Triton Kernels & Custom Attention Kernels
- ONNX Runtime, TensorRT, TensorRT-LLM pipelines
- OpenVINO / TVM / XLA for cross-hardware acceleration

## 🔧 Code Modules in This Folder
- `train_optimizations.py`: Mixed precision, checkpointing, FSDP/ZeRO patterns
- `inference_optimizations.py`: KV cache, quantization, high-throughput serving
- `quantization_toolkit.py`: PTQ/QAT scaffolding with calibration hooks
- `distributed_strategies.py`: FSDP, tensor/sequence parallel examples

## 📈 Practical Recipes
- QLoRA Finetuning: 4-bit adapters on consumer GPUs
- FlashAttention + RoPE: Long sequences with lower memory footprint
- vLLM Serving: Throughput-oriented serving with PagedAttention
- Speculative Decoding: Draft-and-verify for faster generation

## 📚 References
- Dao et al., 2022, 2023 — FlashAttention v1/v2 (arXiv:2205.14135, arXiv:2307.08691)
- Dettmers et al., 2023 — QLoRA (arXiv:2305.14314)
- Kuchaiev et al., 2019 — Mixed precision training (NVIDIA Apex)
- Rajbhandari et al., 2020 — ZeRO: Memory Optimizations for Deep Networks (arXiv:1910.02054)
- Kwon et al., 2023 — Efficient Memory Management for LLM Serving (vLLM)
- Frantar et al., 2022 — GPTQ (arXiv:2210.17323)
- Lin et al., 2023 — AWQ (arXiv:2306.00978)
- Hu et al., 2021 — LoRA (arXiv:2106.09685)
- Shazeer, 2019 — LAMB Optimizer (arXiv:1904.00962)
- Liu et al., 2023 — Sophia (arXiv:2305.14342)

---

This folder includes runnable templates and a companion notebook showcasing the above techniques on toy models, and pointers to integrate with real stacks (DeepSpeed, FSDP, vLLM, TensorRT-LLM).
