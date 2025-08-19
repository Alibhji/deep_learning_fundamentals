# Advanced Low-Level Compute Flowchart

```mermaid
flowchart TD
    subgraph "Host CPU Processing"
        PIX["Pixel Data (CPU RAM)"] --> PREPROC["Preprocessing & Normalization\nImage resizing, normalization\nData type conversion\nMemory layout optimization"]
        
        PREPROC --> DMA_SETUP["DMA Setup\nHost memory pinning\nCUDA stream creation\nMemory transfer scheduling"]
    end
    
    subgraph "GPU Memory Transfer"
        DMA_SETUP --> DMA["DMA Transfer\nCPU RAM → GPU HBM\nPCIe bandwidth utilization\nOverlap with computation"]
        
        DMA --> GPU_MEM["GPU Global Memory\nHBM2/HBM3 bandwidth\nMemory coalescing\nCache line alignment"]
    end
    
    subgraph "GPU Kernel Execution"
        GPU_MEM --> PATCH_EXTRACT["Patch Extraction Kernel\nCUDA kernel launch\nThread block organization\nShared memory usage\nMemory access patterns"]
        
        PATCH_EXTRACT --> GEMM1["GEMM Kernel 1: Patch → Embed\ncuBLAS/cuDNN optimization\nTensor core utilization\nMemory bandwidth optimization\nKernel fusion opportunities"]
        
        GEMM1 --> ATTENTION["Attention Computation\nQ/K/V GEMMs\nSoftmax kernel\nAttention matrix multiplication\nMemory-efficient algorithms"]
        
        ATTENTION --> GEMM2["GEMM Kernel 2: FFN\nLinear layer computations\nActivation functions\nGradient computation\nBackward pass optimization"]
    end
    
    subgraph "Memory Management"
        GPU_MEM --> MEMORY_OPT["Memory Optimization\nPaged memory management\nMemory pooling\nFragmentation avoidance\nCache-aware access patterns"]
        
        MEMORY_OPT --> SHARED_MEM["Shared Memory Usage\nL1 cache optimization\nRegister allocation\nWarp-level parallelism\nMemory coalescing"]
    end
    
    subgraph "Compute Optimization"
        ATTENTION --> COMPUTE_OPT["Compute Optimization\nTensor core utilization\nMixed precision (FP16/BF16)\nKernel fusion\nCustom CUDA kernels"]
        
        COMPUTE_OPT --> PARALLEL["Parallelism Strategies\nData parallelism\nModel parallelism\nPipeline parallelism\nHybrid approaches"]
    end
    
    subgraph "Output & Transfer"
        GEMM2 --> OUTPUT["Output Generation\nFinal layer computation\nActivation functions\nOutput formatting"]
        
        OUTPUT --> HOST_READBACK["Host Readback\nGPU → CPU transfer\nResult processing\nMemory deallocation"]
        
        HOST_READBACK --> SERVING["Serving & Inference\nReal-time processing\nBatch optimization\nLatency optimization"]
    end
    
    subgraph "Performance Optimization"
        KERNEL_OPT["Kernel Optimization\nGrid/block size tuning\nMemory access patterns\nRegister usage optimization\nOccupancy maximization"]
        
        MEMORY_BW["Memory Bandwidth\nHBM bandwidth utilization\nMemory access coalescing\nCache hit rate optimization\nMemory latency hiding"]
        
        COMPUTE_EFF["Compute Efficiency\nTensor core utilization\nSM occupancy\nInstruction-level parallelism\nWarp divergence minimization"]
    end
    
    subgraph "Hardware Considerations"
        GPU_SPECS["GPU Specifications\nCUDA cores, Tensor cores\nMemory bandwidth\nShared memory per SM\nRegister file size"]
        
        MEMORY_HIERARCHY["Memory Hierarchy\nL1 cache (per SM)\nL2 cache (global)\nHBM global memory\nHost system memory"]
        
        INTERCONNECT["Interconnect\nPCIe bandwidth\nNVLink (multi-GPU)\nInfiniBand (distributed)\nNetwork topology"]
    end
    
    subgraph "Key Parameters"
        BATCH_SIZE["batch_size: Processing batch size"]
        IMG_SIZE["img_size: Input image dimensions"]
        PATCH_SIZE["patch_size: Patch extraction size"]
        EMBED_DIM["embed_dim: Embedding dimension"]
        NUM_HEADS["num_heads: Attention heads"]
        SEQ_LEN["seq_len: Sequence length"]
        MEMORY_BW_REQ["memory_bw: Required memory bandwidth"]
        COMPUTE_REQ["compute_req: Required compute capacity"]
    end
    
    subgraph "Performance Metrics"
        THROUGHPUT["Throughput: Images/second"]
        LATENCY["Latency: End-to-end time"]
        MEMORY_USAGE["Memory: GPU memory usage"]
        POWER_EFF["Power: Watts per inference"]
        EFFICIENCY["Efficiency: FLOPS/Watt"]
    end
```
