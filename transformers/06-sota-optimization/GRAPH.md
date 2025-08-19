flowchart TD
    MODEL["Transformer Model"] --> TRAIN["Training Optimizations<br/>(AMP, Checkpointing, FSDP)"]
    MODEL --> INFER["Inference Optimizations<br/>(KV Cache, compile, BetterTransformer)"]
    MODEL --> QUANT["Quantization<br/>(PTQ/QAT)"]
    TRAIN --> SCALE["Distributed / ZeRO / FSDP"]
    INFER --> SERVE["Serving (vLLM / TRT-LLM)"]
    QUANT --> DEPLOY["Deploy"]
