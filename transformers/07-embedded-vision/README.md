# Embedded Vision Tutorial: Tiny Transformer on Microcontroller/Edge SoC

This tutorial walks through building and deploying a minimal vision transformer-like model for a 32×32 RGB image to predict a boolean output: contains human (True/False). We use dummy weights and focus on deployment steps with portability to microcontrollers (MCUs), edge SoCs, or SBCs.

## ✅ Target
- Input: 32×32×3 image
- Output: Binary: human present (1) or not (0)
- Model: Minimal ViT-inspired classifier (few patches + attention-lite)
- Deployment: Prototype on MCU/edge (TensorFlow Lite Micro, ONNX Runtime for ARM, or TVM)

## 🧭 Flow
1. Define a tiny vision transformer model (PyTorch)
2. Optimize for inference (quantization, fusing)
3. Export to on-device format (TFLite/ONNX)
4. Deploy on hardware (MCU or ARM SoC) with step-by-step guide
5. Validate with a test image

## 🏗️ Model Design (Minimal & Readable)
- 4×4 patches → 64 tokens for a 32×32 image
- Tiny attention-lite block (single head) + MLP
- Global token pooling → binary classifier
- Dummy weights for demo (replace with trained weights later)

## ⚡ Inference Optimizations
- Post-training quantization: INT8
- Operator fusion (MatMul + Activation)
- Model pruning (optional)
- Memory-aware execution: static tensors, arena buffers (MCU)

## 🔧 Tooling Options
- TensorFlow Lite Micro (MCU)
- TensorFlow Lite (Edge SoC, ARM)
- ONNX → ONNX Runtime (ARM)
- TVM micro (MCU) / TVM runtime (Edge)

## 🧪 Demo Files
- `tiny_vit_embedded.py`: Minimal model with dummy weights and exporters (ONNX/TFLite)
- `embedded_deploy_notebook.ipynb`: End-to-end deployment notebook
- `firmware_stub/`: C/C++ stub for TFLite Micro runtime

## 📦 Dependencies (Host)
- PyTorch, numpy
- onnx, onnxruntime (for ARM Linux)
- tensorflow / tflite-runtime (depending on path)
- tvm (optional, advanced)

## 🚀 Step-by-step

### 1) Create and export the model
- Define a tiny ViT-like network with PyTorch
- Export to ONNX and/or TFLite
- Apply INT8 quantization

### 2) Choose a deployment path
- MCU path: TFLite Micro (C++)
- Edge SoC path (Raspberry Pi, Jetson Nano): TFLite or ONNX Runtime
- TVM path: Auto-tune for target

### 3) MCU (TFLite Micro) Steps
1. Convert model → `model.tflite` (INT8)
2. Use TFLite Micro interpreter with a fixed tensor arena
3. Copy input (32×32×3, uint8) to input tensor
4. Invoke; read a single output float (sigmoid) or uint8

### 4) ARM SoC (ONNX Runtime) Steps
1. Export ONNX: `tiny_vit.onnx`
2. Install `onnxruntime` on device
3. Load model; run inference with NCHW input
4. Use `argmax`/threshold to get boolean

### 5) TVM Steps (optional)
1. Import ONNX → TVM Relay
2. Build for target (e.g. `llvm -mtriple=aarch64-linux-gnu`)
3. Run with TVM runtime

## 📚 References
- TFLite Micro: https://www.tensorflow.org/lite/microcontrollers
- ONNX Runtime: https://onnxruntime.ai/
- Apache TVM: https://tvm.apache.org/
- TinyML Book: Warden & Situnayake, 2019

---

See `tiny_vit_embedded.py` and the notebook for an end-to-end example including export and dummy inference on host. For real deployment, swap dummy weights for trained weights and retest quantization accuracy.
