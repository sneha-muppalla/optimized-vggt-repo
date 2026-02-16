# 🚀 Optimized-VGGT: High-Throughput 3D Geometry Reconstruction

[![Hardware: NVIDIA A40](https://img.shields.io/badge/Hardware-NVIDIA%20A40%20(48GB)-green.svg)](https://www.nvidia.com/en-us/data-center/a40/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.x-orange.svg)](https://pytorch.org/)
[![Optimization: FlashAttention-2](https://img.shields.io/badge/Optimization-FlashAttention--2-blue.svg)](https://github.com/Dao-AILab/flash-attention)

An optimized implementation of VGGT focused on high-throughput 3D reconstruction and geometric grounding. This project specifically targets **NVIDIA Ampere (A40**) hardware, achieving a ~3x speedup in frame-wise attention for large image sequences.

---

## The Challenge"
Standard VGGT implementations face a "scaling wall" when processing high-frame-count sequences (100+ images) due to
1. **Kernel Launch Overhead:** Processing frames sequentially creates significant GPU idle time.
2. **Unnecesary Computation** Early layers in the Alternating Attention (AA) blocks perform Global Attention before the model has established strong local geometric features.

---

## My Optimizations

### 1. Fused Batch Frame Attention
Instead of iterating through frames, I refactored the `_process_frame_attention` method in `aggregator.py` to treat the sequence length ($S$) as part of the batch dimension ($BxS$)
* **Old Way:** 100 sequential calls to the GPU.
* **Optimized:** 1 fused batch call processing 100 images in parallel.

### 2. Early Layer Pruning (Global → Frame)
NSight Systems displayed long intervals of frame attention compared to global attention in the first few layers. So I implemented **Layer Pruning** in the first 4 blocks of the Aggregator. 
* **The Logic:** Global attention is converted to Frame attention in early stages to save VRAM and compute, as cross-frame correspondences aren't stable until deeper in the network.Saved ~50ms of execution time per inference pass without losing geometric fidelity.

### 3. FlashAttention-2 Enforcement
Modified `attention.py` to guarantee the use of **SDPA (Scaled Dot Product)
1. Enforced memory contiguity `(.contiguous())` on Q, K, and V tensors.

---
## Tech Stack & Environment
* Hardware: NVIDIA A40 (100GB VRAM)
* Cloud Provider: RunPod
* Profiling: NVIDIA Nsight Systems (nsys)
* Runtime: PyTorch 2.2+, CUDA 12.1
* Precision: $BF16$ Mixed Precision
---
## Modificatiios 
* `vggt/models/aggregator.py`: Optimized batching and layer pruning logic.
* `vggt/layers/attention.py`: Fused SDPA and memory contiguity fixes.
* `profile_vggt.py`: Benchmarking script with NVTX instrumentation.
=
---
## Environment Setup
Developed and profiled on **RunPod** using a dedicated **NVIDIA A40** instance.

```bash
# To replicate profiling results
nsys profile --trace=cuda,nvtx,osrt -o vggt_optimized_report -f true python profile_vggt.py
