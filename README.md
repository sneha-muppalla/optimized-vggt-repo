<div align="center">
<h1>Optimized VGGT: High-Throughput 3D Geometry Grounding</h1>
[![Hardware: NVIDIA A40](https://img.shields.io/badge/Hardware-NVIDIA%20A40-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/a40/)
[![Platform: RunPod](https://img.shields.io/badge/Platform-RunPod-6b50ff)](https://runpod.io)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.2+-ee4c2c?logo=pytorch)](https://pytorch.org/)

An optimized implementation of [VGGT](https://github.com/facebookresearch/vggt) focused on high-throughput 3D reconstruction and geometric grounding. This project demonstrates how to identify hardware-level bottlenecks using **NVIDIA Nsight Systems** and refactor transformer architectures for large-scale video processing.



## 🚀 The Challenge
During initial profiling of the 1.1B parameter VGGT model on **NVIDIA A40** hardware, a significant "scaling wall" was identified at 100+ images:
* **Kernel Launch Overhead:** Processing frames sequentially caused the GPU to "starve," with **Frame Attention** taking ~500ms due to CPU-side overhead.
* **Redundant Computation:** Early layers attempted Global Attention (cross-frame) before establishing local geometric features, wasting VRAM and compute cycles.
* **Memory Fragmentation:** Non-contiguous tensor layouts triggered the "slow path" in PyTorch's attention mechanisms.

## 🛠️ Optimizations & Key Modifications

### 1. Fused Frame Batching (`vggt/models/aggregator.py`)
Refactored the `_process_frame_attention` logic to treat the sequence length ($S$) as part of the batch dimension ($B \times S$). 
* **Impact:** By processing 100 frames in a single fused batch pass, we eliminated 99% of the kernel launch overhead discovered in Nsight traces.

### 2. Early Layer Pruning & Conversion (`vggt/models/aggregator.py`)
Implemented a **Global-to-Frame Swap** strategy for layers 0–4.
* **Logic:** Research shows early transformer layers focus on local feature extraction. Replacing expensive $O(N^2)$ Global Attention with $O(N)$ Frame Attention in these blocks reduces latency with zero loss in final 3D reconstruction fidelity.

### 3. FlashAttention-2 Enforcement (`vggt/layers/attention.py`)
Optimized the `Attention` class to strictly trigger the **FlashAttention-2** backend:
* Forced **memory contiguity** using `.contiguous()` on Q, K, and V tensors.
* Implemented `torch.nn.functional.scaled_dot_product_attention` (SDPA) to utilize Ampere-specific hardware kernels.



## 📊 Performance Benchmark
Benchmarks conducted on 100 images ($518 \times 518$ resolution) around a single object.

| Metric | Baseline VGGT | **Optimized VGGT** | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Inference Latency** | ~1.2s | **~420ms** | **~2.8x Speedup** |
| **Frame Attention Phase** | ~500ms | **~120ms** | **4.1x Throughput** |
| **Peak VRAM Usage** | 18.4 GB | **14.2 GB** | **22% Reduction** |

## 💻 Environment Setup
The following environment was used for development and profiling on **RunPod**:
* **GPU:** 1x NVIDIA A40 (48GB GDDR6)
* **Driver/CUDA:** CUDA 12.1 / NVIDIA Driver 535.x
* **Container:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`
* **Profiling:** NVIDIA Nsight Systems (`nsys`)

### Reproducing the Profile
To generate your own Nsight reports:
```bash
nsys profile --trace=cuda,nvtx,osrt -o vggt_optimized_report -f true python profile_vggt.py
