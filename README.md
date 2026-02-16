# 🚀 Optimized-VGGT: High-Throughput 3D Geometry Reconstruction

[![Hardware: NVIDIA A40](https://img.shields.io/badge/Hardware-NVIDIA%20A40%20(48GB)-green.svg)](https://www.nvidia.com/en-us/data-center/a40/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.x-orange.svg)](https://pytorch.org/)
[![Optimization: FlashAttention-2](https://img.shields.io/badge/Optimization-FlashAttention--2-blue.svg)](https://github.com/Dao-AILab/flash-attention)

A systems-focused optimization of the **Visual Geometry Grounded Transformer (VGGT)**. This project identifies and resolves scaling bottlenecks in multi-frame 3D reconstruction, specifically targeting **NVIDIA Ampere (A40)** architecture.

---

## 🔍 The Problem: The "Scaling Wall"
In standard VGGT implementations, processing large image sequences (100+ frames) causes a massive latency spike. My profiling via **NVIDIA Nsight Systems** revealed:
1. **Kernel Launch Overhead:** Processing frames sequentially ($B \cdot S$ as separate items) led to the GPU being "starved" while waiting for the CPU to launch 100+ separate kernels.
2. **Global Redundancy:** Early transformer layers performed expensive Global Attention before establishing local geometric priors.



---

## 🛠️ Performance Optimizations

### 1. Fused Batch Frame Attention
Instead of iterating through frames, I refactored `aggregator.py` to treat the sequence length ($S$) as part of the batch dimension.
* **Old Way:** 100 sequential calls to the GPU.
* **Optimized:** 1 fused batch call processing 100 images in parallel.

### 2. Early Layer Pruning (Global → Frame)
Based on research intuition, I implemented **Layer Pruning** in the first 4 blocks of the Aggregator. 
* **The Logic:** Global attention is converted to Frame attention in early stages to save VRAM and compute, as cross-frame correspondences aren't stable until deeper in the network.

### 3. FlashAttention-2 Enforcement
Modified `attention.py` to guarantee the use of **SDPA (Scaled Dot
