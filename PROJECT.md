# Project: A11-local_language_model-optimizer (A11-LLM-optimizer)

## Overview & Goal
A low-level, high-throughput on-device C++20 and CUDA 12.x/13.x acceleration engine designed for running quantized LLMs (GGUF/FP8/INT4) on RTX 5070 Ti (12GB VRAM) and 32GB system RAM with NVMe direct tensor streaming and speculative sampling.

## Technical Stack & Toolchain
- **Language & Standards:** C++20, CUDA C++ (`nvcc`), CMake 3.26+, Python 3.11+
- **Host Compiler:** MSVC (Visual Studio 2022 Build Tools)
- **Target Hardware:** NVIDIA GeForce RTX 5070 Ti (12GB VRAM), Intel Core Ultra 9, 32GB System RAM
- **Python Environment:** `C:\Users\Aum Namaha\Documents\finalenv`
- **Testing & Quality Assurance:** Catch2 (C++ unit testing), pytest (PyTorch verification), ruff, black
- **Architecture Dependencies:** GGUF Parser, Win32 Overlapped I/O API, CUDA Driver API

## System Architecture
```mermaid
graph TD
    A[GGUF Model File on NVMe] -->|Win32 Overlapped mmap| B[12GB Host RAM Hot-Cache]
    B -->|Async PCIe Gen5 DMA Transfer| C[12GB RTX 5070 Ti VRAM]
    C --> D[CUDA FP8/INT4 Tensor Core Kernels]
    D --> E[Paged KV-Cache Allocator]
    E --> F[Tree-Based Speculative Verifier]
    F --> G[Generated Token Output]
Implementation Phases
[x] Phase 1: Repository Structure & GGUF Metadata Parser Setup

[ ] Phase 2: Asynchronous Windows NVMe Tensor Streaming & Host Hot-Cache

[ ] Phase 3: Paged KV-Cache & Custom CUDA FP8/INT4 Quantized MatMul Kernels

[ ] Phase 4: Tree Speculative Sampling & Performance Benchmarking Engine

[ ] Phase 5: Catch2 Test Verification & CI/CD Push

Command Reference
Python Environment Activation:& "C:\Users\Aum Namaha\Documents\finalenv\Scripts\Activate.ps1"

CMake Configure & Build (PowerShell):cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Releasecmake --build build --config Release

Run C++ Test Suite:.\build\bin\a11_tests.exe

Run PyTorch Cross-Validation:"C:\Users\Aum Namaha\Documents\finalenv\Scripts\python.exe" -m pytest tests/

Deployment & Repository Boundaries
Local Directory: C:\Users\Aum Namaha\Documents\Proj\A11-local_language_model-optimizer

Primary Remote: GitHub (git push origin main)

Secondary Remote: GitLab (git push gitlab main --force-with-lease)

Credentials Location: C:\Users\Aum Namaha\Documents\tempo.ipynb (Strictly Read-Only)