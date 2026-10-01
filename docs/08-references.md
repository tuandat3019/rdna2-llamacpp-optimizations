# References & sources

The optimizations in this repo were built on top of, or inspired by, the following
public sources. Numbers quoted from them are marked as such (their hardware, not ours).

## Primary patch sources

| Source | URL | What we took |
|---|---|---|
| **stew675 / llama-cpp-rdna-boosts** | https://github.com/stew675/llama-cpp-rdna-boosts | 16-block patch series: **V4 native q4_0/q8_0 FA (block 15)**, skip-graphs-for-prefill (block 11), meta-buffer headroom 16→128 (block 09), r22 getenv caching, adaptive MTP (block 01 — we use plain ngram-mod instead), BF16 KV (block 03 — not applicable to RDNA2), chunked GDN prefill (block 02 — not ported) |
| **jstamagal / llama.cpp-rdna2** | https://github.com/jstamagal/llama.cpp-rdna2 | RDNA2-specific fork: **ADD+RMS_NORM+MUL fusion (R7, ported)**, Q4_0 DOT8 MMVQ, native tiled FA arithmetic, Q8_1 activation reuse (not ported), chunked GDN gfx1030 (reference only) |
| **charlie12345 / windows-amd-vllm-multigpu** | https://github.com/charlie12345/windows-amd-vllm-multigpu | Windows multi-GPU reality check (no P2P on Windows; D3D12 cross-adapter fences are system-memory); `NO_PEER_COPY`/`--no-mmap` warnings — which proved **hardware-specific** (see below) |
| **JohnTDI-cpu / llama-hip-p2p-allreduce** | https://github.com/JohnTDI-cpu/llama-hip-p2p-allreduce | Direct-P2P HIP allreduce design (requires P2P — not available to us) |

## Upstream llama.cpp

| Source | URL | Note |
|---|---|---|
| PR #22299 (internal AllReduce) | https://github.com/ggml-org/llama.cpp/pull/22299 | The `allreduce.cu` we enabled for HIP (+51.3% TG on 2×RTX5090 — number from the PR) |
| PR #24152 (SYCL tensor-parallel) | https://github.com/ggml-org/llama.cpp/pull/24152 | Proof that tensor-parallel can win without P2P (+78.6% TG on 2×Intel Arc) — we could not reproduce this for hybrid-GDN models (see lessons) |
| docs/multi-gpu.md | https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md | Tensor mode limitations (quantized KV not supported, no hybrid/SSM support) |
| discussion #27082 | https://github.com/ggml-org/llama.cpp/discussions/27082 | Qwen3.8-27B on dual R9700: Direct-P2P +9.2% TG, Phase-13 F32 Q6_K accumulation +16.4% PP |
| `fattn.cu`, `fattn-tile.cuh`, `fattn-vec.cuh` | https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda | FA dispatch (`need_f16_K/V`), TILE config (nbatch_K=256, nbatch_fa=32 — the architectural reason our custom kernel loses for wide batches), VEC native quantized reads |

## Community / hardware documentation

| Source | URL | Note |
|---|---|---|
| Reddit: Qwen3.8-Flash-Next on 12 GB VRAM | https://www.reddit.com/r/Qwen_AI/comments/1wrk63y/ | PLE/ngram-embedding streamed from SSD (`--lazy-mode`), `--n-cpu-moe`, and the "RAM-resident loading gives 1.87× prefill vs mmap" claim (tested in our queue; not applicable to dense models for the MoE parts) |
| AMD RDNA2 ISA reference | https://www.amd.com/en/support/tech-docs | DPP modifiers on `v_add_f32` — source of the E17-DPPFUSE −15.3% micro-optimization |
| LLVM AMDGPU backend docs | https://llvm.org/docs/AMDGPUUsage.html | `__builtin_amdgcn_readlane` requires a uniform lane-id (explains the OPT8 failure) |
| Microsoft: D3D12 shared heaps | https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps | Cross-adapter resources are system memory only (no VRAM P2P) |
| ROCm issue #787 + Phoronix | https://github.com/ROCm/ROCm/issues/787 | Linux-only P2P (large-BAR / IOMMU requirements) — not portable to Windows |

## Hardware-specific warnings (things that did NOT transfer)

- **`GGML_CUDA_NO_PEER_COPY=ON`** is recommended on Windows R9700 pairs — on our
  6800+6600 it costs **−75% (dual) / −82% (single @128K)** because it also disables the
  event path. Measure before applying.
- **`--no-mmap`** does not exist in this llama.cpp revision; the equivalent is
  `--load-mode none`.
- **MMVQ nwarps sweeps** that help RDNA3/4 lose on RDNA2 (default nwarps=1 is optimal).
