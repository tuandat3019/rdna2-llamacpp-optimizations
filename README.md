# RDNA2 llama.cpp Optimizations (gfx1030 / gfx1032, Windows + ROCm)

Practical, measured optimizations for running large hybrid-SSM LLMs (Qwen3.8-27B-class,
GDN + full-attention, IQ4_XS) on **AMD RDNA2** consumer GPUs under **Windows + ROCm/HIP**,
using a patched llama.cpp.

> Everything here was measured on real hardware, with an explicit methodology
> (kill+verify before every run, GPU-share preflight, repeat-on-suspect runs).
> Optimizations that did not survive A/B are documented too — including the
> dead ends, because they are as useful as the wins.

---

## Headline results

**Model:** `Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP` UD-IQ4_XS (14.25 GB, 65 blocks,
48 recurrent GDN + 17 attention, GQA 24/4, head_dim 256, MTP head in blk.64, native ctx 256K)
**KV cache:** q4_0 both K and V (f16 KV does not fit at 128K on this hardware)

| Context | Single RX 6800 (16 GB) | Dual RX 6800 + RX 6600 (16+8 GB, layer-split) |
|---|---|---|
| 8K  | **97.97 t/s** (PP 371.7) | 107.10 t/s (2-card) |
| 64K | **32.85 t/s** (PP 290.0) | — |
| 128K | 25.9 t/s (previous model; Swift pending) | **29.26 t/s** (PP ~185) |

Baseline before this optimization series (same hardware, stock-ish build, Coletti 27B IQ4_XS):
**18.9 t/s @128K** → after: **29.3 t/s @128K** (dual) — a **+55% end-to-end gain**,
and **98 t/s @8K** on a single 6800.

---

## The optimizations (summary — details in `docs/`)

| # | Optimization | Source | Effect (measured) |
|---|---|---|---|
| 1 | **Native q4_0/q8_0 KV in the FA tile kernel (V4)** — no whole-cache F16 staging per decode step | stew675 block 15 (ported) | **+9.2% TG @128K** (dual), VRAM −472 MB |
| 2 | **Speculative decoding tuning: MTP draft depth + ngram-mod + n-gram length** | this project | **+74.6% (8K)** vs MTP-only; **ngram n-max 24 = +12%** over n-max 9 @128K |
| 3 | **FA decode custom kernel (hybrid, ntok ≤ 2)** | this project (E16/E17) | +2.05% @32K / +0.89% @128K (hybrid) — see `docs/04` |
| 4 | **ADD + RMS_NORM + MUL fusion** | jstamagal R7 (ported) | +0.4% TG, **+7.7% PP** |
| 5 | **Layer-split tuning (1.0,4.0) + VRAM spill analysis** | this project | 1,9 @128K collapses (PP 175→60); 1,4 is the safe split |
| 6 | **Skip HIP graphs for multi-token prefill, keep for decode** | stew675 block 11 (ported) | neutral here @128K, kept |
| 7 | **getenv() caching on hot paths** | stew675 r22 (ported) | neutral here (fork already cached most) |
| 8 | **Meta-buffer compute headroom 16 → 128** | stew675 block 09 (ported) | fixes hybrid-SSM state crash |
| 9 | **ngram-mod + MTP combination** | stew675 block 01 (ngram-mod) + this project | the single biggest TPS lever on this hardware |

**Dead ends (documented so you don't repeat them):** tensor-parallel on Windows without P2P
(barrier 380 µs → unusable), `GGML_CUDA_NO_PEER_COPY` (−75..−82% here), MMVQ nwarps sweeps
(RDNA2 prefers nwarps=1), ubatch 256 at 128K (−29%), deep MTP drafts at 128K
(verify cost scales with KV length), custom FA kernel for ntok ≥ 3 (architecture:
stock TILE processes 256 KV columns/loop).

---

## Repository layout

```
README.md                     — this file
docs/01-hardware-methodology.md
docs/02-optimizations.md      — the full measured table (every experiment, E00→E78)
docs/03-flash-attention-native-kv.md
docs/04-speculative-decoding.md
docs/05-gpu-split-and-vram.md
docs/06-benchmarks.md         — all raw measurement tables
docs/07-lessons-learned.md
docs/08-references.md
```

## Hardware used

- **RX 6800** 16 GB (gfx1030) — primary GPU
- **RX 6600** 8 GB (gfx1032) — secondary (dual-card configs)
- Intel i5-11400F, 32 GB RAM, Windows 11
- ROCm/TheRock clang toolchain, fatbin `gfx1030;gfx1032` (both targets are mandatory —
  a missing gfx1032 fatbin causes `hipErrorInvalidKernelFile`)

## Build

See `docs/01-hardware-methodology.md` for the exact cmake line, and the build rules
(sequential builds only, artifact verification via mtime + strings in the DLLs).
