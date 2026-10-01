# The full optimization log (measured)

Every experiment in this project, with before/after numbers, conditions and the verdict.
IDs (E00, E16, …) are stable references used throughout the docs.

Measurement rules used for every number below:

- **One server at a time.** Kill + verify zero `llama-server` processes before each run
  (two servers competing for VRAM once produced a completely wrong result: 24.25 vs 80.33 t/s).
- **GPU-share preflight.** Check for stray processes holding VRAM and for Sunshine's video
  codec engine activity; decode TPS is sensitive to bandwidth contention, PP is not.
- **Cold + warm rep.** rep1 = full prefill (cold), rep2 = prompt-cache hit (warm). The warm
  rep is the steady-state number; the cold rep includes one-time graph-capture cost (~15%).
- **No profilers during timing runs.**
- **Tie rule:** if two configs are within 1% TG, pick the one using less VRAM.

---

## 1. Baseline & methodology

| ID | What | Result | Verdict |
|---|---|---|---|
| E00 | Baseline production (short prompt) | TG 28.18 t/s, acc .8085 | baseline |
| E02 | 128K baseline (long varied prompt) | **TG 18.90 t/s, acc .8593, prefill 708 s** | baseline for all A/B |
| E03c | Control build vs production | −0.26% = noise | toolchain OK |
| E05 | Roofline | 91 GB/s effective = 17% of theoretical BW; 4400 graph nodes × 5–8 µs = 22–35 ms (40–63% of token time) | **launch/sync-bound, not ALU-bound** |
| E10 | Context sweep | TG 31.2 (2K) → 27.0 (8K) → 21.2 (32K) → 18.9 (99K); PP 240 → 142 | FA is target #1 |
| E13 | Noise floor over time | ±0.2% same prompt | stable |
| E14/E15 | Per-op profiler | decode @128K: **FA 62.5%, MMQ 25%, rest 12.5%** | FA confirmed as #1 |

## 2. Speculative decoding: MTP + ngram-mod (the biggest lever)

The model has a built-in MTP head (blk.64). `--spec-type draft-mtp,ngram-mod` combines
MTP drafts with n-gram lookup drafts.

**8K (CODE_8192 prompt, dual card):**

| Config | TG t/s | Notes |
|---|---|---|
| spec OFF | 20.67 | |
| MTP only (N9, pmin .82) | 45.42 | +120% |
| ngram-mod only (n-max 9, match 45) | 58.53 | ngram alone beats MTP alone |
| **MTP + ngram (N9/.82/m45)** | **74.83** | **+262% vs off** |
| MTP+ngram N12 | 91.27 | |
| MTP+ngram N16 | 91.36 | |
| MTP+ngram N24 | 101.33 | |
| **MTP+ngram N32** | **107.10** | 8K champion |
| N48 / N64 / N96 | 63.6 / 31.1 / 42.1 (PP also collapses) | cliff between 32 and 48 |
| match 64 / match 24 | 74.96 / 64.55 | match 45 ≈ 64 > 24 |

**128K (UNIQUE-128K prompt, dual card) — the optimum is DIFFERENT:**

| Config | TG t/s (warm) | Notes |
|---|---|---|
| N9, pmin .82 (pre-V4) | 21.38 | |
| N9, pmin .82 + V4 band8 | 22.49 | |
| N9, pmin .82 + V4 band40 | 23.34 | V4 total +9.2% |
| N9, pmin .50 | 23.80 | |
| N4, pmin .50 | 25.14 | |
| **N3, pmin .50** | **25.52** | depth floor |
| N6/N12/N16 @128K | 24.77 / 19.97 / 6.17 | deeper = worse (verify cost ∝ KV length) |
| N24/N32 @128K | PP collapse (48–63 t/s) | **draft n-max is the PP killer, not ngram** |
| pmin .85 (N3) | 24.22 | pmin .50 wins |
| ngram n-max 6 / 9 / 12 / **24** / 32 | 24.85 / 25.52 / 26.77 / **28.55** / 24.27 | **n-gram length is a 128K lever!** |
| ngram n-max 24 repeat | **29.26 / 29.30** | reproducible |

**Why ngram n-max matters more at 128K:** the n-gram drafts are high-quality verbatim
continuations; longer allowed drafts (up to 24 tokens) increase accepted tokens per verify
round at almost no MTP compute cost. Beyond ~24, draft quality collapses (acceptance
.86 → .70) and the extra verify work is wasted.

**Final champion (dual, 128K):** `draft-mtp,ngram-mod` N3 pmin .50 ngram match 45 n-max 24,
band2048, layer 1.0,4.0 → **29.26–29.70 t/s**, acceptance .86–.89.

**Single card is different:** on a single RX 6800 the depth optimum moves *up* again —
no split overhead, so deeper drafts pay off:

| Single-card 8K, ngram 24 | N3 | N6 | **N12** | N32 |
|---|---|---|---|---|
| TG | 98.1 | 95.6 | **117.1** | PP-fail (spill) |

**ubatch sweep @128K (dual):** 256 → 17.9 t/s (−29%), **512 → 29.3 (best)**, 640 → 19.2
(−34%), 1024 → prefill collapse. 512 is the exact sweet spot.

## 3. V4: native quantized KV in the FA tile kernel (port of stew675 block 15)

| ID | What | Result |
|---|---|---|
| E65 | V4 ported (q4_0/q8_0 native dequant inside the tile loaders), band gate = decode/verify only | @8K: neutral (106.5 vs 106.5) — expected: conversion cost ∝ n_kv |
| E65 | Discovery: the band gate (n_q ≤ 8) **did not engage** for our spec config (verify batch = 1+NMax > 8) | made band tunable: `KURAI_FA_NATIVE_NQ_MAX` |
| E66 | V4 @128K with band8 | **+5.2%** (21.38 → 22.49) |
| E66 | V4 @128K with band40 (verify batches native) | **+9.2%** (21.38 → 23.34) |
| E66 | Numerics | bit-identical acceptance (acc .8625 identical) |

Also from the same block: the prefill path keeps the staged F16 copy (native prefill was
promising for PP — 258→201 t/s observed once — but crashed in this fork; not shipped).

## 4. Graph fusions

| ID | What | Result |
|---|---|---|
| E57 | ADD + RMS_NORM + MUL fused into one kernel (port of jstamagal R7) | TG +0.4%, **PP +7.7%** (394.8 vs ~366), gate `KURAI_NO_NORM_FUSION=1` to disable |
| E29 | Skip HIP graphs for multi-token prefill, keep per-width decode graphs (stew675 block 11) | neutral on this setup @128K; kept |

## 5. GPU split & VRAM

| ID | What | Result |
|---|---|---|
| E26 | Layer-split tuning: push work to the 6800 | 1,4 → 1,19 = +10.8% TG / +17.3% PP (Coletti 8K) |
| E31 | Split sweep @8K + ngram | 1,25 best (84.79) but later corrected (see E43) |
| E40/E40-VRAM | 128K split 2,9 spills: 6800 16,195/16,368 MB + 623 MB shared; **dwm 4.9 GB + Sunshine 1.5 GB occupy the 6600** | spill confirmed |
| E42/E43 | Two overlapping bench jobs produced a wrong 1,4 number (24.25); corrected: **1,4 = 80.33 @8K, dedicated 17,561 MB, shared 0** | kill+verify rule |
| E44 | 128K VRAM delta vs 8K = +2.2 GB = KV 2.3 GB; constant overhead ~3.3 GB | no hidden overhead |
| E55 | 128K bottleneck = 6600 overflow (needs 4.9 GB, ~1.4 GB free) → TG 27.4 | use 1,4 |
| E60 | split 1,9 @128K: 6800 spills 3 GB → PP collapses 175 → 59.8 t/s, killed | 1,9 rejected |
| E66e | ubatch 256 @128K: spill 2553 → 1500 MB but **TG 17.9 (−29%)** | spill is not the bottleneck; ub256 rejected |
| E79 | ubatch 640/1024 @128K | 640: TG 19.2 (−34%); 1024: prefill collapse → **ub 512 is the exact optimum** |
| E70b | `--load-mode mmap+mlock` (model pinned in RAM) | **29.70 t/s — highest raw number**, plus pageout protection; candidate for production |
| E70b | `--swa-checkpoints` off | 29.45 — tie; keep checkpoints for production (they help real long sessions with context trimming) |
| E66e | band2048 (native prefill, no staging scratch): PP ~258→201 t/s observed, −472 MB, but crashed at 83% prefill in this fork | not shipped |

## 6. Everything else that was tried

| ID | What | Result | Verdict |
|---|---|---|---|
| E18-MMVQ | MMVQ nwarps sweep (community kernel-anvil style) | 28.82 (default) vs 28.66/27.73/23.90 | RDNA2 prefers nwarps=1; all changes lose |
| E18-TENSOR…E56 | Tensor-parallel (allreduce.cu enabled for HIP, 2 upstream bugs fixed: 2D memcpy no-op, graph re-capture) | correct output achieved, but: no P2P → barrier 380 µs → 49.9 ms/token in barriers alone; **uneven split corrupts hybrid-GDN output** (matches upstream issue #29501) | **abandoned** |
| E20 | Barrier floor microbench | event 380 µs, CPU poll 143 µs, doorbell 123 µs | physics of no-P2P Windows |
| E21 | MTP parameter sweep @128K | MTP OFF = 18.38 vs ON = 28.93 (+56%) | MTP is essential |
| E22 | External draft models (0.8B/2B) | 14–17 t/s, acceptance collapse | use the built-in MTP head |
| E25 | Mirrored MTP layer (avoid draft allreduce) | crash, then 28.65 → 22.85 | reverted |
| E27 | **ngram-mod introduced** | 50.37 → **87.96 (+74.6%)** 1-card @8K | kept — biggest single win |
| E28/E30 | `GGML_CUDA_NO_PEER_COPY` (community warning) | −75% dual, −82% single @128K (it also disables the event path) | hardware-specific; do NOT blanket-apply |
| E36 | 1-card vs 2-card | 1-card 6800 is faster at 8K (no split overhead); 2-card needed for 128K VRAM | both useful |
| E57/E58 | Model switch Coletti → Swift-1.5 (65 blocks, 256K native ctx) | Swift 1-card 83.1 @8K, 2-card 74.4 | current model |
| E59 | r22 getenv caching port | neutral (fork already cached) | kept (harmless) |
| E61–E64 | Spec matrix @8K (see §3) | N32 champion 8K | |
| E66–E73 | 128K tuning (see §3) | N3-p50 + ngram n-max 24 = 29.3 | |
| E75/E76 | Repeats & noise study | same config twice: 25.52 vs 23.80 (−7%) → ±5–8% run-to-run noise from GPU sharing; champion N24 reproduced 28.55 / 29.26 / 29.30 | measure twice, take max |
| E77 | **Single 6800** | 8K **97.97** (PP 371.7, spill 280 MB); 64K **32.85** (PP 290, spill 392 MB); N32 fails (spill 2.7 GB) | single-card configs are excellent |
| E78 | Single 6800 deep-draft probe | see `docs/06-benchmarks.md` | |
| E65 env sweep | HSA_NO_SCRATCH_RECLAIM, NO_PEER_COPY, 2D_GATHER, FORCE_UPDATE | 106.60 / 103.56 / 105.77 / 104.87 vs 106.55 → all ties | defaults kept |
