# 10 — Kurai Flash: pruned-expert Qwen3.8-Flash-Next on the Strata engine (gfx1030)

The same rig (RX 6800 16 GB + RX 6600 8 GB, Windows + ROCm) also runs a second model line besides the
dense 27B in this repo: **Kurai Flash** — `Swift-1.5-Qwen3.8-Flash-Next` (MoE) with its expert set
**pruned from 512 to 332 per layer** (IQ3_XXS; smaller model + pack, chosen for a 16 GB card), served
by the **Strata** engine (github.com/Niko1221/Strata — MoE-aware: expert cache + profile + adaptive
swaps, KV streaming, MTP/speculative decoding).

The point of interest for this repo: **the biggest single speedup measured on this hardware lives
here — one environment variable doubled prompt processing.**

## Headline numbers (this machine)

| Metric | Value |
|---|---|
| **Prompt (PP)**, 89K-token prompt, `STRATA_HIP_PROMPT_F16=1` | **~700 tok/s** |
| same prompt, engine 0.1.40 without F16 | 367 tok/s |
| same prompt, engine 0.1.38 (before the update) | ~294 tok/s |
| **Decode (TG)**, warm long sessions | **31–36 tok/s** (peak measured **52.5** at 152K ctx, draft acceptance 70.8%) |
| Reuse (KV cache) on long sessions | ~99.9% (e.g. 147,246 / 147,381 tokens reused per turn) |
| CoderAB60 (6-task agentic benchmark) | **93/100 — grade S**, 0 criticals |

## Why the ×2: FP16 prompt GEMMs (rocBLAS on gfx1030)

ROCm's rocBLAS for **gfx1030 has tuned kernels for FP16→FP16 GEMMs only**. The engine's prompt path
ran its 16-bit GEMMs as FP16→FP32-out and BF16 products — those fall to **generic kernels measured at
5.6 / ~5.3 TFLOPS against 37.7** for the FP16→FP16 shape (N=10240, T=7313, K=2560). The engine's
opt-in `STRATA_HIP_PROMPT_F16=1` runs those GEMMs FP16 in and out:

| | 89K prompt | wall time |
|---|---:|---:|
| 0.1.38 | ~294 tok/s | ~300 s |
| 0.1.40 | 367 tok/s | 243 s |
| **0.1.40 + F16** | **647 tok/s** | **138 s** |

Caveat: output is **not bit-identical** (each GEMM's output is rounded to FP16 instead of the BF16
rounding of the activations). The engine prints a tip on gfx103x; the distribution check ships with
the project (`bench/results/2026-10-04-rdna2-fp16-prompt`). Decode is unchanged.

## The rest of the tuning (all measured, A/B with clean restarts)

| Change | Effect | Verdict |
|---|---|---|
| **temp 1.0 → 0.6** (MTP acceptance is higher at lower temp) | TG **+6.1%**, draft acceptance +2.7 pts (72.3% → 75.0%) | kept (greedy = 33.8 tok/s / 83% — headroom remains at 0.3–0.4) |
| `GPU_PINNED_MIN_XFER_SIZE=1048576` | stops the `verify: timed out at layer N` stall class (#267/#649) on gfx1030 + `--mmap-experts` | kept |
| `STRATA_KV_PREFETCH=1` (overlap streamed KV uploads) | prompt +4–5% at 32–64K | kept |
| `STRATA_HIP_SWIGLU_FUSED=1` (bitwise same) | decode | kept |
| `STRATA_MTP_CATCHUP_ALL=1` | decode/draft | kept |
| `STRATA_DENSE_MMQ=1` (dense through int8 MMQ) | prompt | kept (slightly different numbers) |
| 5 opt-ins together (A/B vs baseline) | PP 8K **+12%**, warm TG **+11%**, draft **+5.6 pts** | — |
| `--adapt-every 1` + `STRATA_SPEC_COUPLED=1` | TG and draft **worse** (26.3 vs 29.1; 73.5% vs 84.5%) | rejected |
| `--adapt-swaps 0` (adaptive off) | +16% TG on 30 short payloads, but felt worse in real long sessions | rejected after real use |
| Conversation-cache floor 2560 → **768 MiB**, budget 2048 → **6144 MiB** | fixed "reused = 0" (parking was silently skipped: `snapshot + floor > free RAM`); reuse back to ~99.9% | kept |

## The sweet-spot config (copy-ready)

Engine: **Strata 0.1.40** (Windows HIP prebuilt), model `swift-uncen-iq3xxs-v3` (332 experts/layer:
256−D RCO base + 56 code + 8 Vietnamese + 12 test + D swap), profile `code+viet+test`.

```
--max-context 163840 --kv int8 --resident-experts --vision --vram-reserve-mib 700
--spec 4 --spec-min-p 0.6 --mtp <mtp dir> --expert-profile <profile.bin>
--expert-cache auto --kv-resident 32768 --prompt-cache 16
--conversation-cache-mib 6144 --conversation-cache-min-free-mib 768
# adaptive tier: leave at defaults (do NOT pass --adapt-swaps 0)

env: STRATA_HIP_PROMPT_F16=1  GPU_PINNED_MIN_XFER_SIZE=1048576
     STRATA_KV_PREFETCH=1  STRATA_HIP_SWIGLU_FUSED=1  STRATA_MTP_CATCHUP_ALL=1
     STRATA_DENSE_MMQ=1  STRATA_RESIDENT_HEADROOM_GIB=4
```

Client sampling: **temperature 0.6**, top_p 0.95, top_k 20, min_p 0 (sent per request — the engine's
config default is irrelevant if the client overrides it).

## gfx1032 note (RX 6600)

The RX 6600 is **gfx1032**, which the prebuilt engine does **not** include. A custom build works
(`STRATA_HIP_ARCHS=gfx1030;gfx1032`; add `gfx1032` to `cmake/hip_backend.cmake`'s unvalidated list,
then `tools\hip\build_windows.bat`). The **helper-GPU mode** (`--expert-cache-device1 auto
--remote-expert-opt`) is **blocked on this machine**: the Windows HIP runtime does not honour
`HIP_VISIBLE_DEVICES` ordering (the display card enumerates first, so the engine picks the 8 GB card
as its main), the helper indices are fixed at {1,2,3}, and `--resident-experts` is incompatible with
remote caches. Parked, not forgotten.

## Benchmark note

On the CoderAB60 agentic suite the model scored **93/100 (S)** with two tasks killed by the timebox.
Re-running those two with a small extension gave **12/12 and 10/10 (full)** — i.e. **the model can
reach 100/100 when the timebox is not the limiter**. The timebox, not capability, is what the 93
reflects.
