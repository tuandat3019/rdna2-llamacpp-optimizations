# GPU split, VRAM budget & spill behavior

## The budget (Swift-1.5, q4_0 KV, dual card)

| Item | Size |
|---|---|
| Weights (UD-IQ4_XS) | 14.25 GB |
| KV cache q4_0 | **18.4 KB/token** → 2.36 GB @128K, 2.95 GB @160K, 4.72 GB @256K |
| Compute buffers + FA staging | ~2–3 GB at 128K (includes the F16 staging scratch when staged) |
| Desktop (dwm + Sunshine) | ~0.8–6 GB depending on which GPU drives the display and streaming state |

The RX 6600 (8 GB) is the tightest resource when it also drives the display and Sunshine:
in earlier sessions dwm (4.9 GB) + Sunshine (1.5 GB) left only ~1.4 GB free on it.

## Split experiments (128K, dual)

| Split | Result |
|---|---|
| 2,9 (old production) | 6800 spills 623 MB; PP 184.7, TG 26.4 (Coletti era) |
| 1,25 | spills **3.2 GB** — rejected |
| **1.0,4.0** | dedicated 17,561 MB total, shared 0 @8K; @128K 6800 peak ded 15,764 + ~1.3 GB shared — **the safe split** |
| 1,9 @128K | 6800 spills 3 GB → **PP collapses 175 → 59.8 t/s** (killed) |
| 1,19 (8K era) | +10.8% TG / +17.3% PP vs 1,4 — but only viable at low context |

At 8K the splits are within ~6% of each other (80.3–84.8 t/s, Coletti era); at 128K the
split choice is dominated by whether the 6800 overflows.

## Spill behavior

- "Shared" GPU memory on these cards is **host RAM used by GPU allocations**. Part of it
  is by design: the server keeps a host-side KV mirror (`--cache-ram 10240`) for prompt
  caching; part is genuine overflow.
- Genuine overflow is visible in the numbers: at 128K with split 1,9 the 6800 shared
  usage hits ~3 GB and prefill collapses. With 1.0,4.0 the shared usage stays ~1.2–1.9 GB
  and performance is stable.
- **Reducing spill is not automatically faster**: ubatch 256 cut spill by ~1 GB but cost
  29% TG (batch efficiency dominates). The V4 native KV path removes ~470 MB of staging
  scratch at zero performance cost — that is the *right* way to save VRAM.

## Context headroom

Per +16K context: **+295 MB** of q4_0 KV. Current 128K champion configuration leaves the
6600 with ~2.4 GB of unused headroom, so a re-split (moving a few more layers to the
6600, e.g. 1.2,3.8) should push the practical ceiling toward ~192K at some TG cost.
Single-card 6800 runs 64K with only ~390 MB of spill and 128K with ~1.5 GB (estimated).

## Single vs dual card

| | Single 6800 | Dual (1.0,4.0) |
|---|---|---|
| 8K | **97.97 t/s** (PP 371.7) | 107.10 t/s (PP ~300) |
| 64K | **32.85 t/s** (PP 290) | — |
| 128K | ~26 t/s (previous model) | **29.26 t/s** |

Single card wins at 8K by ~60% for the same spec config (no split overhead, no
cross-GPU copies); dual card wins where VRAM is needed (128K) and for absolute 8K
throughput with deeper drafts. For publication and reproducibility, the single-card
numbers are the cleanest.
