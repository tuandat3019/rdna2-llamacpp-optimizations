# Raw benchmark tables

All numbers: Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP UD-IQ4_XS, KV q4_0, ubatch 512,
no graphs (HIP graphs off for 128K runs), MTP+ngram spec unless noted.
"warm" = rep2 (prompt-cache hit, steady state). PP = prefill t/s, TG = decode t/s.

## Single RX 6800 (16 GB) — the publishable configuration

Final config: band2048 (native KV everywhere), custom FA kernel OFF, ngram n-max 24, pmin .50, ub 512.

| Context | Draft N | PP | TG (warm) | Acceptance | VRAM peak (ded / shared) | Spill |
|---|---|---|---|---|---|---|
| 8K | N12 | 370.8 | **117.07** | 1.0 | 15,814 / 280 MB | minimal |
| 8K | N3 | 372.1 | 98.12 | 1.0 | 14,453 / 280 MB | minimal |
| 8K | N6 | 373.0 | 95.57 | .98 | 14,907 / 280 MB | minimal |
| 8K | N32 | — | PP-fail (115 t/s) | — | 16,331 / 2,708 MB | heavy |
| 64K | N3 + NG40 | 283.7 | **56.00** | .95 | 15,748 / 452 MB peak | **light (~0.4 GB)** |
| 64K | N3 + NG41 | 282.3 | 55.59 | .93 | 15,748 / 456 MB peak | identical VRAM to NG40 (diff is acceptance noise) |
| 64K | N3 + NG32 | 283.6 | 45.79 | .90 | ~15,7xx | light |
| 64K | N3 + NG24 (old default) | 283.6 | 31.38–31.49 | .77 | 15,748 / 392 MB | **light (~0.4 GB)** |
| 64K | N3 + NG44 / NG48 | 283.8 / 283.7 | 39.79 / 38.77 | .88 / .85 | ~15,6–15,7xx | light (past the NG peak) |
| 64K | N2 / N4 / N5 (+NG32) | ~283.8 | 44.48 / 39.33 / 32.75 | .88 / .67 / .72 | — | N6 spill-skips (no JSON) |
| 64K | N9 / N12 | ~283 | 18.34 / 24.01 | .52 / .65 | — | deep draft collapses (verify cost) |
| 64K | N3 + NG24, ub256 | 270.9 | 40.22 | .69 | 15,564 / 338 MB peak | smaller ub helps shallow ngram only |
| 128K | N3 | 188.1 | **17.20** | .81 | 14,986 / **2,824 MB** | **heavy (~2.8 GB)** |

### Why 64K and 128K spill — and why 128K collapses on a single card

The RX 6800 has 16.4 GB. The budget at each context:

| Item | 8K | 64K | 128K |
|---|---|---|---|
| Weights (UD-IQ4_XS) | 14.25 GB | 14.25 GB | 14.25 GB |
| KV cache q4_0 (18.4 KB/token) | 0.15 GB | 1.18 GB | 2.36 GB |
| Compute buffers + workspace | ~1.5 GB | ~1.6 GB | ~1.8 GB |
| **Total** | ~15.9 GB | **~17.0 GB** | **~18.4 GB** |
| **Over 16.4 GB?** | no | **+0.6 GB → spills to host RAM** | **+2.0 GB → heavy spill** |

The overflow does not fail — the driver places those allocations in host RAM
(GPU "shared" memory), and the model keeps working. But every decode token must read the
spilled KV/weights over PCIe: at 64K the penalty is small (31.4 t/s vs 8K's 117), at 128K
it is severe (**17.2 t/s**, less than half of what the same model does at 64K). Note the
spill shown is *peak*; part of the "shared" usage is by design (the host-side KV mirror
from `--cache-ram`), but at 128K the genuinely spilled portion dominates.

**Practical guidance:** on a single 16 GB card, 64K is the comfortable ceiling (light
spill, **56.0 t/s** with N3/NG40 spec tuning, acc .95, headroom 739 MB); 128K runs but is
spill-bound (17 t/s). Dual-card splits the weights and KV across both GPUs and reaches
29.7 t/s at 128K. At 64K single and dual are tied (~56 vs ~46–51): the 6600's copy
overhead cancels its VRAM relief, and dual 64K runs are less stable run-to-run than single.

## Dual RX 6800 + RX 6600 (layer-split 1.0,4.0)

### 2026-10-02 update — 144K champion + Vision

| Context | Split | PP | TG (median) | rep1 / rep2 TG | Acceptance | VRAM peak 6800 | Headroom | Spill |
|---|---|---|---|---|---|---|---|---|
| **144K** | 1.0,4.0 | **163.3** | **32.30** | 19.34 / 32.30 | .74 / .93 | 15,716 | 652 MB | none |
| **144K + Vision** | 1.0,4.0 | 160.5 | 30.93 | 18.86 / 30.93 | .74 / .93 | 15,716 | 652 MB | none |
| 128K | 1.0,4.0 | 185.4 | 29.30 | 19.69 / 29.30 | .67 | 15,288 | 1,080 MB | none |
| 128K + MLOCK | 1.0,4.0 | 185.4 | **29.70** | 18.18 / 29.70 | .65 | — | — | none |
| 160K | 1.0,4.0 | 155.5 | 18.18 | 18.13 / 18.18 | .76 | 16,144 | 224 MB | **yes — spill-bound** |
| 180K | **1.2,3.8** | 136.1 | 20.93 | 15.79 / 20.93 | .73 | 16,129 | 239 MB | light |

Notes:
- 144K (32.3 t/s) > 128K (29.3) on the same prompt family because the warm-rep acceptance
  is higher (.93 vs .67) — spec acceptance, not raw kernel speed, dominates measured TG.
- Vision (mmproj on the 6600) costs ~1.7% PP / ~4% TG in the text-only bench — inside the
  machine's ±5-8% noise band; the 6800 budget is untouched (same 15,716 MB peak).
- rep1 is measured right after a ~14-minute prefill (hot GPU, cold spec state); rep2 is the
  steady-state number.
- 160K at 1.0,4.0 genuinely spills (224 MB headroom < ~350 MB) and drops to 18.2 t/s.
  Re-splitting to 1.2,3.8 recovers it: 180K runs at 20.9 t/s with light spill.
- VRAM formula (validated within 1 MB): `ded_6800 ≈ 15,170 + (ctx−128K)/16K × 428 MB`,
  −703 MB with split 1.2,3.8. Real spill starts below ~350 MB of headroom.

### Dual @64K (2026-10-03 campaign, same Swift model, UNIQUE-64K ~53K tokens)

| Split | N / NG | PP | TG (warm) | Acceptance | 6800 peak / headroom |
|---|---|---|---|---|---|
| 1.0,4.0 | N13 / NG24 | ~239 | **50.61** | .84 | 15,427 / 1,061 MB — **but reruns at 33.15** (unstable) |
| 2,9 | N12 / NG24 | ~244 | **45.66** | .76 | ~15,5xx — stable best dual |
| 2,9 | N12 / NG9 | 244.1 | 35.50 | .83 | ~15,6xx |
| 2,9 | N12 / NG12–32 | ~244 | 24.0–24.8 | .50–.60 | NG24 is the peak on this split |
| 2,9 | N16 / N20 / N24 | ~243 | 26–29 | .54–.60 | peak 16,072 MB (headroom ~300 MB → spill-bound collapse) |
| 1.2,3.8 | N12 / NG24 | — | 39.59 | .82 | — |
| 1,8 / 1,7 / 2,7 | N12 / NG24 | 236–257 | 33.2 / 28.7 / 28.3 | .62 / .59 / .65 | 1,8 peaks at 16,098 (270 MB headroom → spill-affected) |
| 1,9 / 1,25 | N12 / NG24 | — | 12.5 / 24.2 | .58 / .65 | too tight — collapses |
| 1.0,4.0 | N3 / N6 / N9 / N12 | — | 26.0 / 24.0 / 33.5 / 34.8 | .72 / .58 / .67 / .65 | — |

Takeaway: pushing more layers onto the 6800 helps only up to 2,9 — tighter splits
spill the 6800 (peak >16 GB) and TG collapses. Dual N optimum at 64K is N12–13;
deeper drafts (N16+) collapse like at 128K.

Tensor-parallel re-tested 2026-10-03 (ratios 1:1 and 1:2, 8K and 64K): TG 4.11 t/s,
acceptance **0.000**, garbage output, server HTTP 500 — unusable with this hybrid
GDN model at any ratio. Prefill is also slower than layer-split (416s vs ~190s @64K).

### Vision (mmproj-BF16, 888 MB) at 144K

The vision tower offloads to the **RX 6600** (+779 MB dedicated), leaving the 6800 budget
untouched: 144K + Vision loads at 15,597 MB on the 6800 (headroom 771 MB), no spill, and
the text-only benchmark shows no measurable TG change (the tower idles for text input).
An image smoke test (text + shapes) exercises the actual vision path.


| Context | Config | PP | TG (warm) | Acceptance | Notes |
|---|---|---|---|---|---|
| 8K | N9 .82 m45 | 303 | 74.83 | .854 | |
| 8K | N12 | 301.9 | 91.27 | .809 | |
| 8K | N24 | 301.7 | 101.33 | .745 | |
| 8K | **N32** | 300.4 | **107.10** | .638 | 8K champion; recheck 106.51 |
| 8K | N32, ub256 | | | | (ub256 was tested at 128K, not 8K) |
| 128K | N9 .82 (pre-V4) | 173.5 | 21.38 | .863 | baseline before V4 |
| 128K | N9 .82 + V4 band8 | 178.4 | 22.49 | .863 | |
| 128K | N9 .82 + V4 band40 | 176.3 | 23.34 | .863 | |
| 128K | N3 .50 (ngram 9) | 185.0 | 25.52 | .786 | depth floor wins |
| 128K | N4 .50 | 184.3 | 25.14 | .762 | |
| 128K | N5 .50 | 185.3 | 25.06 | .752 | |
| 128K | N6 .50 | 183.1 | 24.77 | .721 | |
| 128K | N3 .50 + ngram n-max 12 | 183.0 | 26.77 | .846 | |
| 128K | N3 .50 + ngram n-max 24 | 185.5 | **29.26 / 29.30** | .864 | **champion, reproduced ×2** |
| 128K | N3 .50 + ngram n-max 32 | 182.1 | 24.27 | .705 | beyond the peak |
| 128K | N3 .85 | 185.1 | 24.22 | .874 | pmin .50 wins |
| 128K | N3 .50, match 64 | 177.9 | 25.49 | .786 | tie with 45 |
| 128K | N3 .50, match 24 | 180.5 | 22.49 | .727 | −12% |
| 128K | N4 .50, ub256 | 175.9 | 17.88 | .602 | **ub256 rejected (−29%)** |
| 128K | N9 .82, split 1,9 | 59.8 falling | — | — | killed: 6800 spills 3 GB |

## Baseline evolution (dual card, 128K, long varied prompt)

| Build / model | TG | PP | Acceptance |
|---|---|---|---|
| Coletti, stock-ish (E02) | 18.90 | ~142–185 | .859 |
| Coletti + ngram-mod (E27 at 8K; E40 at 128K) | 26.37 | 184.7 | .952 |
| Swift + V4 + N3/.50 + ngram 24 (current) | **29.26** | 185 | .864 |

End-to-end: **+55% TG at 128K** over the starting baseline on the same hardware.

## VRAM reference (dual card, 128K champion)

| | RX 6600 (8 GB) | RX 6800 (16 GB) |
|---|---|---|
| llama-server dedicated | 4,325 MB | 15,764 MB |
| llama-server shared (host RAM) | ~1,400 MB | ~1,300 MB |
| Spill composition | KV mirror / host cache (by design) | mostly KV + buffers |

At 128K with split 1.0,4.0 the 6800 runs close to its limit; each +16K of context adds
~295 MB of q4_0 KV (16 attention layers × 18.4 KB/token). The 6600 has ~2.4 GB headroom
which a re-split (e.g. 1.2,3.8) could use for higher context — at some TG cost.
