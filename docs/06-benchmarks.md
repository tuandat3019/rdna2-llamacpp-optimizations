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
| 64K | N3 | 283.6 | **31.38** | .77 | 15,748 / 392 MB | **light (~0.4 GB)** |
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
spill, 31 t/s); 128K runs but is spill-bound (17 t/s). Dual-card splits the weights and
KV across both GPUs and reaches 29.7 t/s at 128K.

## Dual RX 6800 + RX 6600 (layer-split 1.0,4.0)

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
