# Speculative decoding: MTP + ngram-mod tuning

The model carries a built-in MTP head (`nextn_predict_layers = 1`, in `blk.64`), used as
`--spec-type draft-mtp`. Combining it with **ngram-mod** (n-gram lookup drafts from the
prompt context) is the single biggest TPS lever on this hardware.

## Contribution breakdown (Swift @8K, dual card)

| Config | TG t/s | Δ vs off |
|---|---|---|
| spec OFF | 20.67 | — |
| MTP only (N9, pmin .82) | 45.42 | +120% |
| ngram-mod only (n-max 9, match 45) | 58.53 | +183% |
| **MTP + ngram (N9/.82/m45)** | **74.83** | **+262%** |

The two draft sources are complementary: MTP drafts from the model, n-gram drafts from
verbatim prompt context (code, quoted text, repeated structures).

## Parameters and how they behave

`--spec-draft-n-max` (**N**) — max MTP draft tokens per round.
`--spec-draft-p-min` (**pmin**) — stop drafting when the MTP confidence drops below this.
`--spec-ngram-mod-n-match` — n-gram lookup length (45 works well; 64 ≈ 45; 24 is −12%).
`--spec-ngram-mod-n-max` — **max draft tokens taken from an n-gram match** (independent
of N). This one has a very different optimum at 8K vs 128K.

### The depth (N) curve — 8K vs 128K differ dramatically

**8K:** deeper is better until a cliff:

| N | 1 | 4 | 5 | 9 | 12 | 16 | 24 | **32** | 40 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| TG | 36.0 | 55.7 | 59.0 | 74.8 | 91.3 | 91.4 | 101.3 | **107.1** | 53.2 | 63.6 |

(N ≥ 40 also collapses PP — the cliff is between 32 and 40.)

**128K:** the optimum is at the *bottom* of the range:

| N | 3 | 4 | 5 | 6 | 9 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| TG | **25.5** | 25.1 | 25.1 | 24.8 | 23.8 | 20.0 | 6.2 | PP-fail | PP-fail |

Why: at 128K every draft token must be verified against a ~128K KV cache; the FA cost per
verify batch grows with KV length, so deep drafts stop paying for themselves. And deep
**draft** n-max (≥ 24) additionally degrades *prefill* (PP collapsed 178 → 48–63 t/s and
kept falling — likely MTP draft-context buffers pushing the model into host memory), which
is why we separate the two n-max parameters in the scripts.

**pmin:** .50 beats .82 at 128K (25.5 vs 23.3 at N9); .85 is worse (24.2). Lower pmin =
longer effective drafts = more accepted tokens per round.

### The ngram n-max curve — the 128K lever

At 128K, the **ngram** n-max is a large, independent lever (draft N fixed at 3):

| ngram n-max | 6 | 9 | 12 | **24** | 32 |
|---|---|---|---|---|---|
| TG (warm) | 24.85 | 25.52 | 26.77 | **28.55** | 24.27 |
| acceptance | .85 | .79 | .85 | **.86** | .70 |

n-max 24 reproduced at **29.26 / 29.30** on repeat runs. Beyond ~24 the n-gram drafts
become low quality (acceptance .86 → .70) and the extra verify work is wasted.
Note: the upstream default for this parameter is **64**; the optimum here is 24.

### 64K optimum differs (2026-10-03 campaign, single 6800)

At 64K the KV is half as long, so longer n-gram drafts pay off and the peak moves up:

| ngram n-max (N3, ub512) | 9 | 24 | 32 | **40** | 41 | 44 | 48 |
|---|---|---|---|---|---|---|---|
| TG (warm) | 44.33 | 31.49 | 45.79 | **56.00** | 55.59 | 39.79 | 38.77 |
| acceptance | .91 | .77 | .90 | **.95** | .93 | .88 | .85 |

And the draft-depth curve is U-shaped around N3 (N2 42–44, N3 44–56, N4–N5 falling
to 33–39, N6 spill-skips, N9 trough 18.3, N12 partial recovery 24.0). N and NG
interact: N2 wins at NG24 but loses at NG40 (43.18 vs 56.00), so tune them jointly.
ub256 beats ub512 with shallow ngrams (40.22 vs 31.49 at NG24) but loses with deep
ones (41.04 vs 45.79 at NG32) — batch size and draft length interact too.

### MTP draft statistics (128K, N3)

- warm rep: ~44 verify rounds for 128 tokens; mean draft length 2.41, accepted 1.91/round,
  **2.91 tokens produced per round**; acceptance .86.
- cold rep acceptance is ~2/3 (MTP-only pattern; the n-gram index is still being built).

## Recommended production config

```
--spec-type draft-mtp,ngram-mod
--spec-draft-n-max 3          # 128K; use 32 for 8K-only workloads
--spec-draft-p-min 0.50
--spec-ngram-mod-n-match 45
--spec-ngram-mod-n-max 24     # the 128K sweet spot (upstream default 64 is too deep here)
```

With `KURAI_FA_NATIVE_NQ_MAX=40` so the FA kernel reads the q4_0 cache natively for
verify batches up to ntok ≈ 40.
