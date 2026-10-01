# ARCHIVE — Custom FA decode kernel (E16/E17) — kept for reference, NOT used

> **Status: archived, disabled.** This kernel was developed in this project, reached
> impressive microbenchmark numbers, but does **not** improve production TPS on the
> final configuration. It is documented here for the record. Do not enable it unless
> you are specifically experimenting with `ntok ≤ 2` FA decode.
> Enable/disable: `KURAI_FA_CUSTOM=1` / `=0` (default in the final build: disabled).

## Why it existed

With a quantized KV cache, the stock FA path staged a whole-cache F16 copy on every
decode step for batches with `n_tokens > 2` (the TILE kernel), which at 128K is a large
per-step cost. The stock VEC kernel reads q4_0 natively but only for `n_tokens ≤ 2`.
The custom kernel was meant to provide a fast native-q4_0 path.

## Design

- 1 block per (query row, query head, K-split); 128 threads (4 × wave32)
- Q row into LDS; K/V q4_0 staged into LDS with dequant fused into the dot loop
- online softmax; split-K with a fixup kernel; generic mask
- shape gate: D = 256, KV q4_0, Q F32, contiguous, etc.
- gate: `KURAI_FA_MAX_NTOK` (compile-time; final value 2 → hybrid)
- dozens of `KURAI_FA_*` micro-switches (BK, TP, SKIPAO, DPPFUSE, …) for A/B work

## Microbenchmark results (isolated kernel, kv=111112, toks=1)

| Version | 6600 | 6800 |
|---|---|---|
| Correct baseline | 8.17 ms | 4.10 ms |
| After micro-opt campaign | 2.90 ms (−64%) | 1.51 ms (−63%) |
| After E17 (SKIPAO + DPPFUSE) | **2.13 ms (−26% further)** | — |

Notable wins: u64 nibble-window (−20%), LDS transpose to kill bank conflicts (−41%),
DPP-modifier reduce fusion (−15.3%, bit-exact), mask prefetch (−9%), `__expf` (−9%).

## Production results (why it is archived)

| Test | Result |
|---|---|
| Custom for all ntok (early) | **6.55 t/s vs 18.79 stock — 2.9× slower** |
| Custom vs stock, clean GPU | 95% @32K, 90% @128K |
| Root cause | stock TILE processes **256 KV columns/loop** with quantized `vec_dot` (4 MAC/instr); the custom kernel does 1 row/loop with scalar dequant (1 MAC/instr) — architectural |
| Hybrid (custom ntok ≤ 2 only), old config | +2.05% @32K, +0.89% @128K |
| **Final re-test (E71) on the production config** | **custom OFF 29.54 t/s vs custom ON 29.26–29.30 t/s → no gain** |

The final config (ngram n-max 24) produces wide verify batches (mostly ntok > 2), so the
custom path almost never engages — and when it does, the saving is below noise.

## Lessons worth keeping

- **The right fix was V4** (native quantized KV inside the *stock* tile kernel), not a
  new kernel: +9.2% @128K versus the hybrid's +0.89%.
- Three separate false-PASS checker bugs (bad f16 fill, NaN-blind compare, `#if` on a
  template parameter) — validate the checker before trusting results.
- `__builtin_amdgcn_readlane` needs a uniform lane-id; a VGPR id silently lowers to
  `v_readfirstlane` and produces wrong results.
- **Device-side `printf` in a kernel caused a GPU hang → driver timeout → production
  500s.** Never ship debug printing in kernels.
- The `volatile` LDS experiment produced a +145% result that was pure scheduling
  artifact — it "proved" a theory that was later disproven.

## Files (in the working fork, not part of this documentation repo)

- `ggml/src/ggml-cuda/kurai-fa-kernel.cuh` (~41 KB device code)
- `ggml/src/ggml-cuda/kurai-fa-dev.cuh` (host split calc)
- `ggml/src/ggml-cuda/kurai-fa-decode.cu` (hook into `fattn.cu`)
- `ggml/src/ggml-cuda/kurai-fa-decode.cuh`
- microbench harness: `mfb/fa-bench.cu` + sweep/validation scripts
