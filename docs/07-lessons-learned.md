# Lessons learned (including the expensive ones)

## Measurement discipline

1. **Never bench with two servers running.** Two `llama-server` processes competing for
   VRAM produced 24.25 t/s for a config that actually runs at 80.33 t/s. Kill + verify
   zero servers before every single run.
2. **Verify your checker before trusting any PASS.** Three separate "passing" test
   harnesses were found broken: a bad f16 `memcpy` in the reference filler, a NaN-blind
   comparison (`NaN > max` is false — NaNs were silently ignored), and a `#if HAS_MASK`
   guarding a *template* parameter (always false → dead-code elimination → false PASS).
   Every early "bit-exact" claim had to be re-derived.
3. **GPU sharing is the "slow state".** Stray router child processes holding VRAM and
   Sunshine's video codec engine can cost 20%+ on decode while leaving prefill intact.
   Preflight before A/B; decode is bandwidth-sensitive, prefill is not.
4. **Run-to-run noise at 128K is ±5–8%.** The same config measured 25.52 and 23.80 on
   different runs. Repeat suspect configs; take the max as the clean-state value.
5. **PP early-abort saves 30-minute timeouts.** Monitor the prefill rate; a failing
   config shows it within the first minute (rate collapses and keeps falling).

## Architecture & hardware truths

6. **No P2P on Windows.** `hipDeviceCanAccessPeer = 0`; the cheapest cross-GPU barrier
   costs ~123–380 µs. Tensor-parallel needs ~128 barriers/token → 15–50 ms/token of pure
   synchronization. Layer-split (sequential layers per card) is the only viable split.
7. **Tensor-parallel also breaks hybrid-GDN correctness.** Uneven splits corrupt the
   recurrent (GDN/SSM) state slices: 1,4 loops, 2,8 emits empty output, 1,9 stops after
   one token — with or without speculative decoding. Matches upstream issue #29501.
8. **The FA tile kernel processes 256 KV columns per loop with quantized `vec_dot`
   (4 MAC/instruction).** A hand-written kernel doing 1 row/loop with scalar dequant
   (1 MAC/instruction) cannot close that gap — 90% at 128K at best. The right fix is
   teaching the *stock* kernel to read the quantized cache natively (V4), not writing
   a new kernel. (Our custom kernel survives only as a hybrid for ntok ≤ 2.)
9. **Quantized KV staging cost is ∝ n_kv.** It is invisible at 8K and large at 128K.
   Always validate KV-related optimizations at high context — the 8K numbers will show
   "neutral".
10. **Draft depth does not transfer across context lengths.** 8K wants N=32; 128K wants
    N=3–6. Verify cost scales with KV length; deeper drafts at 128K lose more than they
    gain. The same applies to ngram n-max — but with the opposite optimum (24 at 128K
    beats 9 by +12%), because n-gram drafts are cheap and high-quality.
11. **Deep MTP draft n-max (≥24) destroys prefill at 128K** (PP 178 → 48–63 and falling,
    ending in timeouts). The ngram n-max can go deep safely — separate the two knobs.
12. **Small ubatch is not "safer".** ub256 reduced spill (2553 → 1500 MB) but cost 29%
    TG. Spill is not the primary bottleneck; batch efficiency is.
13. **Community warnings are hardware-specific.** `NO_PEER_COPY=ON` (recommended for
    R9700) costs −75..−82% on 6800+6600. MMVQ nwarps tweaks that help RDNA3/4 lose on
    RDNA2. Always A/B on your own silicon.
14. **VRAM spill is real but context-dependent.** At 128K the 6800 sits at its limit;
    a 1,9 split spills 3 GB and collapses PP. 1.0,4.0 keeps spill manageable. The 6600's
    headroom is unused in this split — a re-split can trade TG for context headroom.

## Build & tooling

15. **Never build twice concurrently** — DLLs silently fail to relink (cost: hours).
    Verify mtime + expected strings in the DLLs after every build.
16. **Device-side `printf` in a kernel caused a GPU hang → driver timeout → production
    500s.** Never leave debug printing in kernels.
17. **llama.cpp filters INFO logs by default** (`LOG_DEFAULT_LLAMA=3 < LOG_LEVEL_TRACE=4`).
    Two wrong conclusions were drawn from "missing" log lines; use `-lv 5` when
    diagnosing.

## What we would do next (if continuing)

- Port **chunked GDN prefill** (jstamagal gfx1030 reference) for PP.
- Investigate the **native-prefill crash** at high band values — it looked *faster*
  (PP ~258→201 t/s) and would remove ~470 MB of staging scratch.
- Port **KQ-mask-derived** (stew675 block 15) — modest VRAM + prefill wins.
- Re-split experiments for 192K+ context on dual card (6600 headroom).
