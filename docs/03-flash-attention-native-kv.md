# Native quantized KV cache in the flash-attention tile kernel (V4)

*Ported from stew675's `llama-cpp-rdna-boosts` block 15 ("campaign memory wins", issue #30).*

## The problem

With a quantized KV cache (q4_0 here — f16 KV does not fit at 128K on 16 GB), the stock
FA tile kernel **stages a whole-cache F16 copy** for each FA call: it dequantizes the
entire used K/V region to F16 scratch, then reads that scratch. The conversion pass is
proportional to `n_kv` and runs **on every decode/verify step** — at 128K that is
hundreds of MB per layer per step, plus the scratch reservation itself (hundreds of MB
of VRAM sized by `n_ctx`).

The stock **VEC** kernel already reads q4_0 natively, but it is only selected for very
small batches. The **TILE** kernel (used for everything ≥ 3 tokens — including every
speculative-verify batch) pays the conversion.

## The fix

Teach the TILE kernel to dequantize **while staging K/V into shared-memory tiles**:
a 16-byte staged chunk = 8 elements = exactly a quarter of a q4_0/q8_0 block, so an
8-element chunk never straddles a block boundary. The dequantized values are
bit-identical to the F16 scratch (single F16 rounding of the exact product).

Two design points from stew675 that we kept:

1. **Band split.** Dequantizing per tile costs ~`n_q * n_kv / ncols`, which amortizes at
   prefill but not at decode — so prefill (`n_q > band`) keeps the staged path and only
   the decode/verify band reads the raw cache.
2. **One predicate, three consumers.** The choice of native-vs-staged must be computed
   by the *same* function in the dispatch, in the scratch-size calculation and in the
   launcher, or the kernel reads raw cache while the launcher still staged (NaN).

## Our port (TILE-only, q4_0 + q8_0)

Files touched in our fork:

| File | Change |
|---|---|
| `fattn-common.cuh` | policy (`GGML_CUDA_FA_KV_NATIVE` 3-state), support predicates, dequant chunk helpers, band predicate `ggml_cuda_fattn_tile_kv_native_type(K, V, Q)` |
| `fattn-tile.cu` | dispatch per compile-time KV type (`GGML_TYPE_F16` / `Q4_0` / `Q8_0`), band split inside the shared helper |
| `fattn-tile.cuh` | `type_KV` template axis, native tile loader (dequant per 16-byte chunk), launcher passes `need_f16 = (K->type != type_KV)` |
| `fattn.cu` | `get_alloc_size` uses the same helper so the scratch exists iff the launcher stages |

The core chunk dequantizer (q4_0; q8_0 analogous):

```cpp
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q4_0_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    const int blk = el / QK4_0;
    const int off = el % QK4_0;
    const char * bp = row + (size_t) blk*sizeof(block_q4_0);
    half d_h; ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    const float d  = __half2float(d_h);
    const float dm = -8.0f*d;
    const int lo   = off < QK4_0/2;
    const int base = lo ? off : off - QK4_0/2;
    const uint8_t * qs = (const uint8_t *) (bp + sizeof(half));
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float v0 = d * (lo ? (b0 & 0x0F) : (b0 >> 4)) + dm;
        const float v1 = d * (lo ? (b1 & 0x0F) : (b1 >> 4)) + dm;
        dst[l] = make_half2(__float2half(v0), __float2half(v1));
    }
}
```

## Results

| Test | Before | After | Notes |
|---|---|---|---|
| 8K dual, N32 spec | 106.51 | 106.55 | neutral (conversion is cheap at 8K — ∝ n_kv) |
| 128K dual, N9 spec, band 8 | 21.38 | **22.49 (+5.2%)** | partial engagement |
| 128K dual, N9 spec, band 40 | 21.38 | **23.34 (+9.2%)** | full engagement for verify batches |
| Acceptance | .8625 | .8625 | **bit-identical numerics** |
| VRAM (band 2048 variant, prefill native) | — | **−472 MB** (dedicated) | from removing the F16 staging scratch |

## The band gate — the trap we hit

stew675's band gate is `n_q ≤ 8` (their verify widths). Our speculative config uses
`verify batch = 1 + draft n-max`; at N9 that is up to 10 tokens, so most verify batches
**silently fell back to the staged path** and the port initially showed only +5.2%.
We made the band tunable at runtime:

```
KURAI_FA_NATIVE_NQ_MAX=40   # default 8 (stew675); 40 covers verify batches up to n-max 39
```

With the band at 40, the full +9.2% materializes. If you run speculative decoding with
deep drafts, remember to raise the band or you leave the win on the table.

## What we did NOT port (and why)

- **MMA native arm** — RDNA2 has no WMMA; the tile kernel is the only relevant path.
- **bf16 KV** — requires `v_dot2_f32_bf16` (gfx11+/RDNA3+); RDNA2 falls back to a slow
  scalar path.
- **The prefill arena** — stew675 moves the prefill staging scratch to a per-context
  arena to reclaim n_ctx-sized VRAM. We observed native-prefill PP around 258→201 t/s
  with the band lifted high (i.e. native prefill looked *faster* on RDNA2), but it
  crashed in our fork at ~83% of a 107K-token prefill and we did not chase it further.
  If you want the VRAM back at high context, this is the place to look.
