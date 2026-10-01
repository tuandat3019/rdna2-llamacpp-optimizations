# Hardware, build & measurement methodology

## Hardware

| Component | Detail |
|---|---|
| Primary GPU | **AMD RX 6800** 16 GB, gfx1030 (RDNA2), ~512 GB/s |
| Secondary GPU | **AMD RX 6600** 8 GB, gfx1032 (RDNA2), ~224 GB/s |
| CPU | Intel i5-11400F (6C/12T, AVX-512) |
| RAM | 32 GB DDR4 |
| OS | Windows 11 |
| Runtime | ROCm/TheRock clang (`D:\opt\rocm`), HIP |

**Important:** both GPUs share the same ROCm toolchain; a fatbin with **both**
`gfx1030` and `gfx1032` is mandatory — a build with only gfx1030 fails on the 6600
with `hipErrorInvalidKernelFile`.

**Display/Sunshine:** dwm + Sunshine occupy ~0.8–6 GB of VRAM depending on configuration
and streaming state. Sunshine's **video codec engine** activity measurably reduces decode
TPS (bandwidth contention) while leaving PP almost unaffected. All A/Bs in this repo were
run with a GPU-share preflight (see below).

## Build (Windows + ROCm)

```bat
set HIP_PATH=D:\opt\rocm
set HIP_PLATFORM=amd
cmake -S <src> -B <build> -G Ninja ^
  -DCMAKE_C_COMPILER="D:\opt\rocm\lib\llvm\bin\clang.exe" ^
  -DCMAKE_CXX_COMPILER="D:\opt\rocm\lib\llvm\bin\clang++.exe" ^
  -DCMAKE_CXX_FLAGS="-ID:\opt\rocm\include --offload-arch=gfx1030 --offload-arch=gfx1032" ^
  -DCMAKE_CROSSCOMPILING=ON -DCMAKE_BUILD_TYPE=Release ^
  -DGPU_TARGETS="gfx1030;gfx1032" -DBUILD_SHARED_LIBS=ON ^
  -DLLAMA_BUILD_TESTS=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF ^
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=OFF ^
  -DCMAKE_SYSTEM_NAME=Windows
cmake --build <build> -j4
```

Build rules learned the hard way:

1. **Never run two builds concurrently** — the DLLs silently fail to relink
   (wasted hours twice). Build serially, and verify the artifact:
   check DLL mtime + grep for expected strings in the produced DLLs.
2. Changing `CMAKE_CXX_FLAGS` (e.g., adding a `-D` macro) forces a full rebuild
   (~25 min); changing one .cu file is ~4 min.
3. `ccache` (`-DCMAKE_HIP_COMPILER_LAUNCHER=ccache`) is recommended; the stew675 project
   measured 282 s → 4.2 s on repeat builds.

## Measurement methodology

- **Bench harness:** a `llama-server` + an HTTP bench client that reports
  PP (prefill t/s), TG (decode t/s), MTP/ngram acceptance, and wall time per request;
  rep1 is cold (full prefill), rep2 is a prompt-cache hit (warm).
- **Kill + verify before every run:** stop all `llama-server` processes, verify 0 remain,
  wait for GPU memory release. A run with two servers competing produced 24.25 t/s for a
  config that actually does 80.33 t/s.
- **GPU-share preflight:** check for stray processes holding VRAM and for Sunshine codec
  activity. Decode is bandwidth-sensitive; prefill is not.
- **PP early-abort:** during 128K prefill, monitor the PP rate; if it drops below a
  threshold (150 t/s here), the config is failing — kill it immediately instead of
  waiting for a 30-minute timeout.
- **Noise:** run-to-run variance is ±5–8% at 128K on this machine (GPU shared with the
  desktop + remote-streaming codec). Repeat suspect configs and take the max
  (clean-state) value; never trust a single measurement.
- **Tie rule:** configs within 1% TG are ties → choose the one with lower VRAM usage.
