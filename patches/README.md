# Patches: Windows single-GPU CUDA build (no NCCL) + CUDA correctness fixes

This directory documents and reproduces the patch set that makes LightGBM's
CUDA build work on **Windows** for a **single GPU**, without NCCL (which has
no official Windows distribution), plus fixes for upstream CUDA bugs that
crash training or corrupt trees when any feature has more than 256 bins
(`max_bin > 255`). See the repository README for the summary.

Full documentation (Chinese): **[README.zh.md](README.zh.md)**

## What is patched

### Windows / toolchain enablement

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `find_package(NCCL REQUIRED)` → `QUIET`, falls back to the bundled stub library exposed as `NCCL::NCCL`; adds `-DFMT_UNICODE=0` to CUDA flags | NCCL is hard-required since #6138 but unavailable on Windows; every NCCL call site only runs with `num_gpu > 1`, so a stub is safe (it aborts loudly if ever invoked) |
| `cmake/nccl-stub/` | new — minimal `nccl.h` subset + aborting implementations | satisfies the compiler and linker; `ncclUniqueId` keeps its 128-byte layout |
| `src/boosting/cuda/nccl_gbdt.hpp` | `#include <pthread.h>` guarded with `!_WIN32` (unused include) | pthread does not exist on Windows |
| `include/LightGBM/meta.h` | `const` → `constexpr` for device-code constants | nvcc 13.x requires inlinable constants in device code |
| `src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` | explicit `static_cast<size_t>` in `std::max` | Win64 `size_t` vs 32-bit `unsigned long` breaks deduction on newer MSVC |
| `external_libs/fmt` submodule | `U""` char32_t literal → integer array | nvcc's EDG front end rejects large char32_t literals |

### Upstream CUDA bugs (max_bin > 255 territory)

Patches `0001-…scratch-buffers…` through `0004-…stale-nu…` (generated with
`git format-patch`); all are live on upstream `main` as of v4.7.0 + 58
commits. Re-apply them on every new baseline until upstream fixes them
(reported in microsoft/LightGBM#6512).

| Patch | Bug | Symptom |
|---|---|---|
| `0001` scratch buffers | global-memory split finder indexes scratch buffers as `hist_offset * 2 + bin` but allocated only `num_total_bin` | illegal memory access ([#6512](https://github.com/microsoft/LightGBM/issues/6512)) **or silent tree corruption** — depends on device VA layout |
| `0002` block-offset buffers | per-leaf `CalcBlockDim` can exceed the full-data grid (up to 80 blocks); offset buffers were sized from the full-data grid only | out-of-bounds writes in `PrepareOffset` |
| `0003` FixHistogram bins | one-thread-per-bin fix kernels drop every bin above `FIX_HISTOGRAM_BLOCK_SIZE` (512) | garbage histograms for `max_bin ≥ 512` (RMSE ~5e38, crashes) |
| `0004` candidate selection | strided gain loops keep the *last* qualifying bin instead of the thread's *best*; numeric kernels never reset `num_cat_threshold` | wrong split thresholds; intermittent host access violation via the categorical-bitset path |

Known remaining upstream issue (not fixed here, warned at runtime): with
heavily **zero-inflated** features and `max_bin > 255`, the histogram kept
for "larger" leaves can disagree with the routed data, producing degenerate
trees (empty-leaf chains) and, occasionally, a host-side crash. `max_bin
≤ 255` is unaffected and matches the CPU learner bit-for-bit.

## Reapplying after an upstream update

Baselines are upstream **stable release tags** (not main snapshots); fork tags
are named `<baseline>-cuda<cuda-version>[-rN]`. Steps, conflict points and the
build commands are described in [README.zh.md](README.zh.md).
