# EXPERIMENTAL: real NCCL on Windows (multi-GPU research)

> Status: **research branch, not for release.** Everything here is validated
> on a single RTX 4090 (WDDM) only. Cross-GPU data paths are **untested**.

## Goal

Evaluate whether a real NCCL can be linked on Windows to enable the upstream
`num_gpu > 1` path (`NCCLGBDT`), replacing the bundled abort-stub.

## NCCL provenance (verified programmatically)

Base: official `NVIDIA/nccl` tag `v2.32.3-1` (commit `12df1a11`). The official
tree already contains a complete Windows OS layer (`src/os/windows.cc` etc.).
The community mirror `WyvernTKC/nccl-windows` (HEAD `9edb3bdb`) adds exactly
6 commits / 7 files / +117 -21 lines on top of that tag:

| commit | change | origin |
|---|---|---|
| `6984de82` | C++14 static_assert fix (CUDA < 13 builds) | community |
| `77c14d6d` | yield-instead-of-sleep in async job waits (Windows timer granularity) | community |
| `f651e34a` | Windows build instructions (README) | community |
| `ab11d001` | fix SHM collisions between communicators | **NVIDIA staging `acc8d06c`** |
| `fefc8477` | fix crash when `NCCL_DEBUG_FILE` set | **NVIDIA staging `9d70ea67`** |
| `9edb3bdb` | README documents the two staging fixes | community |

Saved delta: `exp-nccl/nccl-windows-delta-official-v2.32.3-1.patch`.

## Build (local, CUDA 13.3, sm_89)

```
exp-nccl/scripts/build_nccl.bat    # cmake+Ninja+MSVC, CMAKE_CUDA_ARCHITECTURES=89
exp-nccl/scripts/build_smoke.bat   # single-GPU NCCL smoke exe
exp-nccl/scripts/run_smoke.sh      # runs smoke; plain + NCCL_DEBUG_FILE variants
```

Linking LightGBM against it: set `NCCL_ROOT=<exp-nccl>/install-nccl` before
CMake configure; `cmake/modules/FindNCCL.cmake` picks `nccl_static.lib`.
The CMake status line distinguishes:
- `NCCL not found - using bundled stub library (single-GPU CUDA only)` (default)
- `NCCL found - linking REAL NCCL on Windows (multi-GPU experimental)`

## What is validated locally (1x RTX 4090, WDDM, driver 621.01)

- [x] nccl 2.32.3 builds with MSVC + CUDA 13.3 + sm_89 (untested-by-author combo;
      artifacts: `nccl.dll` 35.7 MB, `nccl_static.lib` + `nccl_device.lib`, cmake package)
- [x] smoke: single-rank comm; AllReduce sum/max (1K/1M), Broadcast, grouped
      Send/Recv self-pair - **all max|err| exactly 0.000e+00**; dual communicator
      destroy order (staging-fix scenario) - no hang
- [x] smoke with `NCCL_DEBUG=INFO NCCL_DEBUG_FILE=...` (staging fix fefc8477) - PASS,
      debug log written; init log shows NET/Socket plugin, 64 channels, healthy
- [x] LightGBM (python wheel, scikit-build) builds linking real NCCL via
      `BUILD_WITH_SHARED_NCCL=ON` + `NCCL_ROOT`; `dumpbin /dependents` on
      `lib_lightgbm.dll` lists `nccl.dll`; runtime deployment = `nccl.dll`
      next to `lib_lightgbm.dll` (see Linking notes)
- [x] equivalence (2026-11-07, 20k x 7 synthetic regression, 20 rounds, seed 7):
      - **max_bin=255: cuda num_gpu=1 AND cuda num_gpu=2 (NCCLGBDT, clamped to
        1 GPU) both match CPU to 6 decimals** - rmse 91.252434, max|dpred| ~5e-07
        (float noise). NCCLGBDT attribution proven via `NCCL_DEBUG=INFO`
        (`NCCL version 2.32.3+cuda13.3`, `comm rank 0 nRanks 1`, ncclCommInitRank
        0.20 s) + the `Only 1 GPUs available, using num_gpu = 1` clamp warning.
      - max_bin=300: cuda diverges from cpu as a *legal variant* (known
        cross-thread tie-break ordering, same class as documented for single-GPU
        CUDA): rmse cpu 91.52 / ng1 83.30 / ng2 82.60; no crash, no garbage
        output; ng2 tracks ng1 (0.8% apart), both far from corruption signatures.
- [x] fail-safe: `num_gpu=2, gpu_device_id_list="0,1"` on a 1-GPU box ->
      graceful degradation, no crash/hang: warnings "Invalid GPU device ID 1 in
      gpu_device_list is ignored" -> "using 1 GPUs instead" -> training completes

## Linking notes (Windows, learned the hard way)

1. **Include path**: when real NCCL is found, `<nccl.h>` is reached only via the
   `NCCL::NCCL` IMPORTED target; objects that include it through `cuda_utils.hu`
   without linking the target fail with C1083. The branch therefore adds
   `include_directories(${NCCL_INCLUDE_DIR})` globally (mirroring the stub branch).
2. **Static link does not work out of the box**:
   - `nccl_static.lib` is built with `/MT`; the python extension links `/MD`
     -> `LNK2038 RuntimeLibrary mismatch`.
   - the static library alone is incomplete: device kernel tables
     (`ncclDevKernelList`, `ncclSymkKernelList`, `ncclDevFuncRowToId`, ...) live
     in `nccl_device.lib` ("the static library, which needs both files").
   -> the branch defaults `BUILD_WITH_SHARED_NCCL=ON` on WIN32 and links the
   `nccl.lib` import library instead; `nccl.dll` must then be reachable at
   runtime (PATH, or later: `os.add_dll_directory` in libpath + wheel payload).
3. **Runtime dll resolution**: CPython >= 3.8 loads `lib_lightgbm.dll` via
   ctypes with winmode flags that search **only** the DLL's own directory and
   system dirs - **PATH is ignored** (verified empirically: PATH containing the
   nccl dir still fails). Deployment recipe: ship `nccl.dll` **next to**
   `lib_lightgbm.dll` (site-packages/lightgbm/bin). `nccl.dll` itself is
   self-contained (CUDA runtime statically linked; only system DLL deps:
   KERNEL32/ole32/ADVAPI32/WS2_32/IPHLPAPI/SETUPAPI).

## What is NOT validated (needs real multi-GPU hardware)

- cross-GPU ring/tree allreduce over NVLink/PCIe P2P (the actual transport
  layer: Windows CUDA IPC absence, WDDM P2P behavior, channel scheduling)
- performance / correctness under 2+ ranks; `NCCLTopology` multi-node path
- original port author tested: 4x V100-SXM2 NVLink **TCC mode only**;
  WDDM (all consumer GeForce) is untested territory - our smoke test above
  is the first WDDM datapoint for the single-rank path.

## Fail-safe reminder

`NCCLTopology` clamps `num_gpu` to `cudaGetDeviceCount()` with a warning, but
`Boosting::CreateBoosting` selects `NCCLGBDT` from the *configured* value, so
`num_gpu=2` on a 1-GPU box runs the full multi-GPU architecture with a
single-rank communicator - useful for testing, not a substitute for 2 GPUs.
