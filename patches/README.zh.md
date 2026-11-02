# LightGBM Windows CUDA 补丁集（绕开 NCCL）

## 背景

上游自 2026-03 的 #6138（CUDA 多 GPU 支持）起，在 `CMakeLists.txt` 中把
`find_package(NCCL REQUIRED)` 写死：即使单卡 CUDA 训练也强制要求 NCCL。
NCCL 官方没有 Windows 发行版，导致 Windows 上无法编译 CUDA 版。

本补丁集的思路是 **stub（桩）库**，而不是删改 NCCL 代码：

- LightGBM 源码中所有 NCCL 调用点都由 `num_gpu > 1` 守卫
  （见 `src/boosting/boosting.cpp` 的 `Boosting::CreateBoosting`：只有
  `device_type == "cuda" && num_gpu > 1` 才会实例化 `NCCLGBDT`），
  单卡训练时 `nccl_communicator_ == nullptr`，运行时永不进入 NCCL 分支。
- 因此只要让 NCCL 代码**编译链接通过**即可。补丁提供一份最小的
  `nccl.h` 桩头文件 + 桩实现库（被调用时直接 abort 并报错），
  运行时行为与真实单卡 NCCL 构建完全一致，且不会静默出错。

上游 main 分支（a6d48a60，stable-21，2026-09-02）验证通过。

## 补丁内容（分支 `cuda-no-nccl`，基于 main `a6d48a60`）

| 文件 | 改动 | 原因 |
|---|---|---|
| `CMakeLists.txt` | `find_package(NCCL REQUIRED)` → `QUIET`，找不到时把 `cmake/nccl-stub/` 编译为静态库并 `ALIAS` 成 `NCCL::NCCL`；CUDA flags 追加 `-DFMT_UNICODE=0` | 绕开 NCCL 硬依赖；nvcc 编 fmt 需关 Unicode 断言 |
| `cmake/nccl-stub/nccl.h` | 新增。LightGBM 实际用到的 NCCL 接口子集（`ncclUniqueId` 保持 128 字节布局，LightGBM 用 `sizeof` 做集合通信） | 提供 `nccl.h` |
| `cmake/nccl-stub/nccl_stub.cpp` | 新增。除诊断函数外全部 `abort()`（fail-loudly，误用多卡立即失败而非静默出错） | 链接占位 |
| `src/boosting/cuda/nccl_gbdt.hpp` | `#include <pthread.h>` 包 `#ifndef _WIN32`（该头实际未使用 pthread，Linux 行为不变） | Windows 无 pthread.h |
| `include/LightGBM/meta.h` | `kMinScore/kMaxScore/kEpsilon/kZeroThreshold` 由 `const` 改 `constexpr` | nvcc 13.x 要求 device 代码引用的常量可内联 |
| `src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` | `std::max(..., 1UL)` → 两侧显式 `static_cast<size_t>` | Win64 上 `size_t`(64位) 与 `unsigned long`(32位) 推导歧义，新版 MSVC 报错 |
| `external_libs/fmt` 子模块 | `format.h` 中 `U"\x9999999a..."` char32_t 字面量改为数值相同的整数数组 | nvcc 的 EDG 前端拒绝大值 char32_t 字面量（MSVC 无此限制） |

## 上游 CUDA 正确性修复（max_bin > 255 领地，v4.7.0.99-cuda13.3-r3 起）

上游 main 在任一特征 `num_bin > 256`（即 `max_bin > 255`，触发
GlobalMemory 分裂搜寻路径）时存在多个缺陷，补丁序列为
`patches/0001-…scratch-buffers…` ~ `0004-…stale-nu…`（`git format-patch` 生成，
在上游修复前每次 rebase 新基线都要重放）：

| 补丁 | 缺陷 | 症状 |
|---|---|---|
| `0001` scratch 缓冲 | GM 分裂搜寻内核按 `hist_offset*2+bin` 索引，但缓冲只分配 `num_total_bin` | 越界写：撞未映射页→illegal memory access（[#6512](https://github.com/microsoft/LightGBM/issues/6512)）；落对齐垫区→**静默污染**（退化成 1~3 叶树） |
| `0002` 块偏移缓冲 | 每叶重算的 `CalcBlockDim` grid 可超过全量值（小叶子最多 80 块） | `PrepareOffset` 越界写 |
| `0003` FixHistogram 丢 bin | 一线程一 bin 的修正内核丢弃 512 以上的 bin | `max_bin ≥ 512` 时直方图垃圾（RMSE ~5e38、崩溃） |
| `0004` 候选选择 | 跨步增益循环保留"最后合格者"而非线程最优；数值内核不重置 `num_cat_threshold` | 分裂阈值选错；陈旧字段诱发类别位图路径的间歇性主机崩溃 |

**遗留**（未修，运行时已加警告）：重度**零膨胀**特征 + `max_bin > 255` 时，
大叶子（直方图减法）的直方图可与实际路由数据错位，产生空叶链退化树，
偶发主机侧崩溃。`max_bin ≤ 255` 完全不受影响，与 CPU 逐位一致。

## 以后如何复用

上游更新后（`git pull`）重新打补丁：

```bash
cd LightGBM
git checkout main && git pull
git checkout cuda-no-nccl && git rebase main   # 冲突时参考下述冲突点
git submodule update --init --recursive
cd external_libs/fmt && git cherry-pick <fmt-patch-commit> && cd ../..
```

补丁文件在本目录（`git format-patch` 生成），直接 apply：

```bash
git am 0001-*.patch            # 主仓库补丁
cd external_libs/fmt
git am 0001-*.patch            # fmt 子模块补丁（单独一个）
```

### 可能的冲突点

- `CMakeLists.txt` 的 `find_package(NCCL ...)` 附近：上游若改了 NCCL 逻辑，
  手动套用"QUIET + stub 回退"结构即可。
- `meta.h` / `cuda_single_gpu_tree_learner.cpp`：上游若自行修复则放弃对应 hunk。
- fmt 子模块指针若升级，`U""` 补丁需在**新 fmt 版本**上重新评估
  （检查新 `format.h` 是否仍有 `fractional_part_rounding_thresholds`）。

## 基线与版本策略（2026-09 起）

1. **基线 = 上游稳定版 tag，不用 main 开发态。** 上游用 `stable` 标签跟踪
   最新稳定发布（当前 = `v4.7.0` = 8f7036f0）。今后更新时：

   ```bash
   git fetch upstream --tags
   git checkout cuda-no-nccl
   # 假设上游发布 v4.8.0，stable 标签随之移动：
   git rebase --onto stable <旧基线> cuda-no-nccl
   # 本仓首个基线（本次 r2 发布）例外：基于 main 开发态 a6d48a60（4.7.0.99）
   ```

2. **版本号约定（沿用上游官方约定，非自造）：**
   - `X.Y.Z` = 上游正式发布（VERSION.txt = `X.Y.Z`）
   - `X.Y.Z.99` = 发布后 main 开发态（如现在的 4.7.0.99，我们当前的 r2 即基于它）
   - 本仓 wheel 与 exe 随基线走：基于 `X.Y.Z.99` 的构建就报 `X.Y.Z.99`

3. **本仓 fork tag 命名 = `<上游基线>-cuda<CUDA版本>[-rN]`：**
   - `v4.7.0.99-cuda13.3-r2`（当前，基线为 main 开发态）
   - 下一次示例：`v4.8.0-cuda13.3`（基线 = stable/v4.8.0）

## 编译方式（Windows + VS2026 + CUDA 13.3，RTX 4090 = sm_89）

CUDA 13.3 没给 VS2026 装 VS 集成插件，**不能用 Visual Studio 生成器**，
必须 Ninja + vcvars 环境（脚本见仓库根：`config_cuda.bat` / `build_cuda.bat`）：

```bat
call vcvars64.bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
cmake --build build -j 12
```

Python 包（官方组装脚本，需 vcvars 环境 + `CMAKE_GENERATOR=Ninja` +
`CUDACXX` 指向 nvcc，见 `install_python_cuda.bat`）：

```bash
sh build-python.sh install --cuda --no-isolation
```

`build-python.sh` 会把源码树（含 `cmake/nccl-stub/`）组装进
`lightgbm-python/` 临时目录再构建 wheel，补丁自动生效。

## 已知限制

- 仅单 GPU（`num_gpu=1`）。误配多卡时训练启动会 abort 并提示这是 stub 构建。
- CUDA 版不支持稀疏特征（官方限制，与补丁无关）。
- AUC 等 metric 在 CPU 上评估（官方限制）。
