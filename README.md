<img src=https://github.com/lightgbm-org/LightGBM/blob/main/docs/logo/LightGBM_logo_black_text.svg width=300 />

> [!IMPORTANT]
> ## Windows CUDA builds for LightGBM
>
> A community fork of LightGBM carrying a minimal patch set that makes
> **CUDA builds work on Windows**: single-GPU training is stable and shipped
> as prebuilt binaries on every release, and **multi-GPU training
> (`num_gpu > 1`) is available experimentally** through a real NCCL built on
> Windows (see the experimental section below).
>
> Why the patch set exists: upstream requires the NCCL library for *all*
> CUDA builds since #6138, but NCCL has no Windows distribution. Release
> builds bundle a tiny NCCL **stub** library (`cmake/nccl-stub/`) that
> satisfies the linker and aborts loudly if ever invoked — no upstream code
> is removed. See `patches/README.md` (English) and `patches/README.zh.md`
> (中文) for details.
>
> **Two prebuilt variants** are published by GitHub Actions on every `v*` tag
> (see the [Releases](https://github.com/neoblackxt/lightgbm-cuda-for-windows/releases) page):
>
> | Variant | Files | `device` options |
> |---|---|---|
> | CUDA (recommended) | `lightgbm.exe`, `lib_lightgbm.dll`, `lightgbm-*.py3-none-win_amd64.whl` | `cpu`, `cuda` |
> | All-in-one | `all-lightgbm.exe`, `all-lib_lightgbm.dll`, `lightgbm-*+all-*.whl` | `cpu`, `cuda`, `gpu` (OpenCL) |
>
> ### `max_bin` note (fixes in `-r3` releases)
>
> Upstream CUDA has several bugs that trigger when **any feature has more
> than 256 bins** (`max_bin > 255`): illegal memory access that kills the
> process, silently corrupted trees, and garbage outputs for `max_bin ≥ 512`.
> `-r3` releases fix all of these (verified against the CPU learner; root
> cause reported in
> [#6512](https://github.com/microsoft/LightGBM/issues/6512)). With heavily
> zero-inflated features and `max_bin > 255` a further upstream bug remains
> (a warning is printed when the risky path is taken). For bit-for-bit
> CPU-equivalent results, keep `max_bin ≤ 255` (the default).
>
> ### Experimental: multi-GPU (`num_gpu > 1`) via a real NCCL
>
> Branch [`experiment/nccl-multigpu`](https://github.com/neoblackxt/lightgbm-cuda-for-windows/tree/experiment/nccl-multigpu)
> links a **real NCCL** instead of the stub. NCCL's official source tree
> (`v2.32.3-1`) already contains a complete Windows OS layer; the community
> mirror [WyvernTKC/nccl-windows](https://github.com/WyvernTKC/nccl-windows)
> adds only a 6-commit build recipe and fixes on top (two of them from
> NVIDIA's own `staging` branch). Provenance, build recipe, pitfalls and
> validation log: [`patches/EXPERIMENTAL-NCCL.md`](https://github.com/neoblackxt/lightgbm-cuda-for-windows/blob/experiment/nccl-multigpu/patches/EXPERIMENTAL-NCCL.md).
>
> **Validated so far** (single RTX 4090, WDDM, CUDA 13.3 — combinations the
> port author had not tested): single-rank NCCL collectives are bit-exact,
> and LightGBM's whole `num_gpu > 1` path (`NCCLGBDT`) runs end-to-end
> through a single-rank NCCL communicator, matching the CPU learner to 6
> decimals at `max_bin = 255`.
>
> **Not yet validated: the cross-GPU transport itself** (NVLink/PCIe ring,
> 2+ ranks). That needs multi-GPU hardware — which is where the community
> comes in:
>
> ```bash
> pip install --force-reinstall https://github.com/neoblackxt/lightgbm-cuda-for-windows/releases/download/nccl-exp1/lightgbm-4.7.0.99+nccl-py3-none-win_amd64.whl
> ```
>
> ```python
> import lightgbm as lgb
> params = {"objective": "regression", "device": "cuda", "num_gpu": 2,
>           "max_bin": 255, "seed": 7, "verbose": 1}
> # 2-GPU machine: expect "Using GPU devices 0,1"; quality should be close
> #   to num_gpu=1 / CPU on the same data.
> # 1-GPU machine: safely clamps ("Only 1 GPUs available, using num_gpu = 1")
> #   and still runs the full multi-GPU code path — a decent smoke test.
> ```
>
> If you have 2+ NVIDIA GPUs on Windows, please try it and report your setup
> (GPU models, driver, WDDM or TCC), whether training completes, quality vs
> `num_gpu=1`, and round time vs `num_gpu=1`, in
> [the community results issue](https://github.com/neoblackxt/lightgbm-cuda-for-windows/issues/2).
>
> Known limits of the upstream multi-GPU path: bagging is not supported
> (`NCCLGBDT` aborts with "Bagging is not supported"); leave
> `NCCL_ENQUEUE_REARCH_ENABLE` and `NCCL_SINGLE_PROC_MEM_REG_ENABLE` unset.
>
> ### Installation
>
> Grab the assets from the [latest release](https://github.com/neoblackxt/lightgbm-cuda-for-windows/releases/latest)
> (Windows 10/11 x64; Python 3.9+ for the wheels).
>
> **Python** — pick ONE wheel:
>
> ```bash
> # CUDA variant (cpu + cuda):
> pip install lightgbm-<version>-py3-none-win_amd64.whl
>
> # All-in-one variant (cpu + cuda + OpenCL):
> pip install lightgbm-<version>+all-py3-none-win_amd64.whl
> ```
>
> e.g. with pip directly from a release URL:
>
> ```bash
> pip install https://github.com/neoblackxt/lightgbm-cuda-for-windows/releases/download/v4.7.0.99-cuda13.3/lightgbm-4.7.0.99+all-py3-none-win_amd64.whl
> ```
>
> - If the official `lightgbm` is already installed, uninstall it first
>   (`pip uninstall lightgbm`) or add `--force-reinstall` — the two variants
>   of this fork also conflict with each other.
> - **No CUDA Toolkit installation needed** (the CUDA runtime is statically
>   linked); only the NVIDIA driver matters — see the hardware table below.
> - Verify: `python -c "import lightgbm; print(lightgbm.__version__)"`
>
> **CLI / C API** — download the binaries and drop them anywhere in `PATH`:
>
> | File | What it is |
> |---|---|
> | `lightgbm.exe` / `all-lightgbm.exe` | standalone trainer: `lightgbm.exe config=train.conf` with `device = cuda` inside |
> | `lib_lightgbm.dll` / `all-lib_lightgbm.dll` | C API library for native programs |
>
> The `all-` prefixed files are the all-in-one build; rename them as you like.
>
> ### Hardware compatibility
>
> CUDA mode is compiled for consumer NVIDIA GPUs, Turing and newer
> (`CUDAARCHS=75;80;86;89;120`):
>
> | GPU family | Examples | Compute capability | CUDA variant `device="cuda"` | All-in-one `device="cuda"` | All-in-one `device="gpu"` (OpenCL) |
> |---|---|---|---|---|---|
> | RTX 50 series (Blackwell) | 5090 / 5080 / 5070 | sm_120 | ✅ | ✅ | ✅ |
> | RTX 40 series (Ada) | 4090 / 4080 / 4070 | sm_89 | ✅ | ✅ | ✅ |
> | RTX 30 series (Ampere) | 3090 / 3080 / 3060 | sm_80 / sm_86 | ✅ | ✅ | ✅ |
> | GTX 16 / RTX 20 series (Turing) | 1660 / 2060 / 2080 | sm_75 | ✅ | ✅ | ✅ |
> | GTX 10 series and older (Pascal and before) | 1060 / 1070 / 900 | sm_61 … | ❌ (CUDA 13 dropped these) | ❌ | ✅ |
>
> - **Driver**: CUDA mode needs an NVIDIA driver new enough for CUDA 13.3
>   (check the `CUDA Version` field in `nvidia-smi` output). The OpenCL mode
>   only needs any GPU driver providing OpenCL 1.2+ (the `OpenCL.dll` that
>   ships with NVIDIA/AMD/Intel drivers).
> - **CPU**: any x86-64 processor, all variants.
> - Data-center cards (A100/H100, sm_90/sm_100) are not in the default arch
>   list; edit `CUDAARCHS` in `.github/workflows/build-windows-cuda.yml` and
>   re-tag to add them.
>
> ### Local benchmark (RTX 4090, 16 threads, 100 boosting rounds)
>
> One all-in-one binary, three `device` values, dense synthetic data
> (`num_leaves=127`, `max_bin=255`, best of 2 runs after warm-up):
>
> | Dataset | CPU | CUDA | OpenCL | GPU speed-up vs CPU |
> |---|---|---|---|---|
> | 1M x 50 | 34.3 ms/round (3.43 s) | 30.1 ms/round (3.01 s) | **25.6 ms/round (2.56 s)** | 1.1 – 1.3x |
> | 2M x 100 | 165.7 ms/round (16.6 s) | **38.8 ms/round (3.88 s)** | 40.1 ms/round (4.01 s) | **~4.3x** |
>
> - Identical training accuracy on all three devices (0.952 / 0.953).
> - Take-away: on small datasets the GPU advantage is marginal (data
>   transfer overhead dominates); at realistic scale both GPU back-ends
>   give a ~4x speed-up and CUDA/OpenCL are close on a consumer card.
> - Prediction runs on the CPU in all three cases (LightGBM does not
>   accelerate `predict` on GPU by default): ~0.25 s per 1M rows and
>   ~0.55 s per 2M×100 rows, identical across devices within noise.
>
> ### Switching devices (Python)
>
> ```python
> params = {"objective": "binary", "device": "cuda"}   # NVIDIA CUDA
> params = {"objective": "binary", "device": "gpu"}    # OpenCL (all-in-one wheel only)
> params = {"objective": "binary", "device": "cpu"}    # CPU (always available)
> # sklearn: LGBMClassifier(device="cuda")
> ```
>
> Notes:
> - No CUDA Toolkit install needed (statically linked runtime); just a recent NVIDIA driver.
> - Switch `device` on a **fresh Dataset**: reusing one `lgb.Dataset` that was already
>   trained on CPU and then training with `device="cuda"` raises an upstream limitation.
> - The OpenCL path (`device="gpu"`) computes sparse feature groups on the CPU
>   (hybrid mode); the CUDA path requires dense data.

> [!NOTE]
> This project moved from `Microsoft/LightGBM` to `lightgbm-org/LightGBM` in March 2026.
> This repository is still the official LightGBM source code, managed by the same maintainers (including the creator of LightGBM).
> For details, see https://github.com/lightgbm-org/LightGBM/issues/7187

Light Gradient Boosting Machine
===============================

[![C++ GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/cpp.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/cpp.yml)
[![Python-package GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/python_package.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/python_package.yml)
[![R-package GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/r_package.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/r_package.yml)
[![CUDA Version GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/cuda.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/cuda.yml)
[![SWIG Wrapper GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/swig.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/swig.yml)
[![Static Analysis GitHub Actions Build Status](https://github.com/lightgbm-org/LightGBM/actions/workflows/static_analysis.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/static_analysis.yml)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/1ys5ot401m0fep6l/branch/main?svg=true)](https://ci.appveyor.com/project/guolinke/lightgbm/branch/main)
[![Documentation Status](https://readthedocs.org/projects/lightgbm/badge/?version=latest)](https://lightgbm.readthedocs.io/)
[![Link checks](https://github.com/lightgbm-org/LightGBM/actions/workflows/lychee.yml/badge.svg?branch=main)](https://github.com/lightgbm-org/LightGBM/actions/workflows/lychee.yml)
[![License](https://img.shields.io/github/license/lightgbm-org/lightgbm.svg)](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE)
[![EffVer Versioning](https://img.shields.io/badge/version_scheme-EffVer-0097a7)](https://jacobtomlinson.dev/effver/)
[![StackOverflow questions](https://img.shields.io/stackexchange/stackoverflow/t/lightgbm?logo=stackoverflow&logoColor=white&label=StackOverflow%20questions)](https://stackoverflow.com/questions/tagged/lightgbm?sort=votes)
[![Python Versions](https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&logoColor=white)](https://pypi.org/project/lightgbm)
[![PyPI Version](https://img.shields.io/pypi/v/lightgbm.svg?logo=pypi&logoColor=white)](https://pypi.org/project/lightgbm)
[![conda Version](https://img.shields.io/conda/vn/conda-forge/lightgbm?logo=conda-forge&logoColor=white&label=conda)](https://anaconda.org/conda-forge/lightgbm)
[![CRAN Version](https://www.r-pkg.org/badges/version/lightgbm)](https://cran.r-project.org/package=lightgbm)
[![NuGet Version](https://img.shields.io/nuget/v/lightgbm?logo=nuget&logoColor=white)](https://www.nuget.org/packages/LightGBM)
[![Winget Version](https://img.shields.io/winget/v/Microsoft.LightGBM)](https://github.com/microsoft/winget-pkgs/tree/master/manifests/m/Microsoft/LightGBM)

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency.
- Lower memory usage.
- Better accuracy.
- Support of parallel, distributed, and GPU learning.
- Capable of handling large-scale data.

For further details, please refer to [Features](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Features.rst).

Benefiting from these advantages, LightGBM is being widely-used in many [winning solutions](https://github.com/lightgbm-org/LightGBM/blob/main/examples/README.md#machine-learning-challenge-winning-solutions) of machine learning competitions.

[Comparison experiments](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Experiments.rst#comparison-experiment) on public datasets show that LightGBM can outperform existing boosting frameworks on both efficiency and accuracy, with significantly lower memory consumption. What's more, [distributed learning experiments](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Experiments.rst#parallel-experiment) show that LightGBM can achieve a linear speed-up by using multiple machines for training in specific settings.

Get Started and Documentation
-----------------------------

Our primary documentation is at https://lightgbm.readthedocs.io/ and is generated from this repository. If you are new to LightGBM, follow [the installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) on that site.

Next you may want to read:

- [**Examples**](https://github.com/lightgbm-org/LightGBM/tree/main/examples) showing command line usage of common tasks.
- [**Features**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Features.rst) and algorithms supported by LightGBM.
- [**Parameters**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parameters.rst) is an exhaustive list of customization you can make.
- [**Distributed Learning**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parallel-Learning-Guide.rst) and [**GPU Learning**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/GPU-Tutorial.rst) can speed up computation.
- [**FLAML**](https://www.microsoft.com/en-us/research/articles/flaml-a-fast-and-lightweight-automl-library/) provides automated tuning for LightGBM ([code examples](https://microsoft.github.io/FLAML/docs/Examples/AutoML-for-LightGBM/)).
- [**Optuna Hyperparameter Tuner**](https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258) provides automated tuning for LightGBM hyperparameters ([code examples](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_tuner_simple.py)).

Documentation for contributors:

- [**How we update readthedocs.io**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/README.rst).
- Check out the [**Development Guide**](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Development-Guide.rst).

News
----

Please refer to changelogs at [GitHub releases](https://github.com/lightgbm-org/LightGBM/releases) page.

External (Unofficial) Repositories
----------------------------------

Projects listed here offer alternative ways to use LightGBM.
They are not maintained or officially endorsed by the `LightGBM` development team.

JPMML (Java PMML converter): https://github.com/jpmml/jpmml-lightgbm

Nyoka (Python PMML converter): https://github.com/SoftwareAG/nyoka

Treelite (model compiler for efficient deployment): https://github.com/dmlc/treelite

lleaves (LLVM-based model compiler for efficient inference): https://github.com/siboehm/lleaves

Hummingbird (model compiler into tensor computations): https://github.com/microsoft/hummingbird

GBNet (use `LightGBM` as a [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)): https://github.com/mthorrell/gbnet

cuML Forest Inference Library (GPU-accelerated inference): https://github.com/rapidsai/cuml

nvForest (GPU-accelerated inference): https://github.com/rapidsai/nvforest

daal4py (Intel CPU-accelerated inference): https://github.com/uxlfoundation/scikit-learn-intelex/tree/master/daal4py

m2cgen (model appliers for various languages): https://github.com/BayesWitnesses/m2cgen

leaves (Go model applier): https://github.com/dmitryikh/leaves

ONNXMLTools (ONNX converter): https://github.com/onnx/onnxmltools

SHAP (model output explainer): https://github.com/shap/shap

Shapash (model visualization and interpretation): https://github.com/MAIF/shapash

dtreeviz (decision tree visualization and model interpretation): https://github.com/parrt/dtreeviz

supertree (interactive visualization of decision trees): https://github.com/mljar/supertree

SynapseML (LightGBM on Spark): https://github.com/microsoft/SynapseML

Kubeflow Fairing (LightGBM on Kubernetes): https://github.com/kubeflow/fairing

Kubeflow Operator (LightGBM on Kubernetes): https://github.com/kubeflow/xgboost-operator

lightgbm_ray (LightGBM on Ray): https://github.com/ray-project/lightgbm_ray

Ray (distributed computing framework): https://github.com/ray-project/ray

Mars (LightGBM on Mars): https://github.com/mars-project/mars

ML.NET (.NET/C#-package): https://github.com/dotnet/machinelearning

LightGBM.NET (.NET/C#-package): https://github.com/rca22/LightGBM.Net

LightGBM Ruby (Ruby gem): https://github.com/ankane/lightgbm-ruby

LightGBM4j (Java high-level binding): https://github.com/metarank/lightgbm4j

LightGBM4J (JVM interface for LightGBM written in Scala): https://github.com/seek-oss/lightgbm4j

Julia-package: https://github.com/IQVIA-ML/LightGBM.jl

lightgbm3 (Rust binding): https://github.com/Mottl/lightgbm3-rs

MLServer (inference server for LightGBM): https://github.com/SeldonIO/MLServer

MLflow (experiment tracking, model monitoring framework): https://github.com/mlflow/mlflow

FLAML (AutoML library for hyperparameter optimization): https://github.com/microsoft/FLAML

MLJAR AutoML (AutoML on tabular data): https://github.com/mljar/mljar-supervised

Optuna (hyperparameter optimization framework): https://github.com/optuna/optuna

LightGBMLSS (probabilistic modelling with LightGBM): https://github.com/StatMixedML/LightGBMLSS

LightGBM-MoE (Mixture-of-Experts / regime-switching extension): https://github.com/kyo219/LightGBM-MoE

darts (time series forecasting and anomaly detection with LightGBM): https://github.com/unit8co/darts

mlforecast (time series forecasting with LightGBM): https://github.com/Nixtla/mlforecast

skforecast (time series forecasting with LightGBM): https://github.com/skforecast/skforecast

`{bonsai}` (R `{parsnip}`-compliant interface): https://github.com/tidymodels/bonsai

`{mlr3extralearners}` (R `{mlr3}`-compliant interface): https://github.com/mlr-org/mlr3extralearners

lightgbm-transform (feature transformation binding): https://github.com/Microsoft/LightGBM-transform

`postgresml` (LightGBM training and prediction in SQL, via a Postgres extension): https://github.com/postgresml/postgresml

`pyodide` (run `lightgbm` Python-package in a web browser): https://github.com/pyodide/pyodide

`vaex-ml` (Python DataFrame library with its own interface to LightGBM): https://github.com/vaexio/vaex

Support
-------

- Ask a question [on Stack Overflow with the `lightgbm` tag](https://stackoverflow.com/questions/ask?tags=lightgbm), we monitor this for new questions.
- Open **bug reports** and **feature requests** on [GitHub issues](https://github.com/lightgbm-org/LightGBM/issues).

How to Contribute
-----------------

Check [CONTRIBUTING](https://github.com/lightgbm-org/LightGBM/blob/main/CONTRIBUTING.md) page.

Microsoft Open Source Code of Conduct
-------------------------------------

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

Reference Papers
----------------

Yu Shi, Guolin Ke, Zhuoming Chen, Shuxin Zheng, Tie-Yan Liu. "Quantized Training of Gradient Boosting Decision Trees" ([link](https://proceedings.neurips.cc/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract.html)). Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 18822-18833.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "[A Communication-Efficient Parallel Algorithm for Decision Tree](https://proceedings.neurips.cc/paper/2016/hash/10a5ab2db37feedfdeaab192ead4ac0e-Abstract.html)". Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

Huan Zhang, Si Si and Cho-Jui Hsieh. "[GPU Acceleration for Large-scale Tree Boosting](https://arxiv.org/abs/1706.08359)". SysML Conference, 2018.

License
-------

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) for additional details.
