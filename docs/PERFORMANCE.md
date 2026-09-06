# Performance and acceleration experiments

## Test environment and timing contract

Measurements were made on 2026-09-06, Windows 10.0.29648, AMD Ryzen 7 5700X (8 cores / 16 threads), NVIDIA GeForce RTX 4060 8 GB, driver 620.02, Vulkan SDK **1.4.341.1**, MSVC **19.51.36256**, AVX2/FMA CPU build. Python 3.11.4, PyTorch **2.7.1+cu118**, ONNX Runtime GPU **1.22.0** (CPU and CUDA providers), ggml **v0.19.0**.

The final comparison is batch 1, four CPU threads, three warmups and **ten timed samples**, sequential engines on identical log-Mel matrices. Every timed call goes from **host F32 input to host F32 output**, including GPU upload, download and synchronization. PyTorch disables TF32 and cuDNN benchmark search. ONNX CUDA disables TF32 and uses HEURISTIC cuDNN convolution selection. ONNX models use dynamic time, fixed batch 1 and opset 17.

Model load, ONNX export, graph construction, Vulkan pipeline creation, WAV I/O, Mel extraction and pitch decoding are excluded from this network table. Native CLI JSON records load/frontend/first-inference times separately. Native `--benchmark-audio` adds a separate samples-to-F0 measurement. Do not interpret first-inference cost as steady-state latency.

Inputs are seeded harmonic glides with voiced/unvoiced intervals and noise, generated at 16 kHz. The durations 1/3/10 s yield 101/301/1001 valid frames and **128/320/1024 padded frames**. RTF uses original audio duration, not padded duration.

## Final network medians

| Engine | 1 s, ms | 3 s, ms | 10 s, ms | 10 s RTF |
| --- | ---: | ---: | ---: | ---: |
| ggml CPU F32, fused GRU | 165.830 | 390.670 | 1249.777 | 0.124978 |
| ggml CPU F32, unrolled GRU | 160.569 | 385.105 | 1254.215 | 0.125421 |
| ggml CPU F16 im2col, F32 weights | 218.051 | 519.624 | 1644.870 | 0.164487 |
| **ggml Vulkan strict F32** | **17.708** | **23.509** | **57.033** | **0.005703** |
| ggml Vulkan mixed arithmetic | 18.163 | 19.258 | 42.676 | 0.004268 |
| PyTorch CPU | 101.217 | 260.415 | 464.081 | 0.046408 |
| PyTorch CUDA | 31.562 | 39.116 | 79.844 | 0.007984 |
| ONNX Runtime CPU | 42.705 | 84.043 | 271.464 | 0.027146 |
| ONNX Runtime CUDA | 23.195 | 34.379 | 75.208 | 0.007521 |

Strict Vulkan is about **1.40x faster than PyTorch CUDA**, **1.32x faster than ORT CUDA**, and **21.9x faster than this ggml CPU implementation** for the 10 s fixture. It remains more than 175x realtime. These ratios describe this machine and workload; CPU ggml does not outperform the optimized CPU frameworks.

At 10 s, the strict Vulkan graph has **599 graph nodes, 461 accelerator compute nodes, zero CPU compute nodes, one graph split** and **34,078,720 bytes (32.5 MiB) of scheduler compute buffers**. That is not total process/GPU memory: weights and backend resources are additional. The F32 GGUF is about 345 MiB.

Mixed mode retains F32 storage in this table but permits reduced-precision operands internally. Its validation limit differs from strict mode, and it was slightly slower for the shortest final case. F16 storage is a separate option with a known accuracy failure; neither is silently enabled by the default.

Full individual samples, P95, numerical errors and commands are in [benchmark-final.json](results/windows-2026-09-06/benchmark-final.json) and [all-results.md](results/windows-2026-09-06/all-results.md).

## Separate audio-to-F0 and provider diagnostics

A later independent run measured in-memory mono 16 kHz audio to decoded F0 (3 warmups / 10 samples), excluding file I/O and model load:

| Engine | 1 s, ms | 3 s, ms | 10 s, ms |
| --- | ---: | ---: | ---: |
| CPU F32 | 176.345 | 392.793 | 1207.967 |
| Vulkan strict F32 | 15.545 | 22.553 | 51.993 |

These runs have different clock/scheduling conditions from the main table. Some end-to-end medians are lower than earlier network medians; do not subtract them to infer frontend cost. See [diagnostics.json](results/windows-2026-09-06/diagnostics.json).

The ONNX profile assigns **333 nodes to CUDA and 6 to CPU**. All 124 Conv, 5 ConvTranspose, GRU and MatMul execute on CUDA. CPU nodes are Gather (2), Slice (1), Concat (2) and Unsqueeze (1). The CUDA comparison therefore includes some CPU execution. Instrumented timings are excluded from the table.

The local ORT setup explicitly preloaded CUDA 12.8 libraries; PyTorch itself was a CUDA 11.8 build. ORT emitted a version-compatibility warning, but CUDA initialization, node placement and output comparisons succeeded in this recorded environment. Prefer compatible PyTorch/ORT CUDA packages for a fresh setup.

## What accelerated the model

The implementation was measured through multiple independent changes; the earlier exploratory runs use fewer repetitions and are not interchangeable with final framework timing.

1. **Memory policy.** For the same 32-frame exploratory graph with CPU GRU and default mixed Vulkan arithmetic, upstream host-visible device allocation gave approximately 161 ms. Selecting non-host-visible device allocation reduced it to approximately 31 ms. This behavior matches the warning signs observed in the FCPE.cpp reference. The default build now selects device-local memory for discrete hardware, retaining an override for other devices.
2. **Persistent GPU GRU.** A portable unrolled graph was implemented first. A tagged custom operator and Vulkan shader eliminate CPU recurrence transfers and thousands of per-frame operations. The initial row-major shader was correct but memory-inefficient.
3. **Coalesced recurrent weights.** Transposing the 768x256 recurrent matrix lets adjacent invocations read neighboring weight values. In the controlled 1/10 s tuning run, strict Vulkan with im2col fell from **41.75/235.35 ms** (row-major recurrence) to **25.31/94.41 ms**. CPU recurrence gave **33.96/128.61 ms**, while full graph unrolling gave **32.91/121.47 ms**.
4. **Existing direct convolution.** ggml v0.19.0 already includes direct/implicit-GEMM Conv2d and ConvTranspose2d kernels. Using direct Conv2d instead of materialized im2col reduced the same strict tuning cases to **21.15/58.98 ms**, with probability max error below 9e-7. This path and coalesced GRU are now defaults.
5. **BatchNorm folding.** Evaluation BN scale/offset is folded into convolution tensors at conversion, removing runtime normalization nodes. The unused E2E0 timbre-filter branch is removed. No separate speed claim is assigned without an unfused benchmark.
6. **Mixed arithmetic.** The fast-direct tuning case reached **10.82/46.61 ms**, but final short-input samples demonstrate normal variability and do not justify enabling it by default.

The complete tuning run is [tuning.json](results/windows-2026-09-06/tuning.json). Its `vulkan` label means the earlier **im2col + transposed GRU** graph; `vulkan-direct` corresponds to today's default. To reproduce old labels with the current executable, explicitly pass `--im2col` for old non-direct variants. The graph policy changed after that run; the raw JSON is preserved unchanged.

[benchmark-initial.json](results/windows-2026-09-06/benchmark-initial.json) retains the first full run before the layout/direct-convolution improvements. Its PyTorch CUDA scope excluded transfers and it used ORT CPU 1.22.1; **use the final archive for fair framework comparisons**.

## Reproduction

```powershell
$env:NUMBA_CACHE_DIR = "$PWD\work\numba"
python scripts/benchmark.py --seconds 1 3 10 --variants cpu cpu-unroll cpu-fast vulkan vulkan-fast --threads 4 --warmup 3 --runs 10 --output work/benchmark-final.json
```

Install a compatible `onnxruntime-gpu` package to include its CUDA provider. The local measurement reused the existing isolated installation at `F:/ggml/FCPE.cpp/.ort-deps`:

```powershell
$env:RMVPE_ORT_PATH = 'F:\ggml\FCPE.cpp\.ort-deps'
python scripts/benchmark.py --cuda-dir 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8' --seconds 1 3 10 --variants cpu cpu-unroll cpu-fast vulkan vulkan-fast --warmup 3 --runs 10 --output work/benchmark-final.json
```

`RMVPE_ORT_PATH` is optional and simply selects an existing isolated Python package directory. No FCPE inference is involved. The script verifies requested provider initialization and fails on CUDA initialization fallback. Actual CUDA node assignment is checked separately by `scripts/diagnose.py`, whose profiling time is excluded.

For current-graph ablations:

```powershell
python scripts/benchmark.py --skip-reference-timing --seconds 1 10 --warmup 2 --runs 5 --variants vulkan vulkan-im2col vulkan-im2col-row-major vulkan-cpu-gru vulkan-unroll vulkan-fast --output work/tuning-current.json
python scripts/diagnose.py --ort-path F:/ggml/FCPE.cpp/.ort-deps
python scripts/record_results.py
```

Native audio-to-F0 measurements and ONNX placement are recorded in `diagnostics.json` alongside the main archive. Source, input, binary and model hashes are retained in `provenance.json`. Large binaries, models, WAVs and raw tensors are excluded from Git.

## Practical limits and remaining work

The model is bidirectional and evaluated whole-input. Independent chunking changes context and is not a free memory optimization. Graphs are cached by frame count, and first-call pipeline work may dominate one-shot usage. Benchmark a reused Model instance when that reflects the intended application.

The CPU convolution path is the main performance deficit; tuned CPU convolution/GEMM integration is worth further work. On Vulkan, the single-workgroup GRU and small bias/activation dispatches are remaining targets. Future improvements must retain precise crop/layout semantics and the documented numerical limits. See [BACKENDS.md](BACKENDS.md) and the unvalidated [Metal plan](METAL.md).

## Storage size follow-up

The `f16-intermediate` profile and optional BSZ packaging reduce the tested model to 203.119 MiB / 170.512 MiB respectively. These are storage measurements; the framework speed tables above still use the documented F32 model. Strict CPU/Vulkan execution expands F16 weights for F32 computation, so no inference speedup or memory reduction is claimed. See [MODEL_SIZE.md](MODEL_SIZE.md) for quality, archive sizes and reproduction.
