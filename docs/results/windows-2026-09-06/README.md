# Windows measurements, 2026-09-06

Ryzen 7 5700X / RTX 4060, MSVC, ggml v0.19.0, Vulkan SDK 1.4.341.1.

| Artifact | Contents |
| --- | --- |
| [benchmark-final.json](benchmark-final.json) | Nine engines/configurations, three lengths, 3 warmups and 10 samples; final host-to-host framework comparison |
| [all-results.md](all-results.md) | Median, P95, RTF, error and native CPU/GPU placement |
| [benchmark-initial.json](benchmark-initial.json) | Historical first pass; PyTorch CUDA excluded transfers, ORT version differs |
| [tuning.json](tuning.json) | Seven Vulkan variants, two lengths; 2 warmups and 5 samples; earlier im2col defaults |
| [diagnostics.json](diagnostics.json) | Separate samples-to-F0 timings and instrumented ONNX placement |
| [validation-cpu-f32.json](validation-cpu-f32.json) | CPU F32, 19/19 pass |
| [validation-vulkan-f32.json](validation-vulkan-f32.json) | Vulkan F32, 19/19 pass |
| [validation-cpu-f16.json](validation-cpu-f16.json) | F16 storage: JFK pitch-limit failure retained |
| [validation-vulkan-f16.json](validation-vulkan-f16.json) | F16 storage: JFK pitch-limit failure retained |
| [validation-vulkan-fast.json](validation-vulkan-fast.json) | Mixed arithmetic, separate relaxed limits |
| [provenance.json](provenance.json) | Model, source, input and final binary hashes |

Models, upstream code, WAVs and raw tensors are not committed. See [PERFORMANCE.md](../../PERFORMANCE.md) and [VALIDATION.md](../../VALIDATION.md) for reproduction.

Samples remain intact. Final binary hashes may include later CLI/tests/documentation changes that did not change measured network algorithms; provenance states this explicitly.
