# Windows model-storage measurements, 2026-09-06

Ryzen 7 5700X / RTX 4060 8 GB, Windows, MSVC, ggml v0.19.0, Vulkan SDK 1.4.341.1, four CPU threads, strict arithmetic. These are model-agreement and storage-size experiments, not annotated pitch-accuracy or inference-speed benchmarks.

| Artifact | Contents |
| --- | --- |
| [screening.json](screening.json) | Layer policy screening using JFK only |
| [audio-sources.json](audio-sources.json) | Source URLs, source-file SHA256 and excerpt offsets for held-out narration/song fixtures |
| [fixture-reproduction.json](fixture-reproduction.json) | Regenerated audio sample hashes and exact sample equality; WAV timestamps can differ |
| [verification.json](verification.json) | Local tool tests, original conversion hash preservation and result-check summary |
| [f16-validation.json](f16-validation.json) | Broad F16 vs original PyTorch; 24 cases, four failures retained |
| [f16-edges-validation.json](f16-edges-validation.json) | Rejected input/output edge-layer policy; two failures retained |
| [f16-intermediate-validation.json](f16-intermediate-validation.json) | Selected policy vs PyTorch; 23/24, libri1 frontend failure retained |
| [cross-validation.json](cross-validation.json) | Research candidate vs native F32 Vulkan, CPU and Vulkan 24/24 each |
| [conversion-equivalence.json](conversion-equivalence.json) | Formal converter vs research candidate: all 272 tensor names/shapes/types/bytes identical |
| [converted-model.json](converted-model.json) | Formal converter's model provenance, policy and SHA256 |
| [production-cpu-comparison.json](production-cpu-comparison.json) | Formal converter output vs native F32 Vulkan on 24 fixtures, including exact frontend equality |
| [production-vulkan-comparison.json](production-vulkan-comparison.json) | Same check with Vulkan candidate |
| [production-packaging.json](production-packaging.json) | Formal GGUF/package sizes, round-trip SHA256 and legacy archive compatibility |
| [compressed-candidate.json](compressed-candidate.json) | Historical package of the research candidate (smaller GGUF metadata) |
| [lossless-probe.json](lossless-probe.json) | Earlier direct/byte-shuffle/bit-shuffle compression probes with exact restoration |

Research JSON files preserve their original measurements and failures. The candidate used for screening and the original quality reports has SHA256 `b47d8a9db19341c9b45fd2baa474b15bb038577dc7695623af2762abbbd4513f`; the formal converter adds provenance metadata and produces `9cc338bd949a601cedbd114aec4eb69a98999fd0bb31f16856afee5e3ead9b2f`. Tensor contents match exactly. The production comparisons test the latter file with the new reusable tool.

Every end-to-end profile fails the unchanged 0.005 frontend bound for libri1 (observed 0.008909225). The selected policy has no pitch-limit or UV failures. Its separate same-frontend comparison passes for all cases. Do not report the end-to-end suite as 24/24.

The original research `.bsz` archive is compatible with `compress_model.py unpack`. The older pure-lossless shuffled probe files lack its 48-byte global header and are not BSZ files. Timings in packaging JSON are single runs and include file I/O; the original probe timings used different scopes. They should not be compared as a throughput benchmark.

Local model/audio/tensor paths in historical reports describe the original workspace. Assets are not distributed. See [MODEL_SIZE.md](../../MODEL_SIZE.md) for supported commands and limitations, and [the original performance archive](../windows-2026-09-06/README.md) for PyTorch/ONNX Runtime speed comparisons.
