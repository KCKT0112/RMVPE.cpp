# Model storage and quality

The supported size/quality option is **`--dtype f16-intermediate`**. It stores the 33 convolution-weight tensors under `unet.intermediate.` as F16, while encoder, decoder, output, GRU, biases and frontend tensors remain F32. This is lossy weight storage with very small measured output differences on the current fixtures. F32 remains the default/reference.

## Measured sizes

The source is the original 230917 checkpoint. MiB means 1,048,576 bytes.

| Storage / package | Bytes | MiB | Reduction from F32 |
| --- | ---: | ---: | ---: |
| F32 GGUF | 361,883,776 | 345.119 | — |
| **Intermediate-only F16 GGUF** | **212,986,112** | **203.119** | **41.15%** |
| Above GGUF, byte shuffle + Zstd 9 (`.bsz`) | **178,794,609** | **170.512** | **50.59%** |
| Existing broad F16 GGUF | 183,112,672 | 174.630 | 49.40% |

The broad F16 profile is smaller as a directly loadable GGUF, but had isolated deviations of up to **31.14 cents** on the expanded fixtures. The intermediate-only profile is the recommended reduced-size option for the tested workload.

Conversion and reversible packaging are separate steps. F16 conversion changes weight values. BSZ packaging restores the resulting GGUF **byte for byte** and introduces no further loss.

## Convert, pack and use

Install the main Python requirements and download the original checkpoint as described in the [README](../README.md), then:

```sh
python scripts/convert_rmvpe.py models/original/model.pt models/rmvpe-f16-intermediate.gguf --dtype f16-intermediate
python -m pip install -r requirements-compression.txt
python scripts/compress_model.py pack models/rmvpe-f16-intermediate.gguf models/rmvpe-f16-intermediate.gguf.bsz
python scripts/compress_model.py unpack models/rmvpe-f16-intermediate.gguf.bsz models/rmvpe-restored.gguf
```

Pack/unpack destinations must be new files in existing directories. The tool publishes the destination only after completing the operation, refuses to overwrite an existing file, and removes only its own temporary file on failure. Unpacking verifies block lengths, Zstd frame checksums, total length and the restored SHA256. Its stdout JSON records sizes, the model hash and elapsed time. Keep the conversion manifest when distributing a package.

The C++ loader accepts the unpacked GGUF directly:

```powershell
.\build-vulkan\bin\rmvpe-cli.exe --model models/rmvpe-restored.gguf --wav input.wav --backend Vulkan0 --output pitch.csv
```

A BSZ archive is not a native GGUF and cannot be passed to `--model`. No new C++ operators are needed for the selected F16 storage policy: the current CPU and Vulkan paths already support mixed F16/F32 tensors. Under strict arithmetic, F16 weights are expanded for F32 computation. **Smaller disk/download size does not imply less RAM/VRAM or faster inference**; graph temporary buffers can increase. No speedup is claimed for this storage change. Framework speed comparisons remain in [PERFORMANCE.md](PERFORMANCE.md). Metal is still unvalidated on Windows; see [METAL.md](METAL.md).

## Quality measurements and retained failures

The expanded evaluation contains **24 cases and 8,062 valid frames**: the original 19 validation cases, male and female narration from LibriSpeech, and three ten-second excerpts of Karissa Hobbs' *Let's Go Fishin'*. Only JFK speech was used to screen layer policies. The added recordings were held out from that screening. The rejected edge-layer candidate applied the broad F16 eligibility rule except for tensor prefixes `cnn.`, `fc.1.`, `unet.encoder.layers.0.` and `unet.decoder.layers.4.`, which stayed F32.

All three storage profiles were evaluated with the original PyTorch model, strict native Vulkan arithmetic, probability tolerance 0.01, pitch tolerance 5 cents, no voiced/unvoiced disagreements, and the existing frontend tolerance 0.005.

| Storage policy | Max pitch difference vs PyTorch, cents | Frames above 5 cents | UV disagreements | End-to-end cases passing |
| --- | ---: | ---: | ---: | ---: |
| Broad F16 | 31.142229 | 3 | 0 | 20/24 |
| F16 with input/output edge layers kept F32 (research candidate) | 15.671407 | 2 | 0 | 22/24 |
| **Intermediate-only F16** | **0.003755** | **0** | **0** | **23/24** |

**The `libri1` end-to-end test remains a failure.** Its maximum native-vs-PyTorch log-Mel difference is 0.008909225, above the existing 0.005 limit, identically across all three storage policies. The limit was not raised and the failed JSON results are preserved. The selected profile's maximum probability difference versus PyTorch is 0.000109375; its pitch errors are all below 0.1 cent, but this does not override the frontend failure.

To isolate the storage effect, a separate comparison uses native F32 Vulkan as the reference, with identical WAV inputs and native frontend processing:

| Intermediate-only candidate backend | Cases passing | Probability max absolute difference | Pitch max difference, cents | UV disagreements |
| --- | ---: | ---: | ---: | ---: |
| CPU | 24/24 | 2.50340e-6 | 0.001049 | 0 |
| Vulkan strict | 24/24 | 2.02656e-6 | 0.001157 | 0 |

The comparison limits are probability 2e-4, pitch 0.1 cent and zero UV disagreements. These are **model-agreement tests**, without annotated ground-truth singing pitch or a large representative singing corpus. They do not establish universal perceptual equivalence or accuracy. INT8/INT4 storage and quantized convolution kernels have not been implemented or evaluated; their quality and backend support cannot be inferred from these F16 results.

## Reproduce quality comparisons

Set a writable Numba cache on restricted Windows accounts:

```powershell
$env:NUMBA_CACHE_DIR = "$PWD\work\numba"
```

Prepare the extra recordings from the hash-pinned [source manifest](results/windows-2026-09-06-storage/audio-sources.json). The script downloads missing OGG files only with `--download`. Supply the original JFK WAV separately; it was sourced from whisper.cpp's sample through the FCPE.cpp validation workspace. WAV hashes in the archived validation results identify the exact tested files. libsndfile includes a write timestamp in the WAV PEAK chunk, so regenerated WAV file hashes can differ with identical samples. The preparation tool records stable little-endian F32 sample hashes in `fixtures.json`; [fixture reproduction](results/windows-2026-09-06-storage/fixture-reproduction.json) verifies exact sample equality with the tested recordings.

```sh
python scripts/prepare_storage_audio.py --download --jfk path/to/recording_0_jfk.wav
python scripts/validate.py --cli build-vulkan/bin/rmvpe-cli.exe --backend Vulkan0 --model models/rmvpe-f16-intermediate.gguf --prob-atol 0.01 --pitch-atol 5 --wav work/storage-audio/recording_0_jfk.wav --wav work/storage-audio/libri1.wav --wav work/storage-audio/libri3.wav --wav work/storage-audio/fishin_10s.wav --wav work/storage-audio/fishin_45s.wav --wav work/storage-audio/fishin_90s.wav --work work/storage-validation --output work/storage-validation.json
```

The validation command is expected to return a nonzero status for the retained `libri1` frontend failure. Preserve that report. To reproduce the broad F16 row, substitute `models/rmvpe-f16.gguf` and use separate work/output paths.

Then compare all 24 generated WAVs against native F32, independently of the PyTorch frontend:

```sh
python scripts/compare_models.py --reference models/rmvpe-f32.gguf --model models/rmvpe-f16-intermediate.gguf --fixtures work/storage-validation --work work/storage-compare-cpu --output work/storage-compare-cpu.json
python scripts/compare_models.py --reference models/rmvpe-f32.gguf --model models/rmvpe-f16-intermediate.gguf --fixtures work/storage-validation --cli build-vulkan/bin/rmvpe-cli.exe --backend Vulkan0 --work work/storage-compare-vulkan --output work/storage-compare-vulkan.json
```

Both commands default to the Vulkan F32 reference. For CPU-only machines, pass `--reference-cli build-cpu/bin/rmvpe-cli.exe --reference-backend cpu`; the recorded numbers above use Vulkan as the reference. `--wav` can add user recordings. The comparison tool checks exact frontend equality, output frame counts, probabilities and decoded pitch, and returns a nonzero status on failure. It clears Vulkan experiment overrides and uses strict arithmetic. Run tests sequentially.

The formal converter's 272 tensor names, shapes, storage types and data bytes were verified identical to the tested research candidate. Added provenance metadata changes the file size by 416 bytes and changes the whole-file hash. The published hash is:

```text
9cc338bd949a601cedbd114aec4eb69a98999fd0bb31f16856afee5e3ead9b2f
```

The earlier research candidate hash is `b47d8a9db19341c9b45fd2baa474b15bb038577dc7695623af2762abbbd4513f`. Its 170.521 MiB package and quality results are retained as historical measurements. The formal converter output is 203.119 MiB and packages to 170.512 MiB with the tested zstandard 0.22.0. Compression bytes can vary with library versions. See [conversion equivalence](results/windows-2026-09-06-storage/conversion-equivalence.json), [production packaging](results/windows-2026-09-06-storage/production-packaging.json) and the [complete results index](results/windows-2026-09-06-storage/README.md).

Tool regression tests cover storage boundaries, archives spanning 8 MiB blocks, truncation/corruption, invalid sizes, trailing or concatenated frames, checksums, and existing/concurrently created output preservation:

```sh
python -m pip install -r requirements-compression.txt pytest
python -m pytest tests/test_storage.py -q
```

## Pure lossless compression

The baseline F32 GGUF has 272 tensors, 361,860,148 tensor bytes, just 23,628 bytes of metadata/alignment, no duplicate tensors, and about 0.07% zero elements. Removing redundant containers or zeros cannot halve it. The intermediate module accounts for 82.3% of F32 tensor bytes, which motivated the selected policy.

The separate [lossless probe](results/windows-2026-09-06-storage/lossless-probe.json) measured:

| Input | Method | Archive bytes | Reduction |
| --- | --- | ---: | ---: |
| F32 | Zstd level 9, direct | 334,015,884 | 7.70% |
| F32 | Byte shuffle + Zstd level 9 | 305,801,097 | 15.50% |
| F32 | Byte shuffle + Zstd level 19 | 302,816,277 | 16.32% |
| Broad F16 | Byte shuffle + Zstd level 9 | 153,500,766 | 16.17% |

All probe variants verified exact SHA256 restoration. The old shuffled probes had block headers but **no global header** and are not BSZ archives. The supported tool adds a 48-byte global header. For reversible F32 packaging, use:

```sh
python scripts/compress_model.py pack models/rmvpe-f32.gguf models/rmvpe-f32.gguf.bsz --level 19
```

Higher compression levels were not consistently smaller on F16, so level 9 is the default. The one-shot compression/decompression timings in JSON include different I/O and verification scopes and are not formal throughput benchmarks.

## BSZ format

All integers are little-endian. The format is specific to this packaging tool:

1. Eight bytes `RMVPEBS1`, original length (u64), and original SHA256 (32 raw bytes).
2. Repeated blocks: original length (u64), compressed length (u64), one checksummed Zstd frame.
3. Each original block contains at most 8 MiB and its length is divisible by four. Shuffle the block as `uint8.reshape(-1,4).T.copy()`; reverse it as `uint8.reshape(4,-1).T.copy()`.
4. The decoder requires the declared total size and SHA256 to match exactly, bounds each frame's compressed size and decode window, and rejects trailing bytes.

This is a reversible byte permutation of the entire file, including headers and alignment; no tensor-specific parsing is involved. The embedded SHA256 detects corruption and is not an authenticity signature.

Model asset terms are unchanged. Project tools are MPL-2.0; neither conversion nor packaging grants a license to redistribute the original weights. See [NOTICE.md](../NOTICE.md).
