# RMVPE.cpp

[简体中文](README_CN.md)

Native C++17 pitch extraction for the [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) **230917** model, using [ggml](https://github.com/ggml-org/ggml) and GGUF. The runtime includes waveform loading, resampling, the log-Mel frontend, the residual U-Net, bidirectional GRU and local-average pitch decoding. It does not require Python, PyTorch or ONNX Runtime for inference.

**CPU and Vulkan are implemented and tested on Windows.** Vulkan includes a persistent GRU shader, device-local allocation policy and a strict F32 default. Metal and native ggml CUDA build options are provided for development, but have not been validated. See [backend details](docs/BACKENDS.md) and [Metal work remaining](docs/METAL.md).

This project adapts infrastructure from [KCKT0112/FCPE.cpp](https://github.com/KCKT0112/FCPE.cpp), and follows [game.cpp](https://github.com/KakaruHayate/game.cpp) and [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) for project and distribution conventions. FCPE.cpp originates from [CNChTu/FCPE](https://github.com/CNChTu/FCPE), by CN_ChiTu. Exact revisions and reused components are listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Build

Requirements: CMake 3.18+, a C/C++17 compiler, Git, and optionally Ninja. The pinned tensor engine is ggml **v0.19.0**. CMake downloads its hash-checked archive unless `third_party/ggml` or `RMVPE_GGML_SOURCE_DIR` supplies it.

```sh
git clone https://github.com/KCKT0112/RMVPE.cpp.git
cd RMVPE.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

For Windows, use a **Developer PowerShell for Visual Studio**. The tested installation used MSVC 19.51 and Ninja:

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 8

$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.341.1'
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DRMVPE_VULKAN=ON -DRMVPE_TEST_BACKEND=Vulkan0
cmake --build build-vulkan -j 8
ctest --test-dir build-vulkan --output-on-failure
.\build-vulkan\bin\rmvpe-cli.exe --list-backends
```

Executables and shared backend libraries are in `build*/bin`. Keep the corresponding ggml DLLs with the executable. For CPU distribution to other machines, configure `-DGGML_NATIVE=OFF -DRMVPE_CPU_AVX2=OFF` and choose ggml's CPU ISA options for your target; the measured Windows build used AVX2/FMA.

The Vulkan build generates a patched copy in the build directory and leaves the ggml checkout intact. The integration deliberately fails on incompatible source anchors; use the pinned ggml version.

## Download and convert the model

Python is needed only for conversion, reference validation and benchmarks:

```sh
python -m pip install -r requirements.txt
git clone https://github.com/yxlllc/RMVPE.git reference/RMVPE
git -C reference/RMVPE checkout 0aabafba18289ca938a73af0b0297686abf4922d
python scripts/download_model.py
python scripts/convert_rmvpe.py models/original/model.pt models/rmvpe-f32.gguf
python scripts/convert_rmvpe.py models/original/model.pt models/rmvpe-f16.gguf --dtype f16
```

The downloader retrieves the requested [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), checks SHA256 and extracts only `model.pt`. The converter loads it with `weights_only=True`, validates the E2E0 architecture, folds evaluation BatchNorm into convolution weights, and omits the timbre-filter branch that E2E0 never executes. GRU weights, affine terms, window and filterbank remain F32 in both storage formats.

| Output | Bytes | Description |
| --- | ---: | --- |
| `rmvpe-f32.gguf` | 361,883,776 | Recommended reference/default storage |
| `rmvpe-f16.gguf` | 183,112,672 | F16 convolution/projection storage; other tensors F32 |

F16 storage failed the 5-cent pitch limit on one real-speech case (about **12.39 cents** maximum deviation, with no voiced/unvoiced mismatch). It is an experimental size option; use F32 when numerical parity matters. See [validation details](docs/VALIDATION.md).

Each output includes GGUF provenance metadata and a `.gguf.json` manifest with source/checkpoint/output hashes. Weights, ONNX models and original Python sources are not committed.

If librosa/Numba cache creation stalls in a restricted Windows account, select a writable cache before conversion:

```powershell
$env:NUMBA_CACHE_DIR = "$PWD\work\numba"
```

**Licensing:** this port's code is MPL-2.0. No explicit license was found in the inspected original RMVPE source repository or weight archive. That does not make the model MPL-licensed or public domain. Review [NOTICE.md](NOTICE.md) before redistributing model assets.

## Extract pitch

```powershell
.\build-cpu\bin\rmvpe-cli.exe --model models/rmvpe-f32.gguf --wav input.wav --output pitch.csv --threads 4
.\build-vulkan\bin\rmvpe-cli.exe --model models/rmvpe-f32.gguf --wav input.wav --output pitch.csv --backend Vulkan0 --threads 4
```

The CSV contains `time_s,f0_hz,confidence`, with a 10 ms hop. Unvoiced frames have F0 zero. The default confidence threshold is **0.03**, using upstream's strict `confidence > threshold` condition. PCM 8/16/24/32-bit and IEEE float 32/64-bit RIFF WAV are accepted, stereo/multichannel audio is averaged to mono, and non-16 kHz audio is sinc-resampled.

Ordinary extraction runs the network once. `--benchmark-audio` measures in-memory samples to F0, including preprocessing and decoding but excluding file I/O. To benchmark a reusable session, add `--warmup 3 --runs 10`. JSON timing and backend-placement data go to stdout; logs go to stderr. First-call graph allocation is reported separately from repeated timings.

```powershell
.\build-vulkan\bin\rmvpe-cli.exe --model models/rmvpe-f32.gguf --wav input.wav --backend Vulkan0 --warmup 3 --runs 10
```

For an existing log-Mel matrix, use `--mel input.f32`. The format is little-endian contiguous F32 **[frames,128]**, with a positive multiple of 32 frames. It represents the already padded network input. `--probabilities output.f32` exports **[padded_frames,360]**; `--dump-mel output.f32` exposes the native frontend. WAV-to-CSV output is cropped to `floor(resampled_samples/160)+1` frames.

The frontend reproduces upstream **audio inference**: right zero-padding of the waveform, centered reflective 1024-point STFT, periodic Hann window, magnitude, HTK Mel with Slaney area normalization (128 bins, 30-8000 Hz), clamp at `1e-5`, then natural log. Upstream's separate `mel2hidden` helper has a different reflection-padding convention; raw-Mel callers must supply their own intended padding.

An instance accepts at most 16,384 padded frames; actual memory limits depend on the device. Inference processes the whole input, with full backward-GRU context. Independent chunking changes boundary predictions and is not automatically substituted. Viterbi decoding, training, streaming, arbitrary checkpoint architectures and quantized GGUF weights are not implemented.

## C++ API

```cpp
#include <rmvpe/rmvpe.h>

rmvpe::Options options;
options.backend = "Vulkan0"; // "cpu", "auto", or a name from rmvpe::devices()
options.threads = 4;
rmvpe::Model model("models/rmvpe-f32.gguf", options);
auto audio = rmvpe::read_wav("input.wav");
auto result = model.infer(audio.samples, audio.sample_rate);
// result.f0 and result.confidence contain one value per valid 10 ms frame.
```

Use `add_subdirectory(path/to/RMVPE.cpp)` and link `rmvpe::rmvpe`. See [examples/basic.cpp](examples/basic.cpp). Build the standalone consumer with cmake -S examples -B build-example -G Ninja and cmake --build build-example; its executable is in uild-example/bin. The model owns its weights, scheduler and cached graph; use one instance per concurrent caller. Errors are C++ exceptions.

## Measured performance

Windows 10.0.29648, Ryzen 7 5700X, RTX 4060 8 GB, NVIDIA 620.02, Vulkan SDK 1.4.341.1. Batch 1, four CPU threads, three warmups, ten timed runs. All figures are median milliseconds for **the same padded log-Mel input to host F32 probabilities**, including GPU transfers and synchronization; loading, compilation, waveform preprocessing and decoding are excluded.

| Engine | 1 s / 128 frames | 3 s / 320 frames | 10 s / 1024 frames |
| --- | ---: | ---: | ---: |
| ggml CPU F32 | 165.83 | 390.67 | 1249.78 |
| **ggml Vulkan strict F32** | **17.71** | **23.51** | **57.03** |
| ggml Vulkan mixed arithmetic | 18.16 | 19.26 | 42.68 |
| PyTorch 2.7.1 CPU | 101.22 | 260.42 | 464.08 |
| PyTorch 2.7.1 CUDA | 31.56 | 39.12 | 79.84 |
| ONNX Runtime 1.22.0 CPU | 42.70 | 84.04 | 271.46 |
| ONNX Runtime 1.22.0 CUDA | 23.20 | 34.38 | 75.21 |

The strict Vulkan 10 s result corresponds to RTF **0.00570**, about **175x realtime**. Native CPU is substantially slower than ONNX Runtime CPU on this machine. Mixed arithmetic has a separate error budget and is opt-in; it is not consistently faster for short inputs.

These are reproducible engineering measurements on deterministic harmonic signals, not a singing-pitch accuracy evaluation or a guarantee for other hardware. See [PERFORMANCE.md](docs/PERFORMANCE.md) for raw samples, timing scope, optimization ablations and commands, and [VALIDATION.md](docs/VALIDATION.md) for numerical results and their limits.

## License and acknowledgements

Project contributions use **[MPL-2.0](LICENSE)**. Preserve [NOTICE.md](NOTICE.md), [THIRD_PARTY.md](THIRD_PARTY.md) and the applicable notices in [licenses/](licenses/) when redistributing. Third-party code and model assets retain their own terms. When distributing MPL-covered binaries, provide the corresponding MPL-covered source and modifications as required by the license.

Thanks to the RMVPE authors, yxlllc, ggml contributors, KCKT0112/FCPE.cpp, CN_ChiTu's FCPE, KakaruHayate/game.cpp, KakaruHayate/pc-nsf-hifigan.cpp and pocketfft contributors. This is an independent implementation, provided as is.
