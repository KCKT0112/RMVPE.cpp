# Sources and licenses

See [NOTICE.md](NOTICE.md) for distribution obligations and the separate, unresolved license status of the inspected RMVPE model assets.

| Source | Inspected commit / release | Relationship |
| --- | --- | --- |
| [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) | `0aabafba18289ca938a73af0b0297686abf4922d`; model release 230917 | Architecture, audio-padding/STFT convention, PyTorch numerical reference and local-average decoder |
| [KCKT0112/FCPE.cpp](https://github.com/KCKT0112/FCPE.cpp) | `c1f0dadd916f37abe7ab884e71eccd145488002e` | MPL-2.0 loader/scheduler, WAV/resampler, frontend and CMake patterns adapted |
| [KakaruHayate/pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) | `e46c93308f0bbec36e1d08707f94250c79abab56` | MPL-2.0 project/license reference; pocketfft header provenance |
| [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp) | `ffd3e20b307a6bf9cc1797a6551d68955dc0b656` | MPL-2.0 project/API/GGUF integration reference |
| [ggml-org/ggml](https://github.com/ggml-org/ggml) | v0.19.0, `30bf8685ed4eb0a47f2b06229543327749904150` | MIT tensor engine and backend dependency |
| [CNChTu/FCPE](https://github.com/CNChTu/FCPE) | Official `torchfcpe 0.0.4` wheel referenced by FCPE.cpp | MIT, copyright 2023 CN_ChiTu; inherited reference notice retained |
| [mreineck/pocketfft](https://github.com/mreineck/pocketfft) | Header copied unmodified from the inspected FCPE.cpp | BSD-3-Clause FFT dependency |
| [librosa/librosa](https://github.com/librosa/librosa) | Installed Python dependency; retained ISC license | HTK Mel filterbank with Slaney area normalization used by the converter |

CMake's fallback ggml v0.19.0 archive SHA256 is `cfb6512adda2853e6500a7c5b23f326987cb4c723e9f8f93c6c5a7e7e4861648`. The default existing-checkout path is `third_party/ggml`. The downloaded checkout is excluded from the source repository.

The native WAV reader and sinc resampler in `src/audio.cpp` are adapted from FCPE.cpp, which identifies its inherited FCPE material. The model computation in `src/rmvpe.cpp` implements RMVPE's architecture; its resource-management and backend scaffolding are adapted from FCPE.cpp. Vulkan build-copy patching and SPIR-V embedding follow FCPE.cpp. The RMVPE persistent GRU shader and its custom-op contract are new here.

The optional speech validation uses [whisper.cpp/samples/jfk.wav](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav), obtained from the existing local FCPE.cpp validation workspace. WAVs and generated tensors stay in ignored local directories. Only hashes and measurements are distributed.

Python packages are development tools for conversion, ONNX export, validation and benchmarking. They are not native runtime dependencies. If bundling them or their CUDA/cuDNN libraries, preserve their own licenses and distribution terms.
