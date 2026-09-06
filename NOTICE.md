# Notices, provenance and distribution

RMVPE.cpp is an independent C++17 / ggml implementation of the **230917 E2E0** model from [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE). It is not an official RMVPE release and is not endorsed by the upstream authors. Project names identify their respective upstream projects; no trademark rights are granted.

## Project license

This repository's original source, build integration, scripts, tests and documentation are licensed under the **Mozilla Public License 2.0**, except where an individual file or a third-party notice states otherwise. The complete terms are in [LICENSE](LICENSE).

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at <https://mozilla.org/MPL/2.0/>.

MPL-2.0 covers this port's contributions and adaptations of MPL-covered FCPE.cpp infrastructure. It does not replace the MIT, BSD or ISC grants on third-party material and does not grant a license to the original RMVPE checkpoint.

## RMVPE source and checkpoint

| Item | Provenance |
| --- | --- |
| Numerical and architectural reference | [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE), commit `0aabafba18289ca938a73af0b0297686abf4922d` |
| Authors credited by the source | Original RMVPE project contributors; upstream repository maintained by yxlllc |
| Release | [230917](https://github.com/yxlllc/RMVPE/releases/tag/230917) |
| Exact archive | [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip) |
| Archive member | `model.pt`, containing a `model` state dictionary |
| Archive SHA256 | `54ae40d9c066d998b94574f6ef0deea19ed1565bd655b3f0d9b1ad612fb5309c` |
| Checkpoint SHA256 | `19dc1809cf4cdb0a18db93441816bc327e14e5644b72eeaae5220560c6736fe2` |

**No explicit license file was present in the inspected upstream repository or the downloaded archive.** This is a statement about those inspected materials, not an assertion that the model is public domain, MIT or MPL. The local original checkout, archive, checkpoint and converted GGUF assets are not included in this source distribution. Obtain any required upstream permission and follow applicable terms before redistributing those assets. GGUF conversion does not create or change their license.

The runtime implements the model's mathematical operations in C++; it does not bundle or invoke upstream Python source. Conversion and verification scripts load a separately obtained upstream checkout. The GGUF metadata and adjacent JSON manifest retain source and checkpoint hashes, storage precision and the license-status note.

## Reference projects and FCPE origin

- [KCKT0112/FCPE.cpp](https://github.com/KCKT0112/FCPE.cpp), MPL-2.0: C++ model-loader/backend-scheduler structure, WAV reader, resampling implementation, pocketfft frontend organization, CMake build-copy patch integration and notice organization were adapted. This relationship is source reuse as well as design reference.
- FCPE.cpp itself ports [CNChTu/FCPE](https://github.com/CNChTu/FCPE), by **CN_ChiTu**, copyright 2023, MIT. Its numerical reference is the official `torchfcpe 0.0.4` release. We retain [licenses/FCPE-MIT.txt](licenses/FCPE-MIT.txt) for inherited material. RMVPE.cpp uses the **RMVPE** checkpoint, not FCPE weights.
- [KakaruHayate/pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp), MPL-2.0: project organization, native audio inference and distribution-notice reference; also the earlier source of the pocketfft header carried through FCPE.cpp.
- [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp), MPL-2.0: ggml/GGUF project structure and public API reference.

The vocoder's separately dual-licensed patch directory, NSF-HiFiGAN/DiffSinger weights, datasets and model-specific non-commercial conditions are not imported into this project.

## Third-party licenses

| Component | License | Retained notice |
| --- | --- | --- |
| ggml v0.19.0 | MIT | [ggml-MIT.txt](licenses/ggml-MIT.txt) |
| pocketfft, unmodified vendored header | BSD-3-Clause | Header notice and [pocketfft-BSD-3-Clause.txt](licenses/pocketfft-BSD-3-Clause.txt) |
| Inherited FCPE reference material | MIT | [FCPE-MIT.txt](licenses/FCPE-MIT.txt) |
| librosa, converter's filterbank dependency | ISC | [librosa-ISC.txt](licenses/librosa-ISC.txt) |

The original ggml checkout is unchanged. Vulkan integration produces a build-local source copy and retains the upstream MIT-covered implementation; this project's integration and new GRU shader are MPL-2.0. The pocketfft header is not relicensed under MPL. See [THIRD_PARTY.md](THIRD_PARTY.md) for exact inspected revisions.

## Distribution reminders

- Keep LICENSE, this notice, THIRD_PARTY.md and the relevant `licenses/` files with redistributions. CMake installs them under `share/doc/rmvpe`.
- When distributing executable forms of MPL-covered files, make the corresponding Source Code Form, including your changes, available under MPL-2.0 as required by section 3.2 and explain how recipients can obtain it. The project source URL is <https://github.com/KCKT0112/RMVPE.cpp>; modified binaries must identify the source corresponding to those modifications.
- MPL uses file-level copyleft and permits Larger Works under other terms subject to its requirements. The license text controls.
- Keep model/recording/data permissions separate from this project's source-code license. Benchmark recordings and weights are not part of this distribution.

## Warranty and validation limits

The software is provided **as is**, without warranty, subject to the applicable license disclaimers. Recorded measurements apply to the listed hardware, inputs and versions. F16 storage and mixed arithmetic can alter confidence or pitch decisions. Windows CPU and Vulkan were tested; Metal and native ggml CUDA have not been validated. PyTorch CUDA and ONNX Runtime CUDA results do not validate ggml CUDA. Consult the validation and performance documents before extrapolating results.
