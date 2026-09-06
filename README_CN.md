# RMVPE.cpp

[English README](README.md)

基于 **C++17、ggml 和 GGUF** 的 [yxlllc/RMVPE](https://github.com/yxlllc/RMVPE) 原生音高提取实现，支持上游 **230917 E2E0** 模型。从 WAV 读取、重采样、log-Mel 前端，到残差 U-Net、双向 GRU 和局部加权音高解码，均由原生代码完成。推理不依赖 Python、PyTorch 或 ONNX Runtime。

**Windows CPU 和 Vulkan 已实现并实测。** Vulkan 包含新增的持久化 GRU 算子，默认采用严格 F32 运算。Metal 和原生 ggml CUDA 预留构建开关，尚未验证；PyTorch/ONNX Runtime 的 CUDA 测试不等于 ggml CUDA 验证。详见 [后端说明](docs/BACKENDS.md) 和 [Metal 待办](docs/METAL.md)。

项目参考并复用了 [KCKT0112/FCPE.cpp](https://github.com/KCKT0112/FCPE.cpp) 的加载器、后端调度、音频和构建组织，同时参考 [game.cpp](https://github.com/KakaruHayate/game.cpp) 与 [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp)。FCPE.cpp 的上游是 **CN_ChiTu 的 [CNChTu/FCPE](https://github.com/CNChTu/FCPE)**。准确来源、版本和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 构建

依赖 CMake 3.18+、C/C++17 编译器、Git，可选 Ninja。ggml 固定为 **v0.19.0**，CMake 自动下载经过校验的归档，或使用 `third_party/ggml` / `RMVPE_GGML_SOURCE_DIR` 指定的源码。

Windows 下在 **Visual Studio Developer PowerShell** 中运行：

```powershell
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 8

$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.341.1'
cmake -S . -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release -DRMVPE_VULKAN=ON -DRMVPE_TEST_BACKEND=Vulkan0
cmake --build build-vulkan -j 8
ctest --test-dir build-vulkan --output-on-failure
.\build-vulkan\bin\rmvpe-cli.exe --list-backends
```

程序和所需 ggml DLL 位于 `build*/bin`。分发程序时保留对应 DLL。向其他 CPU 分发时使用 `-DGGML_NATIVE=OFF -DRMVPE_CPU_AVX2=OFF`，并根据目标机器选择 ggml 指令集开关。本机测试构建使用 AVX2/FMA。

Vulkan 集成只生成构建目录内的 ggml 源码副本，不修改依赖源码。补丁锚点不匹配时配置会报错，请使用固定版本。

## 下载、解包和转换

```powershell
python -m pip install -r requirements.txt
git clone https://github.com/yxlllc/RMVPE.git reference/RMVPE
git -C reference/RMVPE checkout 0aabafba18289ca938a73af0b0297686abf4922d
python scripts/download_model.py
python scripts/convert_rmvpe.py models/original/model.pt models/rmvpe-f32.gguf
python scripts/convert_rmvpe.py models/original/model.pt models/rmvpe-f16.gguf --dtype f16
```

下载脚本获取指定的 [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，校验 SHA256 后只解包 `model.pt`。转换时折叠推理模式 BatchNorm，移除 E2E0 不执行的 timbre-filter 权重，输出 272 个张量。F32 文件为 **361,883,776 字节**，F16 文件为 **183,112,672 字节**。F16 仅压缩卷积和投影权重，GRU、偏置及前端张量保持 F32。GGUF 元数据和相邻 JSON 保存来源、精度及哈希。

F16 存储在真实语音 JFK 样本中出现约 **12.39 音分**的局部偏差，超过预设的 5 音分限值，尽管清浊音判断一致。它作为实验性的体积选项保留，精度敏感用途建议使用 F32，详见 [验证文档](docs/VALIDATION.md)。

若受限 Windows 账号导入 librosa 时卡在 Numba 缓存，可先设置可写目录：

```powershell
$env:NUMBA_CACHE_DIR = "$PWD\work\numba"
```

**许可提醒：本项目贡献代码使用 MPL-2.0，但本次检查的 RMVPE 原始仓库与权重归档均未包含明确许可文件。不能因此将权重声明为 MPL、MIT 或公有领域。** 本仓库不提交原始模型、转换权重或上游 Python 代码；重新分发模型前请确认相应权利与许可。详见 [NOTICE.md](NOTICE.md)。

## 使用

```powershell
.\build-cpu\bin\rmvpe-cli.exe --model models/rmvpe-f32.gguf --wav input.wav --output pitch.csv --threads 4
.\build-vulkan\bin\rmvpe-cli.exe --model models/rmvpe-f32.gguf --wav input.wav --output pitch.csv --backend Vulkan0 --threads 4
```

CSV 输出 `time_s,f0_hz,confidence`，步长 10 ms，清音 F0 为 0。默认阈值 **0.03**，使用上游 `confidence > threshold` 规则。支持常用整数 PCM 与浮点 RIFF WAV；多声道取平均，其他采样率使用 sinc 重采样到 16 kHz。

普通调用执行一次网络。加入 `--warmup 3 --runs 10` 可测试复用会话，stdout 输出 JSON，stderr 输出后端日志。首次分配和执行与后续测量分别记录。`--benchmark-audio` 测量内存中的音频样本到 F0，包含前端及解码，不含文件读写。

`--mel input.f32` 输入小端 F32 的 **[frames,128]** 矩阵，帧数必须是正的 32 倍数，调用者负责补齐。`--probabilities output.f32` 输出 **[补齐帧数,360]** 概率，`--dump-mel output.f32` 导出前端结果。WAV 的 CSV 裁剪为 `floor(重采样后样本数/160)+1` 帧。

音频前端遵循上游 `infer_from_audio`：先在波形右侧补零，再做 centered reflect STFT、周期 Hann 窗、幅度谱、HTK Mel / Slaney 面积归一化、`1e-5` 截断和自然对数。上游单独的 `mel2hidden` 使用不同的 Mel 反射补齐方式，原始 Mel 接口不会替调用者混用两者。

每次最多接收 16,384 个补齐帧，实际长度受设备内存约束。默认整段计算以保留双向 GRU 上下文，不自动切块。暂不实现流式、Viterbi、训练、任意架构检查点或量化 GGUF。

C++ 调用示例见英文 README 和 [examples/basic.cpp](examples/basic.cpp)，链接目标为 `rmvpe::rmvpe`。每个并发调用者使用独立实例；同一实例可以顺序接收不同长度输入。

## 实测速度

Windows 10.0.29648、Ryzen 7 5700X、RTX 4060 8 GB、NVIDIA 620.02、Vulkan SDK 1.4.341.1；batch=1，CPU 4 线程，预热 3 次，测量 10 次。下表为中位耗时，单位 ms，统一测量 **主机 F32 log-Mel 输入到主机 F32 概率输出**，包含 GPU 传输和同步，排除加载、编译、音频前端与解码。

| 引擎 | 1 秒 / 128 帧 | 3 秒 / 320 帧 | 10 秒 / 1024 帧 |
| --- | ---: | ---: | ---: |
| ggml CPU F32 | 165.83 | 390.67 | 1249.78 |
| **ggml Vulkan 严格 F32** | **17.71** | **23.51** | **57.03** |
| ggml Vulkan 混合精度 | 18.16 | 19.26 | 42.68 |
| PyTorch 2.7.1 CPU | 101.22 | 260.42 | 464.08 |
| PyTorch 2.7.1 CUDA | 31.56 | 39.12 | 79.84 |
| ONNX Runtime 1.22.0 CPU | 42.70 | 84.04 | 271.46 |
| ONNX Runtime 1.22.0 CUDA | 23.20 | 34.38 | 75.21 |

严格 F32 Vulkan 的 10 秒输入 RTF 为 **0.00570**，约 **175 倍实时**。ggml CPU 明显慢于 ORT CPU；混合精度有独立误差范围且短输入未必更快，所以不作为默认配置。输入为可复现的合成谐波信号，这些数据不是歌声音高准确率评测。

19 个端到端验证样本、累计 2,182 帧：F32 CPU 和 Vulkan 全部通过，最大音高误差分别约 0.00390 / 0.00367 音分；混合精度最大约 2.41 音分。所有配置在本次样本中的清浊音判断均与上游一致，但 F16 的 12.39 音分偏差明确保留为失败。

[性能文档](docs/PERFORMANCE.md) 保存完整原始数据、优化前后对比和复现方法；[验证文档](docs/VALIDATION.md) 记录边界、重采样和真实录音误差及限制。

## 许可和来源

本项目贡献使用 **[MPL-2.0](LICENSE)**。重新分发时保留 [NOTICE.md](NOTICE.md)、[THIRD_PARTY.md](THIRD_PARTY.md) 和适用第三方许可。分发 MPL 覆盖的二进制时，按协议提供对应源码和修改。pocketfft、ggml、继承的 FCPE 材料和模型资产分别保留原有条件。

感谢 RMVPE 作者及 yxlllc、ggml、KCKT0112/FCPE.cpp、CN_ChiTu/FCPE、KakaruHayate/game.cpp、KakaruHayate/pc-nsf-hifigan.cpp 和 pocketfft 贡献者。本项目是独立实现，按原样提供。
