# Metal status and validation plan

**Not built or tested in the Windows environment used for this release.** No Metal speed or accuracy number is claimed.

The CMake option `RMVPE_METAL=ON` passes through to ggml's Metal backend. On macOS, begin with:

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DRMVPE_METAL=ON
cmake --build build-metal -j
build-metal/bin/rmvpe-cli --list-backends
build-metal/bin/rmvpe-cli --model models/rmvpe-f32.gguf --wav input.wav --backend MTL0
```

Use the actual backend name listed by the tool. The fused GRU has a CPU callback but no Metal implementation; expect scheduler fallback unless `--unroll-gru` is used. Convolution selection probes backend support and falls back to the im2col graph when direct Conv2d is unavailable. Unsupported operators should be counted, not hidden.

The `strict F32` claim in the README applies to tested CPU and the patched Vulkan build. ggml's Metal matrix kernels may internally use reduced precision. This project does not import FCPE.cpp's Metal precision and fusion patches, which were designed for a different graph; do not infer equivalent precision from F32 GGUF storage alone.

On Apple hardware, complete these steps before describing Metal as supported:

1. Build and run CPU tests, then verify direct Conv2d, average pooling, transpose-convolution crop and all tensor permutations on Metal.
2. Run `scripts/validate.py --cli build-metal/bin/rmvpe-cli --backend MTL0` against the pinned checkpoint, including an additional real WAV. Test `--unroll-gru` separately. Record CPU fallback nodes, graph splits and allocation sizes.
3. Compare F32 probabilities and voiced/unvoiced output before relaxing numerical limits. If F32 operands are rounded, specialize suitable existing ggml Metal kernels or document a separately measured mixed-precision mode.
4. Implement a Metal persistent GRU only if profiling warrants it. Mirror reset-after PyTorch gate equations, reverse traversal, zero initial state, matrix layout and threadgroup barriers. Reuse the independent GRU tests; the present accelerator GRU test intentionally requires a backend implementation and will reject unsupported Metal custom operations.
5. Compare whole-input host-to-host median/P95 latency with PyTorch CPU/MPS and ONNX Runtime CPU/CoreML as available. Synchronize GPU work and report unsupported-provider fallback explicitly. Check at least 1, 3 and 10 seconds.
6. Exercise short audio, resampling, repeated calls, changing lengths, low confidence and the memory limit. Do not introduce automatic chunking without separately documenting changed bidirectional context.

The Vulkan shader uses a private custom-op contract independent of the GPU language. A Metal implementation can recognize that contract without changing the public C++ API, but source/build integration must remain pinned and tested.
