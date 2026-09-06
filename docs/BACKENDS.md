# Backend implementation and acceleration

## Architecture and operator mapping

The supported checkpoint is E2E0(4, 1, (2,2)): five encoders, four intermediate stages, five decoders, four residual blocks per stage, channels 16-512, then a 3-channel convolution, one bidirectional 256-unit GRU and a 512-to-360 sigmoid projection. The registered timbre filter is unused in E2E0 and omitted from GGUF. Evaluation BatchNorm is folded at conversion, except the initial scalar affine normalization.

| Operation | CPU | Vulkan default |
| --- | --- | --- |
| Conv2d | Explicit F32 im2col + matmul | Existing ggml direct/implicit-GEMM Conv2d |
| BatchNorm | Folded into weights/bias at conversion | Same |
| ReLU, add, concat | ggml | ggml |
| AvgPool2d 2x2 | ggml | ggml |
| ConvTranspose2d 3x3, stride 2 | ggml p0 operator + view/copy crop | Existing ggml operator + crop |
| GRU input projections | Batched ggml matmul | Batched ggml matmul |
| GRU recurrent scan | Project fused CPU custom operator | Project persistent compute shader |
| Final projection / sigmoid | ggml | ggml |
| Audio frontend / decode | Native CPU | Native CPU outside network graph |

The transpose-convolution crop is deliberate: ggml's p0 output has size 2*input+1. Removing the leading row and column gives PyTorch padding=1/output_padding=1, including the last row/column. A symmetric crop would be wrong.

## New GRU operator

ggml v0.19.0 already provides all required convolution and elementwise primitives. A new public ggml enum is not required for functionality: `--unroll-gru` expresses reset/update/candidate recurrence with existing operators, balanced concatenation, and full backward context. However, its per-frame graph size and dispatch overhead justify a fused operator.

`src/gru.cpp` implements a tagged `GGML_OP_CUSTOM` with a normal CPU callback. The Vulkan build adds support only for that tag and exact F32 shapes. No generic custom operation is advertised as GPU-supported. The tag occupies bytes 48-55 of the pinned version's op_params, beyond its callback payload; this is a private versioned contract, not a change to public ggml ABI.

The shader runs one 256-thread workgroup per direction for the full sequence. Shared memory stores hidden state; barriers ensure each recurrence step consumes the complete previous state before replacing it. Input projections are batched outside the scan. A transposed recurrent matrix makes neighboring invocations read adjacent weights. The graph's normal dependency tracking, descriptors, scheduler and command submission remain in use.

CPU has scalar and MSVC AVX2/FMA dot implementations. The tested build targets AVX2. The CPU path and shader are checked against an independent F64 recurrence for 32 combinations of direction, matrix layout and length, plus repeated execution to catch stale hidden state.

A device with insufficient workgroup support or an unpatched ggml build falls back to the CPU callback. `execution_info()` and CLI JSON report actual CPU/accelerator compute-node counts and graph splits. The tested default Vulkan network has **zero CPU compute nodes**. The audio frontend and decoder still run on CPU and are outside those counts.

## Precision and memory policy

ggml v0.19.0 can round F32 operands to F16 in cooperative matrix paths even when accumulation precision is F32. The generated Vulkan backend disables those paths by default. F16 GGUF weights are expanded to F32 in the strict graph; this changes storage accuracy, not the intended arithmetic mode.

On discrete RTX 4060, upstream host-visible device-memory allocation caused a large slowdown. The build copy prefers the existing non-host-visible path. Upstream's UMA handling remains in place. This policy is an empirical choice for the tested hardware and can be overridden for other devices.

| Switch | Purpose |
| --- | --- |
| `--im2col` / `Options.direct_conv=false` | Use the earlier convolution graph for ablation |
| `--unroll-gru` | Use portable ggml recurrence instead of the fused scan |
| `--fast` | Permit F16 graph intermediates/storage operands |
| `RMVPE_VK_FAST=1` | Permit upstream mixed/cooperative Vulkan arithmetic |
| `RMVPE_VK_CPU_GRU=1` | Force fused recurrent operators to CPU |
| `RMVPE_VK_GRU_ROW_MAJOR=1` | Disable the coalesced recurrent-weight layout |
| `RMVPE_VK_HOST_VISIBLE=1` | Restore upstream host-visible memory policy |

Environment switches are read by presence; remove the variable to disable it. GPU arithmetic policy is initialized once per process, so set it **before** loading any backend. For the measured mixed path, set `RMVPE_VK_FAST=1` and pass `--fast`. Do not treat that path as strict F32. External GGML_VK_DISABLE_* variables can still restrict fast mode.

## Remaining opportunities

- CPU needs better convolution implementations or tuned GEMM integration. The current CPU F32 graph is correct but does not compete with ORT's optimized convolutions. F16 im2col and GRU unrolling did not provide a reliable improvement in the measured CPU configurations.
- Vulkan's recurrent scan occupies only one workgroup per direction. More cooperative matrix-vector designs or gate fusion may help; sequence steps are intrinsically dependent.
- Bias/ReLU and residual fusion can reduce small dispatches, especially at short lengths. Further shader work should preserve shape/alias checks and be justified by measured gain.
- F16 storage currently incurs F32 expansion in strict mode. Prepacking or one-time expansion could improve warm latency/memory without changing the precision contract.
- F32 CPU/Vulkan numerical validation is stronger than F16/mixed validation. No quantized weight formats are advertised.
- Metal needs dedicated Apple hardware validation and potentially a persistent GRU shader. Native ggml CUDA also remains untested.

Build patches fail on missing/non-unique anchors rather than silently applying to an incompatible ggml version. The integration never rewrites the dependency checkout.
