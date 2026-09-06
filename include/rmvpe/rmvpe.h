// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rmvpe {
struct Config {
    int sample_rate = 16000, hop_size = 160, n_fft = 1024, win_size = 1024, mel_bins = 128;
};
struct Options {
    std::string backend = "cpu";
    int threads = 0;
    bool unroll_gru = false; // Portable all-ggml graph; slower than fused CPU recurrence.
    bool fast = false; // F16 im2col/matmul operands; may change pitch decisions.
    bool direct_conv = true; // Native GPU convolution; CPU uses F32 im2col.
};
struct ExecutionInfo {
    int graph_nodes = 0, graph_splits = 0, cpu_compute_nodes = 0, accelerator_compute_nodes = 0;
    std::size_t compute_buffer_bytes = 0;
};
struct Audio { std::vector<float> samples; int sample_rate = 0; };
struct Result { std::vector<float> f0, confidence; };
std::vector<std::string> devices();
Audio read_wav(const std::string & path);
std::vector<float> resample(const std::vector<float> & audio, int from, int to);
Result decode(const std::vector<float> & probabilities, float threshold = 0.03f);

// One instance per concurrent caller. Matrices are row-major [frames, channels].
// probabilities() accepts a positive multiple of 32 frames, including any padding.
// mel() zero-pads waveform before a centered reflective STFT as in upstream audio inference.
class Model {
public:
    explicit Model(const std::string & path, const Options & options = {});
    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    std::string backend_name() const;
    ExecutionInfo execution_info() const;
    std::vector<float> mel(const std::vector<float> & audio, int sample_rate) const;
    std::vector<float> probabilities(const std::vector<float> & mel);
    Result infer(const std::vector<float> & audio, int sample_rate, float threshold = 0.03f);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace rmvpe
