// SPDX-License-Identifier: MPL-2.0
// Loader, backend scheduling and frontend structure adapted from KCKT0112/FCPE.cpp.
// See NOTICE.md and THIRD_PARTY.md for provenance and retained licenses.
#include "rmvpe/rmvpe.h"
#include "gru.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace rmvpe {
namespace {
void require(bool v, const std::string & s) { if (!v) throw std::runtime_error(s); }
void finite(const std::vector<float> & v) {
    require(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }), "Non-finite input");
}
void init() { static std::once_flag once; std::call_once(once, [] { ggml_backend_load_all(); }); }
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Meta = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
}
std::vector<std::string> devices() {
    init(); std::vector<std::string> names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) names.emplace_back(ggml_backend_dev_name(ggml_backend_dev_get(i)));
    return names;
}
struct Model::Impl {
    Options opts;
    int threads = 1;
    Backend primary{nullptr, ggml_backend_free}, cpu{nullptr, ggml_backend_free};
    Context weights{nullptr, ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    Context graph_ctx{nullptr, ggml_free};
    Scheduler sched{nullptr, ggml_backend_sched_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * output = nullptr, * zero = nullptr;
    size_t cached_frames = 0;
    std::vector<float> basis, window;
    ggml_tensor * tensor(const std::string & name) const {
        auto * t = ggml_get_tensor(weights.get(), name.c_str());
        require(t != nullptr, "Missing tensor: " + name); return t;
    }
    void shape(const std::string & name, std::array<int64_t, 4> dims, bool f32 = false) {
        auto * t = tensor(name);
        require(std::equal(dims.begin(), dims.end(), t->ne), "Invalid shape: " + name);
        require(t->type == GGML_TYPE_F32 || (!f32 && t->type == GGML_TYPE_F16), "Unsupported type: " + name);
    }
    void check_conv(const std::string & p, int in, int out, int k = 3, bool trans = false) {
        shape(p + ".weight", {k,k,trans ? out : in,trans ? in : out});
        shape(p + ".bias", {out,1,1,1}, true);
    }
    void check_block(const std::string & p, int in, int out) {
        check_conv(p + ".conv.0", in, out);
        check_conv(p + ".conv.3", out, out);
        if (in != out) check_conv(p + ".shortcut", in, out, 1);
    }
    std::vector<float> floats(const std::string & name) {
        auto * t = tensor(name);
        require(t->type == GGML_TYPE_F32, "Expected F32: " + name);
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t)); finite(v); return v;
    }
    Impl(const std::string & path, const Options & options) : opts(options) {
        require(opts.threads >= 0 && opts.threads <= 1024, "threads must be 0..1024");
        threads = opts.threads ? opts.threads : static_cast<int>(std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
        ggml_context * raw = nullptr;
        Meta meta(gguf_init_from_file(path.c_str(), {true, &raw}), gguf_free); weights.reset(raw);
        require(meta && weights, "Cannot read GGUF: " + path);
        require(gguf_get_n_tensors(meta.get()) == 272, "Expected 272 RMVPE tensors");
        auto k = gguf_find_key(meta.get(), "general.architecture");
        require(k >= 0 && gguf_get_kv_type(meta.get(), k) == GGUF_TYPE_STRING &&
            std::string(gguf_get_val_str(meta.get(), k)) == "rmvpe", "Expected RMVPE GGUF");
        k = gguf_find_key(meta.get(), "rmvpe.version");
        require(k >= 0 && gguf_get_kv_type(meta.get(), k) == GGUF_TYPE_UINT32 && gguf_get_val_u32(meta.get(), k) == 1, "Unsupported format version");
        shape("input.scale", {1,1,1,1}, true); shape("input.bias", {1,1,1,1}, true);
        int in = 1;
        for (int layer = 0; layer < 5; ++layer) {
            int out = 16 << layer;
            for (int j = 0; j < 4; ++j) { check_block("unet.encoder.layers." + std::to_string(layer) + ".conv." + std::to_string(j), in, out); in = out; }
        }
        for (int layer = 0; layer < 4; ++layer)
            for (int j = 0; j < 4; ++j) { check_block("unet.intermediate.layers." + std::to_string(layer) + ".conv." + std::to_string(j), in, 512); in = 512; }
        for (int layer = 0; layer < 5; ++layer) {
            int out = in / 2;
            auto p = "unet.decoder.layers." + std::to_string(layer);
            check_conv(p + ".conv1.0", in, out, 3, true);
            for (int j = 0; j < 4; ++j) check_block(p + ".conv2." + std::to_string(j), j ? out : 2*out, out);
            in = out;
        }
        check_conv("cnn", 16, 3);
        shape("fc.1.weight", {512,360,1,1}); shape("fc.1.bias", {360,1,1,1}, true);
        for (int d = 0; d < 2; ++d) {
            auto p = "gru." + std::to_string(d);
            shape(p + ".weight_ih", {384,768,1,1}, true); shape(p + ".weight_hh", {256,768,1,1}, true);
            shape(p + ".bias_ih", {768,1,1,1}, true); shape(p + ".bias_hh", {768,1,1,1}, true);
        }
        shape("mel.basis", {513,128,1,1}, true); shape("mel.window", {1024,1,1,1}, true);
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot open model");
        const auto size = static_cast<uint64_t>(file.tellg()), start = static_cast<uint64_t>(gguf_get_data_offset(meta.get()));
        require(start <= size, "Truncated GGUF header");
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            auto off = gguf_get_tensor_offset(meta.get(), i), len = gguf_get_tensor_size(meta.get(), i);
            require(off <= size-start && len <= size-start-off, "Truncated GGUF data");
        }
        init();
        auto * dev = opts.backend == "cpu" ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
            opts.backend == "auto" ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : ggml_backend_dev_by_name(opts.backend.c_str());
        if (!dev && opts.backend == "auto") dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        require(dev != nullptr, "Unavailable backend: " + opts.backend);
        primary.reset(ggml_backend_dev_init(dev, nullptr)); require(bool(primary), "Backend initialization failed");
        std::vector<ggml_backend_t> backends{primary.get()};
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            require(bool(cpu), "CPU backend unavailable"); backends.push_back(cpu.get());
        }
        for (auto * backend : backends) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto set = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
            if (set) set(backend, threads);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(weights.get(), primary.get())); require(bool(buffer), "Weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            auto * t = tensor(gguf_get_tensor_name(meta.get(), i)); std::vector<char> bytes(ggml_nbytes(t));
            file.seekg(static_cast<std::streamoff>(start + gguf_get_tensor_offset(meta.get(), i)));
            file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())); require(bool(file), "Weight read failed");
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        basis = floats("mel.basis"); window = floats("mel.window");
        sched.reset(ggml_backend_sched_new(backends.data(), nullptr, static_cast<int>(backends.size()), 131072, false, true));
        require(bool(sched), "Scheduler initialization failed");
    }
    ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
        if (!opts.fast && w->type == GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F32);
        auto * y = ggml_mul_mat(ctx, w, x); ggml_mul_mat_set_prec(y, GGML_PREC_F32); return y;
    }
    ggml_tensor * conv(ggml_context * ctx, ggml_tensor * x, const std::string & p) {
        auto * w = tensor(p + ".weight");
        if (!opts.fast && w->type == GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F32);
        if (opts.direct_conv && ggml_backend_dev_type(ggml_backend_get_device(primary.get())) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto * direct = ggml_conv_2d_direct(ctx, w, x, 1, 1, static_cast<int>(w->ne[0]/2), static_cast<int>(w->ne[1]/2), 1, 1);
            if (ggml_backend_supports_op(primary.get(), direct))
                return ggml_add(ctx, direct, ggml_reshape_3d(ctx, tensor(p + ".bias"), 1, 1, w->ne[3]));
        }
        auto * col = ggml_im2col(ctx, w, x, 1, 1, static_cast<int>(w->ne[0]/2), static_cast<int>(w->ne[1]/2), 1, 1, true,
                               opts.fast ? GGML_TYPE_F16 : GGML_TYPE_F32);
        // im2col rows are contiguous patches. Multiply patches as the left operand
        // to produce spatial-contiguous WHCN output without a full activation transpose.
        auto * y = matmul(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1]*col->ne[2]),
                         ggml_reshape_2d(ctx, w, w->ne[0]*w->ne[1]*w->ne[2], w->ne[3]));
        y = ggml_reshape_3d(ctx, y, x->ne[0], x->ne[1], w->ne[3]);
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, tensor(p + ".bias"), 1, 1, w->ne[3]));
        ggml_set_name(y, p.c_str()); return y;
    }
    ggml_tensor * block(ggml_context * ctx, ggml_tensor * x, const std::string & p) {
        auto * y = ggml_relu(ctx, conv(ctx, x, p + ".conv.0"));
        y = ggml_relu(ctx, conv(ctx, y, p + ".conv.3"));
        return ggml_add(ctx, y, x->ne[2] == y->ne[2] ? x : conv(ctx, x, p + ".shortcut"));
    }
    ggml_tensor * gru_unroll(ggml_context * ctx, ggml_tensor * x, int direction) {
        auto p = "gru." + std::to_string(direction);
        auto * h = zero;
        std::vector<ggml_tensor *> states(static_cast<size_t>(x->ne[1]));
        for (int64_t step = 0; step < x->ne[1]; ++step) {
            auto t = direction ? x->ne[1] - 1 - step : step;
            auto * a = ggml_view_1d(ctx, x, 768, t*x->nb[1]);
            auto * b = ggml_add(ctx, matmul(ctx, tensor(p + ".weight_hh"), h), tensor(p + ".bias_hh"));
            auto view = [&](ggml_tensor * v, int i) { return ggml_view_1d(ctx, v, 256, i*256*sizeof(float)); };
            auto * r = ggml_sigmoid(ctx, ggml_add(ctx, view(a,0), view(b,0)));
            auto * z = ggml_sigmoid(ctx, ggml_add(ctx, view(a,1), view(b,1)));
            auto * n = ggml_tanh(ctx, ggml_add(ctx, view(a,2), ggml_mul(ctx, r, view(b,2))));
            h = ggml_add(ctx, n, ggml_mul(ctx, z, ggml_sub(ctx, h, n)));
            states[static_cast<size_t>(t)] = ggml_reshape_2d(ctx, h, 256, 1);
        }
        // Balanced concatenation avoids quadratic copies across sequence length.
        while (states.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t i = 0; i < states.size(); i += 2)
                next.push_back(i+1 < states.size() ? ggml_concat(ctx, states[i], states[i+1], 1) : states[i]);
            states = std::move(next);
        }
        return states[0];
    }
    void build(size_t frames) {
        if (cached_frames == frames) return;
        cached_frames = 0; ggml_backend_sched_reset(sched.get());
        const size_t nodes = opts.unroll_gru ? 4096 + 64*frames : 4096;
        require(nodes <= 131072, "Unrolled GRU exceeds graph capacity; use fused mode or shorter input");
        graph_ctx.reset(ggml_init({nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        require(bool(graph_ctx), "Graph metadata allocation failed"); auto * ctx = graph_ctx.get();
        graph = ggml_new_graph_custom(ctx, nodes, false);
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, static_cast<int64_t>(frames), 1); ggml_set_input(input); ggml_set_name(input, "mel_input");
        auto * x = ggml_add(ctx, ggml_mul(ctx, input, tensor("input.scale")), tensor("input.bias"));
        std::vector<ggml_tensor *> skip;
        for (int layer = 0; layer < 5; ++layer) {
            for (int j = 0; j < 4; ++j) x = block(ctx, x, "unet.encoder.layers." + std::to_string(layer) + ".conv." + std::to_string(j));
            skip.push_back(x); x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
        }
        for (int layer = 0; layer < 4; ++layer)
            for (int j = 0; j < 4; ++j) x = block(ctx, x, "unet.intermediate.layers." + std::to_string(layer) + ".conv." + std::to_string(j));
        for (int layer = 0; layer < 5; ++layer) {
            auto p = "unet.decoder.layers." + std::to_string(layer);
            auto * w = tensor(p + ".conv1.0.weight");
            if (!opts.fast && w->type == GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F32);
            auto * up = ggml_conv_transpose_2d_p0(ctx, w, x, 2);
            // p=1, output_padding=1: remove the leading row/column from p0 output.
            x = ggml_cont(ctx, ggml_view_3d(ctx, up, 2*x->ne[0], 2*x->ne[1], up->ne[2], up->nb[1], up->nb[2], up->nb[0]+up->nb[1]));
            x = ggml_relu(ctx, ggml_add(ctx, x, ggml_reshape_3d(ctx, tensor(p + ".conv1.0.bias"), 1,1,x->ne[2])));
            x = ggml_concat(ctx, x, skip[4-layer], 2);
            for (int j = 0; j < 4; ++j) x = block(ctx, x, p + ".conv2." + std::to_string(j));
        }
        x = conv(ctx, x, "cnn"); // [mel, time, channel] -> [channel*mel, time]
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 0,2,1,3)), 384, static_cast<int64_t>(frames));
        std::array<ggml_tensor *, 2> hs;
        zero = nullptr;
        if (opts.unroll_gru) { zero = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256); ggml_set_input(zero); }
        for (int d = 0; d < 2; ++d) {
            auto p = "gru." + std::to_string(d);
            auto * projected = ggml_add(ctx, matmul(ctx, tensor(p + ".weight_ih"), x), tensor(p + ".bias_ih"));
            auto * recurrent = tensor(p + ".weight_hh");
            if (!opts.unroll_gru && std::string(ggml_backend_name(primary.get())).find("Vulkan") == 0 && !std::getenv("RMVPE_VK_GRU_ROW_MAJOR"))
                recurrent = ggml_cont(ctx, ggml_transpose(ctx, recurrent));
            hs[d] = opts.unroll_gru ? gru_unroll(ctx, projected, d) : gru_fused(ctx, projected, recurrent, tensor(p + ".bias_hh"), d != 0);
            ggml_set_name(hs[d], d ? "gru_reverse" : "gru_forward");
        }
        x = ggml_concat(ctx, hs[0], hs[1], 0);
        output = ggml_sigmoid(ctx, ggml_add(ctx, matmul(ctx, tensor("fc.1.weight"), x), tensor("fc.1.bias")));
        ggml_set_name(output, "probabilities"); ggml_set_output(output); ggml_build_forward_expand(graph, output);
        require(ggml_backend_sched_alloc_graph(sched.get(), graph), "Graph allocation failed; use shorter input");
        cached_frames = frames;
    }
};
Model::Model(const std::string & path, const Options & opts) : impl_(new Impl(path, opts)) {}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;
std::string Model::backend_name() const { return ggml_backend_name(impl_->primary.get()); }
ExecutionInfo Model::execution_info() const {
    ExecutionInfo info; const auto & m = *impl_; if (!m.graph) return info;
    info.graph_nodes = ggml_graph_n_nodes(m.graph); info.graph_splits = ggml_backend_sched_get_n_splits(m.sched.get());
    for (int i = 0; i < info.graph_nodes; ++i) {
        auto * n = ggml_graph_node(m.graph, i);
        if (n->op == GGML_OP_NONE || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
        auto * b = ggml_backend_sched_get_tensor_backend(m.sched.get(), n); if (!b) continue;
        if (ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU) ++info.cpu_compute_nodes;
        else ++info.accelerator_compute_nodes;
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(m.sched.get()); ++i)
        info.compute_buffer_bytes += ggml_backend_sched_get_buffer_size(m.sched.get(), ggml_backend_sched_get_backend(m.sched.get(), i));
    return info;
}
std::vector<float> Model::probabilities(const std::vector<float> & mel) {
    require(!mel.empty() && mel.size() % (128*32) == 0, "Mel must contain a positive multiple of 32 frames, each with 128 bins");
    require(mel.size()/128 <= 16384, "At most 16384 padded frames per call"); finite(mel);
    auto & m = *impl_; m.build(mel.size()/128);
    ggml_backend_tensor_set(m.input, mel.data(), 0, mel.size()*sizeof(float));
    if (m.zero) { std::array<float,256> zero{}; ggml_backend_tensor_set(m.zero, zero.data(), 0, sizeof(zero)); }
    require(ggml_backend_sched_graph_compute(m.sched.get(), m.graph) == GGML_STATUS_SUCCESS, "Graph execution failed");
    std::vector<float> out(mel.size()/128*360); ggml_backend_tensor_get(m.output, out.data(), 0, out.size()*sizeof(float));
    finite(out); return out;
}
std::vector<float> Model::mel(const std::vector<float> & audio, int sample_rate) const {
    require(!audio.empty(), "Empty audio"); auto a = resample(audio, sample_rate, 16000); finite(a);
    size_t frames = (a.size()/160 + 1 + 31)/32*32;
    require(frames <= 16384, "At most 16384 padded frames per call");
    a.resize(frames*160-160, 0.0f); // exact upstream right-zero-padding formula
    std::vector<float> out(frames*128);
    const auto & m = *impl_;
    std::array<std::pair<int,int>,128> support;
    for (int bin = 0; bin < 128; ++bin) {
        int lo = 0, hi = 513;
        while (lo < hi && m.basis[bin*513+lo] == 0) ++lo;
        while (hi > lo && m.basis[bin*513+hi-1] == 0) --hi;
        support[bin] = {lo, hi};
    }
    auto work = [&](size_t begin, size_t end) {
        std::array<float,1024> time; std::array<std::complex<float>,513> spectrum; std::array<float,513> mag;
        for (size_t t = begin; t < end; ++t) {
            for (int j = 0; j < 1024; ++j) {
                int64_t k = static_cast<int64_t>(t*160)+j-512;
                if (k < 0) k = -k;
                if (k >= static_cast<int64_t>(a.size())) k = 2*static_cast<int64_t>(a.size())-2-k;
                time[j] = a[static_cast<size_t>(k)]*m.window[j];
            }
            pocketfft::r2c<float>({1024}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0, true, time.data(), spectrum.data(), 1.0f);
            for (int k = 0; k < 513; ++k) mag[k] = std::abs(spectrum[k]);
            for (int bin = 0; bin < 128; ++bin) {
                float sum = 0; for (int k = support[bin].first; k < support[bin].second; ++k) sum += m.basis[bin*513+k]*mag[k];
                out[t*128+bin] = std::log(std::max(sum,1e-5f));
            }
        }
    };
    const size_t workers = std::min<size_t>(m.threads, std::max<size_t>(1,frames/64)); std::vector<std::future<void>> jobs;
    for (size_t i = 1; i < workers; ++i) jobs.emplace_back(std::async(std::launch::async, work, frames*i/workers, frames*(i+1)/workers));
    work(0,frames/workers); for (auto & job : jobs) job.get(); return out;
}
Result decode(const std::vector<float> & probabilities, float threshold) {
    require(!probabilities.empty() && probabilities.size()%360 == 0, "Expected [frames,360] probabilities"); finite(probabilities);
    require(std::isfinite(threshold) && threshold >= 0 && threshold <= 1, "Threshold must be 0..1");
    Result result; size_t frames = probabilities.size()/360; result.f0.resize(frames); result.confidence.resize(frames);
    for (size_t t = 0; t < frames; ++t) {
        const auto * p = probabilities.data()+t*360; int center = static_cast<int>(std::max_element(p,p+360)-p);
        result.confidence[t] = p[center]; if (p[center] <= threshold) continue;
        double sum = 0, weight = 0;
        for (int j = std::max(0,center-4); j < std::min(360,center+5); ++j) { sum += p[j]*(20.0*j+1997.3794084376191); weight += p[j]; }
        if (weight > 0) result.f0[t] = static_cast<float>(10*std::exp2(sum/weight/1200));
    }
    return result;
}
Result Model::infer(const std::vector<float> & audio, int sample_rate, float threshold) {
    auto a = resample(audio,sample_rate,16000); auto p = probabilities(mel(a,16000)); p.resize((a.size()/160+1)*360); return decode(p,threshold);
}
} // namespace rmvpe
