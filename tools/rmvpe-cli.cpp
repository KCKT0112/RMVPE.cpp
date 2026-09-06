// SPDX-License-Identifier: MPL-2.0
#include "rmvpe/rmvpe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
static std::vector<float> read(const std::string & path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate); if (!f) throw std::runtime_error("Cannot read " + path);
    auto n = f.tellg(); if (n <= 0 || n%4) throw std::runtime_error("Expected nonempty raw little-endian F32 file");
    std::vector<float> out(static_cast<size_t>(n)/4); f.seekg(0); f.read(reinterpret_cast<char *>(out.data()),n);
    if (!f) throw std::runtime_error("Truncated input"); return out;
}
static void write(const std::string & path, const std::vector<float> & data) {
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char *>(data.data()),static_cast<std::streamsize>(data.size()*4));
    if (!f) throw std::runtime_error("Cannot write " + path);
}
int main(int argc, char ** argv) try {
    rmvpe::Options opts;
    std::string model, wav, mel_path, output, prob, dump;
    int warmup = 1, runs = 1; float threshold = 0.03f;
    bool benchmark_requested=false, benchmark_audio=false;
    rmvpe::Audio audio;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string { if (++i == argc) throw std::runtime_error("Missing value for " + arg); return argv[i]; };
        auto integer = [&](const std::string & s) { size_t n=0; int x=std::stoi(s,&n); if(n!=s.size()) throw std::runtime_error("Invalid integer"); return x; };
        if (arg == "--help" || arg == "-h") {
            std::cout << "rmvpe-cli --model MODEL.gguf (--wav input.wav | --mel input.f32) [options]\n"
                "  --backend cpu|auto|Vulkan0|MTL0 --threads N --output pitch.csv\n"
                "  --probabilities output.f32 --dump-mel mel.f32 --threshold 0.03\n"
                "  --warmup N --runs N --benchmark-audio --fast --unroll-gru --im2col --list-backends\n"
                "Mel files: little-endian F32 [frames,128]; frames must be divisible by 32.\n"
                "Timing JSON is written to stdout; logs go to stderr.\n"; return 0;
        } else if (arg == "--list-backends") { for (auto & d : rmvpe::devices()) std::cout << d << '\n'; return 0; }
        else if (arg == "--model") model = value();
        else if (arg == "--wav") wav = value();
        else if (arg == "--mel") mel_path = value();
        else if (arg == "--backend") opts.backend = value();
        else if (arg == "--threads") opts.threads = integer(value());
        else if (arg == "--output") output = value();
        else if (arg == "--probabilities") prob = value();
        else if (arg == "--dump-mel") dump = value();
        else if (arg == "--warmup") { warmup = integer(value()); benchmark_requested=true; }
        else if (arg == "--runs") { runs = integer(value()); benchmark_requested=true; }
        else if (arg == "--benchmark-audio") { benchmark_audio=true; benchmark_requested=true; }
        else if (arg == "--threshold") { auto s=value(); size_t n=0; threshold=std::stof(s,&n); if(n!=s.size()) throw std::runtime_error("Invalid threshold"); }
        else if (arg == "--unroll-gru") opts.unroll_gru = true;
        else if (arg == "--fast") opts.fast = true;
        else if (arg == "--direct-conv") opts.direct_conv = true;
        else if (arg == "--im2col") opts.direct_conv = false;
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (model.empty() || wav.empty() == mel_path.empty()) throw std::runtime_error("Specify --model and exactly one of --wav / --mel; see --help");
    if (warmup < 0 || warmup > 1000 || runs <= 0 || runs > 10000) throw std::runtime_error("Invalid warmup/runs");
    if (!std::isfinite(threshold) || threshold < 0 || threshold > 1) throw std::runtime_error("Threshold must be 0..1");
    auto start = Clock::now(); rmvpe::Model engine(model,opts); auto loaded = Clock::now();
    std::vector<float> mel; size_t output_frames=0; double duration=0;
    if (!wav.empty()) {
        audio = rmvpe::read_wav(wav); const auto & a=audio; duration=static_cast<double>(a.samples.size())/a.sample_rate;
        output_frames=static_cast<size_t>((static_cast<uint64_t>(a.samples.size())*16000+a.sample_rate-1)/a.sample_rate)/160+1;
        mel=engine.mel(a.samples,a.sample_rate);
    } else { mel=read(mel_path); output_frames=mel.size()/128; duration=output_frames*0.01; }
    auto prepared=Clock::now();
    if (!dump.empty()) write(dump,mel);
    auto first_start=Clock::now(); auto p=engine.probabilities(mel); auto first_end=Clock::now();
    if(benchmark_audio && wav.empty()) throw std::runtime_error("--benchmark-audio requires --wav");
    auto evaluate=[&] { if(benchmark_audio) (void)engine.infer(audio.samples,audio.sample_rate,threshold); else p=engine.probabilities(mel); };
    std::vector<double> times;
    if(benchmark_requested) {
        for (int i=0;i<warmup;++i) evaluate();
        for (int i=0;i<runs;++i) { auto a=Clock::now(); evaluate(); times.push_back(ms(a,Clock::now())); }
    } else times.push_back(ms(first_start,first_end));
    if (!prob.empty()) write(prob,p);
    p.resize(output_frames*360); auto result=rmvpe::decode(p,threshold);
    if (!output.empty()) {
        std::ofstream f(output); f << "time_s,f0_hz,confidence\n" << std::setprecision(9);
        for(size_t i=0;i<result.f0.size();++i) f << i*0.01 << ',' << result.f0[i] << ',' << result.confidence[i] << '\n';
        if(!f) throw std::runtime_error("Cannot write output CSV");
    }
    auto sorted=times; std::sort(sorted.begin(),sorted.end()); size_t n=sorted.size(); double median=(sorted[(n-1)/2]+sorted[n/2])/2;
    auto info=engine.execution_info();
    std::cout << std::setprecision(9) << "{\"backend\":\"" << engine.backend_name() << "\",\"frames\":" << mel.size()/128
        << ",\"output_frames\":" << output_frames << ",\"duration_s\":" << duration
        << ",\"load_ms\":" << ms(start,loaded) << ",\"frontend_ms\":" << ms(loaded,prepared)
        << ",\"first_inference_ms\":" << ms(first_start,first_end) << ",\"median_ms\":" << median << ",\"rtf\":" << median/(duration*1000)
        << ",\"graph_nodes\":" << info.graph_nodes << ",\"graph_splits\":" << info.graph_splits
        << ",\"cpu_compute_nodes\":" << info.cpu_compute_nodes << ",\"accelerator_compute_nodes\":" << info.accelerator_compute_nodes
        << ",\"compute_buffer_bytes\":" << info.compute_buffer_bytes << ",\"samples_ms\":[";
    for(size_t i=0;i<times.size();++i) std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n"; return 0;
} catch(const std::exception & e) { std::cerr << "rmvpe: " << e.what() << '\n'; return 1; }
