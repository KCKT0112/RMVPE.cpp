// SPDX-License-Identifier: MPL-2.0
#include <rmvpe/rmvpe.h>
#include <iostream>
int main(int argc, char ** argv) try {
    if(argc!=3) { std::cerr << "Usage: rmvpe-example model.gguf input.wav\n"; return 1; }
    rmvpe::Model model(argv[1]);
    auto audio=rmvpe::read_wav(argv[2]);
    auto result=model.infer(audio.samples,audio.sample_rate);
    for(size_t i=0;i<result.f0.size();++i)
        std::cout << i*.01 << ',' << result.f0[i] << ',' << result.confidence[i] << '\n';
    return 0;
} catch(const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
