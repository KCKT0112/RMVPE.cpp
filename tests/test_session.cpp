// SPDX-License-Identifier: MPL-2.0
#include <rmvpe/rmvpe.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
static void check(bool ok,const char * msg) { if(!ok) throw std::runtime_error(msg); }
int main(int argc,char ** argv) try {
    if(argc<2) throw std::runtime_error("Usage: rmvpe-session-tests model.gguf [CPU|Vulkan0]");
    rmvpe::Options opts; opts.backend=argc>2?argv[2]:"cpu"; opts.threads=4;
    rmvpe::Model model(argv[1],opts);
    std::vector<float> mel(32*128);
    for(size_t i=0;i<mel.size();++i) mel[i]=-4+2*std::sin(static_cast<float>(i)*.37f);
    auto first=model.probabilities(mel);
    auto longer=mel; longer.insert(longer.end(),mel.begin(),mel.end());
    check(model.probabilities(longer).size()==64*360,"Length change lost output frames");
    auto repeat=model.probabilities(mel);
    check(first==repeat,"Changing graph length leaked recurrent state");
    bool rejected=false; try { model.probabilities(std::vector<float>(128)); } catch(const std::exception &) { rejected=true; }
    check(rejected,"Accepted non-multiple-of-32 Mel frames");
    opts.unroll_gru=true; rmvpe::Model unrolled(argv[1],opts);
    auto reference=unrolled.probabilities(mel);
    float maximum=0;
    for(size_t i=0;i<first.size();++i) maximum=std::max(maximum,std::abs(first[i]-reference[i]));
    check(maximum<1e-5f,"Unrolled/fused GRU network mismatch");
    std::cout << opts.backend << ": graph resize, recurrence reset and portable graph parity passed; max " << maximum << '\n';
    return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
