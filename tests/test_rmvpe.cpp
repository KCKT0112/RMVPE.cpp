// SPDX-License-Identifier: MPL-2.0
#include "rmvpe/rmvpe.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
static void check(bool ok) { if(!ok) throw std::runtime_error("Test failed"); }
template<class F> void rejects(F fn) { bool failed=false; try { fn(); } catch(const std::exception &) { failed=true; } check(failed); }
int main() try {
    std::vector<float> p(3*360,0); p[0]=1; p[360+359]=1; p[720+100]=0.03f;
    auto r=rmvpe::decode(p);
    check(std::abs(r.f0[0]-31.7f)<0.001f);
    check(std::abs(r.f0[1]-10*std::exp2((1997.3794084376191+359*20)/1200))<0.001);
    check(r.f0[2]==0 && r.confidence[2]==0.03f);
    rejects([] { rmvpe::decode({}); });
    rejects([&] { rmvpe::decode(p,-1); });
    p[0]=std::numeric_limits<float>::infinity(); rejects([&] { rmvpe::decode(p); });
    std::vector<float> a(4410,1); auto b=rmvpe::resample(a,44100,16000);
    check(b.size()==1600 && std::abs(b[800]-1)<0.001);
    check(rmvpe::resample(a,44100,44100)==a);
    rejects([&] { rmvpe::resample(a,0,16000); });
    rejects([] { rmvpe::read_wav("does-not-exist.wav"); });
    std::cout << "decoder, resampler and invalid-input tests passed\n"; return 0;
} catch(const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
