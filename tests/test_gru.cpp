// SPDX-License-Identifier: MPL-2.0
#include "gru.h"
#include "gru-tag.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
static void check(bool ok, const char * msg) { if(!ok) throw std::runtime_error(msg); }
int main(int argc, char ** argv) try {
    ggml_backend_load_all(); const char * name=argc>1?argv[1]:"CPU";
    auto * backend=ggml_backend_init_by_name(name,nullptr); check(backend,"Missing backend");
    for(int frames : {1,7,32,65}) for(int reverse : {0,1}) for(int transposed : {0,1}) {
        auto * ctx=ggml_init({2*1024*1024,nullptr,true});
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,768,frames);
        auto * w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,transposed?768:256,transposed?256:768);
        auto * b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,768);
        auto * y=rmvpe::gru_fused(ctx,x,w,b,reverse!=0);
        check(rmvpe_gru_valid(y),"Custom contract rejected valid GRU");
        check(ggml_backend_supports_op(backend,y),"Backend cannot run fused GRU");
        std::mt19937 rng(37); std::normal_distribution<float> dist(0,.08f);
        std::vector<float> xv(frames*768),wv(768*256),bv(768),canonical(wv.size());
        for(auto & v:xv) v=dist(rng)*8; for(auto & v:canonical) v=dist(rng); for(auto & v:bv) v=dist(rng)*4;
        for(int j=0;j<768;++j) for(int k=0;k<256;++k) wv[transposed?k*768+j:j*256+k]=canonical[j*256+k];
        auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); check(buffer,"Allocation failed");
        ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4); ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4); ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
        auto * graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Execution failed");
        std::vector<float> actual(frames*256); ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
        std::vector<double> h(256),next(256); double maximum=0;
        for(int step=0;step<frames;++step) {
            int t=reverse?frames-1-step:step;
            for(int j=0;j<256;++j) {
                double gates[3]={bv[j],bv[256+j],bv[512+j]};
                for(int g=0;g<3;++g) for(int k=0;k<256;++k) gates[g]+=canonical[(g*256+j)*256+k]*h[k];
                double r=1/(1+std::exp(-xv[t*768+j]-gates[0]));
                double z=1/(1+std::exp(-xv[t*768+256+j]-gates[1]));
                double n=std::tanh(xv[t*768+512+j]+r*gates[2]);
                next[j]=(1-z)*n+z*h[j];
                check(std::isfinite(actual[t*256+j]),"Non-finite output");
                maximum=std::max(maximum,std::abs(actual[t*256+j]-next[j]));
            }
            h=next;
        }
        check(maximum<2e-5,"GRU differs from independent F64 recurrence");
        // Reuse the graph: recurrence must start from zero, not the last call's state.
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Second execution failed");
        std::vector<float> repeat(actual.size()); ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);
        check(repeat==actual,"GRU state leaked between calls");
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend); std::cout<<name<<": 32 GRU direction/layout/length cases passed\n"; return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
