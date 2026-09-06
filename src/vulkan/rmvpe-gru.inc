// SPDX-License-Identifier: MPL-2.0
// Uses ggml's buffer dependencies, descriptors and command submission.
static void rmvpe_vk_gru(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * out) {
    GGML_ASSERT(rmvpe_gru_valid(out));
    auto * x=out->src[0]; auto * w=out->src[1]; auto * b=out->src[2];
    const std::array<uint32_t,7> pc={uint32_t(out->ne[1]), uint32_t(out->op_params[RMVPE_GRU_TAG_SLOT+1]),
        uint32_t(get_misalign_bytes(ctx,x)/4), uint32_t(get_misalign_bytes(ctx,w)/4),
        uint32_t(get_misalign_bytes(ctx,b)/4), uint32_t(get_misalign_bytes(ctx,out)/4),uint32_t(w->ne[0] == 768)};
    auto & pipeline=ctx->device->pipeline_rmvpe_gru;
    ggml_pipeline_request_descriptor_sets(ctx,pipeline,1);
    ggml_vk_dispatch_pipeline(ctx,subctx,pipeline,
        {ggml_vk_tensor_subbuffer(ctx,x,true),ggml_vk_tensor_subbuffer(ctx,w,true),
         ggml_vk_tensor_subbuffer(ctx,b,true),ggml_vk_tensor_subbuffer(ctx,out,true)},pc,{1,1,1});
}
