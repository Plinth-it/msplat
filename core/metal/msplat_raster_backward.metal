#include "msplat_common.metal"

kernel void rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins, // int2
    constant float* packed_xy_opac, // float3: (x, y, sigmoid(opacity))
    constant float* packed_conic,   // float3
    constant float* packed_rgb,     // float3: raw SH
    constant float* packed_opacity_comp,
    constant float* background, // single float3
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output, // float3
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb, // float3
    device atomic_float* v_opacity,
    device atomic_float* v_refine,
    constant uint* gt_packed,
    constant uint& use_alpha_loss,
    constant float& alpha_loss_grad_scale,
    constant half* packed_conic_half,
    constant half* packed_rgb_half,
    constant half* packed_opacity_comp_half,
    constant uint& use_half_sorted_buffers,
    uint3 gp [[thread_position_in_grid]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]],
    uint warp_size [[threads_per_simdgroup]],
    uint wr [[thread_index_in_simdgroup]]
) {
    uint i = gp.y;
    uint j = gp.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);

    const float px = (float)j;
    const float py = (float)i;
    // clamp this value to the last pixel
    const int32_t pix_id = min((int32_t)(i * img_size.x + j), (int32_t)(img_size.x * img_size.y - 1));

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = read_packed_int2(tile_bins, tile_id);
    const int num_batches = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup int32_t id_batch[RAST_BLOCK_SIZE];
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];
    threadgroup float opacity_comp_batch[RAST_BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = read_packed_float3(v_output, pix_id);
    const float target_alpha = inside ? packed_gt_alpha(gt_packed, (uint)pix_id) : 0.0f;
    const float alpha_loss_grad = (use_alpha_loss != 0 && inside)
        ? alpha_loss_grad_scale * (((1.0f - T_final) > target_alpha) ? 1.0f
            : (((1.0f - T_final) < target_alpha) ? -1.0f : 0.0f))
        : 0.0f;
    // Hoist loop-invariant background load and T_final * bg product
    const float3 bg = {background[0], background[1], background[2]};
    const float3 T_final_bg = T_final * bg;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int warp_bin_final = warp_reduce_all_max(bin_final, warp_size);

    // Subtile-level early exit: compute max bin_final across all warps,
    // skip leading batches where all gaussians are beyond any pixel's bin_final.
    const uint warp_id = tr / warp_size;
    constexpr uint NUM_WARPS = RAST_BLOCK_SIZE / 32;
    threadgroup int warp_max_finals[NUM_WARPS];
    if (wr == 0) warp_max_finals[warp_id] = warp_bin_final;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tile_max_bin_final = warp_max_finals[0];
    for (uint w = 1; w < NUM_WARPS; w++)
        tile_max_bin_final = max(tile_max_bin_final, warp_max_finals[w]);
    int dead_count = max(0, (int)(range.y - 1) - tile_max_bin_final);
    int first_batch = dead_count / RAST_BLOCK_SIZE;

    for (int b = first_batch; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - RAST_BLOCK_SIZE * b;
        int batch_size = min(RAST_BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            id_batch[tr] = gaussian_ids_sorted[idx];
            // Sequential reads from packed sorted-order buffers
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_sorted_float3(
                packed_conic, packed_conic_half, idx, use_half_sorted_buffers);
            rgbs_batch[tr] = read_packed_sorted_float3(
                packed_rgb, packed_rgb_half, idx, use_half_sorted_buffers);
            opacity_comp_batch[tr] = read_packed_sorted_float(
                packed_opacity_comp, packed_opacity_comp_half, idx, use_half_sorted_buffers);
        }
        // wait for other threads to collect the gaussians in batch
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0,batch_end - warp_bin_final); t < batch_size; ++t) {
            // Broadcast batch data from lane 0 → all lanes in SIMD group.
            // All threads read the same index t, so one threadgroup read +
            // simd_broadcast replaces 32 redundant threadgroup reads.
            float3 b_conic = float3(0.0f);
            float3 b_xy_opac = float3(0.0f);
            float3 b_rgb = float3(0.0f);
            float b_opacity_comp = 1.0f;
            int32_t b_id = 0;
            if (wr == 0) {
                b_conic = conic_batch[t];
                b_xy_opac = xy_opacity_batch[t];
                b_rgb = rgbs_batch[t];
                b_opacity_comp = opacity_comp_batch[t];
                b_id = id_batch[t];
            }
            b_conic = simd_broadcast(b_conic, 0);
            b_xy_opac = simd_broadcast(b_xy_opac, 0);
            b_rgb = simd_broadcast(b_rgb, 0);
            b_opacity_comp = simd_broadcast(b_opacity_comp, 0);
            b_id = simd_broadcast(b_id, 0);

            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float2 delta;
            float vis;
            if(valid){
                opac = b_xy_opac.z;
                delta = {b_xy_opac.x - px, b_xy_opac.y - py};
                float sigma = fma(0.5f,
                    fma(b_conic.x, delta.x * delta.x, b_conic.z * delta.y * delta.y),
                    b_conic.y * delta.x * delta.y);
                if (sigma < 0.f || sigma >= 5.55f) {
                    valid = 0;
                } else {
                    vis = exp(-sigma);
                    alpha = min(0.999f, opac * b_opacity_comp * vis);
                    if (alpha < 1.f / 255.f) {
                        valid = 0;
                    }
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp_reduce_all_or(valid, warp_size)) {
                continue;
            }

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_refine_local = 0.f;
            //initialize everything to 0, only set if the lane is valid
            if(valid && alpha<0.999f){
                // compute the current T for this gaussian
                // alpha = opac * vis (guaranteed since alpha < 0.99 = min cap)
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = fac * v_out;

                // b_rgb has raw SH output; clamp inline: max(raw + 0.5, 0)
                const float3 rgb = max(b_rgb + 0.5f, 0.f);
                // contribution from this pixel + background
                v_alpha += dot(fma(rgb, T, fma(-buffer, ra, -ra * T_final_bg)), v_out);
                v_alpha += alpha_loss_grad * T_final * ra;
                // update the running sum
                buffer = fma(rgb, fac, buffer);

                // v_sigma = d(loss)/d(sigma) = -alpha * v_alpha
                const float v_sigma = -alpha * v_alpha;
                v_conic_local = (0.5f * v_sigma) * float3(delta.x * delta.x,
                                                           delta.x * delta.y,
                                                           delta.y * delta.y);
                v_xy_local = v_sigma * float2(
                    fma(b_conic.x, delta.x, b_conic.y * delta.y),
                    fma(b_conic.y, delta.x, b_conic.z * delta.y));
                // Fused sigmoid derivative: dL/d(logit) = -v_sigma * (1 - opac)
                v_opacity_local = -v_sigma * (1.f - opac);
                float final_alpha = max(1.0f - T_final, 1e-5f);
                v_refine_local = length(v_xy_local * float2((float)img_size.x, (float)img_size.y)) / final_alpha;
            }

            v_rgb_local = warpSum3(v_rgb_local, warp_size, wr);
            v_conic_local = warpSum3(v_conic_local, warp_size, wr);
            v_xy_local = warpSum2(v_xy_local, warp_size, wr);
            v_opacity_local = warpSum(v_opacity_local, warp_size, wr);
            v_refine_local = warpSum(v_refine_local, warp_size, wr);

            if (wr == 0) {
                // Fused clamp_min backward: zero gradient where raw_color + 0.5 < 0
                if (b_rgb.x + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 0, v_rgb_local.x, memory_order_relaxed);
                if (b_rgb.y + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 1, v_rgb_local.y, memory_order_relaxed);
                if (b_rgb.z + 0.5f >= 0.f) atomic_fetch_add_explicit(v_rgb + 3*b_id + 2, v_rgb_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_conic + 3*b_id + 0, v_conic_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 1, v_conic_local.y, memory_order_relaxed);
                atomic_fetch_add_explicit(v_conic + 3*b_id + 2, v_conic_local.z, memory_order_relaxed);

                atomic_fetch_add_explicit(v_xy + 2*b_id + 0, v_xy_local.x, memory_order_relaxed);
                atomic_fetch_add_explicit(v_xy + 2*b_id + 1, v_xy_local.y, memory_order_relaxed);

                atomic_fetch_add_explicit(v_opacity + b_id, v_opacity_local, memory_order_relaxed);
                atomic_fetch_add_explicit(v_refine + b_id, v_refine_local, memory_order_relaxed);
            }
        }
    }
}

// Brush-style per-splat backward rasterization. One workgroup owns one 16x16
// tile and one SIMD group owns up to 32 splats at a time. Pixel state is walked
// in forward order with diagonal scheduling so each thread accumulates one
// splat's full tile gradient in registers before issuing atomics.
kernel void rasterize_backward_persplat_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins,
    constant float* packed_xy_opac,
    constant float* packed_conic,
    constant float* packed_rgb,
    constant float* packed_opacity_comp,
    constant float* background,
    constant float* out_img,
    constant float* final_Ts,
    constant float* v_output,
    device atomic_float* v_xy,
    device atomic_float* v_conic,
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    device atomic_float* v_refine,
    constant uint* gt_packed,
    constant uint& use_alpha_loss,
    constant float& alpha_loss_grad_scale,
    constant half* packed_conic_half,
    constant half* packed_rgb_half,
    constant half* packed_opacity_comp_half,
    constant uint& use_half_sorted_buffers,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint thread_rank [[thread_index_in_threadgroup]]
) {
    constexpr uint SPLAT_BATCH = 32;
    constexpr uint TILE_PIXELS = BLOCK_SIZE;

    uint tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    if (blockIdx.x >= tile_bounds.x || blockIdx.y >= tile_bounds.y) {
        return;
    }

    uint2 tile_origin = uint2(blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y);
    float3 bg = float3(background[0], background[1], background[2]);
    float pix_v_out_tail_scale = max(1.0f, float(img_size.x) * float(img_size.y));
    float inv_pix_v_out_tail_scale = 1.0f / pix_v_out_tail_scale;

    threadgroup half4 pix_state[TILE_PIXELS];
    threadgroup half4 pix_v_out_tail[TILE_PIXELS];
    threadgroup half pix_inv_final_alpha[TILE_PIXELS];
    threadgroup int range_start = 0;
    threadgroup int range_end = 0;

    if (thread_rank == 0) {
        int2 range = read_packed_int2(tile_bins, (int)tile_id);
        range_start = range.x;
        range_end = range.y;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint num_splats_in_tile = (uint)max(0, range_end - range_start);
    if (num_splats_in_tile == 0) {
        return;
    }
    uint rounds = (num_splats_in_tile + SPLAT_BATCH - 1) / SPLAT_BATCH;

    for (uint pix_rank = thread_rank; pix_rank < TILE_PIXELS; pix_rank += SPLAT_BATCH) {
        uint2 local_xy = uint2(pix_rank % BLOCK_X, pix_rank / BLOCK_X);
        uint2 pix_loc = tile_origin + local_xy;
        bool inside = pix_loc.x < img_size.x && pix_loc.y < img_size.y;
        if (inside) {
            uint pix_id = pix_loc.y * img_size.x + pix_loc.x;
            float T_final = final_Ts[pix_id];
            float3 final_rgb = read_packed_float3(out_img, (int)pix_id);
            float3 v_out = read_packed_float3(v_output, (int)pix_id);
            float target_alpha = packed_gt_alpha(gt_packed, pix_id);
            float alpha_loss_grad = (use_alpha_loss != 0)
                ? alpha_loss_grad_scale * (((1.0f - T_final) > target_alpha) ? 1.0f
                    : (((1.0f - T_final) < target_alpha) ? -1.0f : 0.0f))
                : 0.0f;

            pix_state[pix_rank] = half4(float4(final_rgb - T_final * bg, 1.0f));
            pix_v_out_tail[pix_rank] = half4(float4(v_out, T_final * (alpha_loss_grad - dot(bg, v_out)))
                * pix_v_out_tail_scale);
            pix_inv_final_alpha[pix_rank] = half(1.0f / max(1.0f - T_final, 1e-5f));
        } else {
            pix_state[pix_rank] = half4(0.0h);
            pix_v_out_tail[pix_rank] = half4(0.0h);
            pix_inv_final_alpha[pix_rank] = 0.0h;
        }
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint batch_idx = 0; batch_idx < rounds; ++batch_idx) {
        uint splat_offset = batch_idx * SPLAT_BATCH + thread_rank;
        bool splat_active = splat_offset < num_splats_in_tile;
        int sorted_idx = range_start + (int)splat_offset;
        int32_t gaussian_id = 0;
        float3 xy_opac = float3(0.0f);
        float3 conic = float3(0.0f);
        float3 raw_rgb = float3(0.0f);
        float opacity_comp = 1.0f;

        if (splat_active) {
            gaussian_id = gaussian_ids_sorted[sorted_idx];
            xy_opac = read_packed_float3(packed_xy_opac, sorted_idx);
            conic = read_packed_sorted_float3(
                packed_conic, packed_conic_half, sorted_idx, use_half_sorted_buffers);
            raw_rgb = read_packed_sorted_float3(
                packed_rgb, packed_rgb_half, sorted_idx, use_half_sorted_buffers);
            opacity_comp = read_packed_sorted_float(
                packed_opacity_comp, packed_opacity_comp_half, sorted_idx, use_half_sorted_buffers);
        }

        uint num_splats_this_batch = min(SPLAT_BATCH, num_splats_in_tile - batch_idx * SPLAT_BATCH);
        uint total_iters = num_splats_this_batch + TILE_PIXELS - 1;
        float2 v_xy_acc = float2(0.0f);
        float3 v_conic_acc = float3(0.0f);
        float3 v_rgb_acc = float3(0.0f);
        float v_opacity_acc = 0.0f;
        float v_refine_acc = 0.0f;
        float3 rgb = max(raw_rgb + 0.5f, 0.0f);

        for (uint iter = 0; iter < total_iters; ++iter) {
            bool active_iter = splat_active
                && iter >= thread_rank
                && (iter - thread_rank) < TILE_PIXELS;
            if (active_iter) {
                uint pix_rank = iter - thread_rank;
                float4 state = float4(pix_state[pix_rank]);
                if (state.w > 1e-4f) {
                    uint2 local_xy = uint2(pix_rank % BLOCK_X, pix_rank / BLOCK_X);
                    uint2 pix_loc = tile_origin + local_xy;
                    float2 delta = float2(xy_opac.x - (float)pix_loc.x,
                                          xy_opac.y - (float)pix_loc.y);
                    float sigma = fma(0.5f,
                        fma(conic.x, delta.x * delta.x, conic.z * delta.y * delta.y),
                        conic.y * delta.x * delta.y);
                    if (sigma >= 0.0f && sigma < 5.55f) {
                        float alpha = min(0.999f, xy_opac.z * opacity_comp * exp(-sigma));
                        if (alpha >= 1.0f / 255.0f) {
                            float next_T = state.w * (1.0f - alpha);
                            if (next_T <= 1e-4f) {
                                pix_state[pix_rank] = half4(float4(state.xyz, 0.0f));
                            } else {
                                float vis = alpha * state.w;
                                float3 new_remain = state.xyz - vis * rgb;
                                float4 v_out_tail = float4(pix_v_out_tail[pix_rank]) * inv_pix_v_out_tail_scale;
                                float3 v_out = v_out_tail.xyz;

                                v_rgb_acc += vis * v_out;

                                float ra = 1.0f / (1.0f - alpha);
                                float v_alpha = dot(state.w * rgb - new_remain * ra, v_out)
                                    + v_out_tail.w * ra;
                                float v_sigma = -alpha * v_alpha;

                                float2 v_xy_local = v_sigma * float2(
                                    fma(conic.x, delta.x, conic.y * delta.y),
                                    fma(conic.y, delta.x, conic.z * delta.y));
                                v_xy_acc += v_xy_local;
                                v_conic_acc += (0.5f * v_sigma) * float3(
                                    delta.x * delta.x,
                                    delta.x * delta.y,
                                    delta.y * delta.y);
                                v_opacity_acc += -v_sigma * (1.0f - xy_opac.z);
                                v_refine_acc += length(v_xy_local * float2((float)img_size.x, (float)img_size.y))
                                    * float(pix_inv_final_alpha[pix_rank]);

                                pix_state[pix_rank] = half4(float4(new_remain, next_T));
                            }
                        }
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (splat_active) {
            if (raw_rgb.x + 0.5f >= 0.0f) {
                atomic_fetch_add_explicit(v_rgb + 3 * gaussian_id + 0, v_rgb_acc.x, memory_order_relaxed);
            }
            if (raw_rgb.y + 0.5f >= 0.0f) {
                atomic_fetch_add_explicit(v_rgb + 3 * gaussian_id + 1, v_rgb_acc.y, memory_order_relaxed);
            }
            if (raw_rgb.z + 0.5f >= 0.0f) {
                atomic_fetch_add_explicit(v_rgb + 3 * gaussian_id + 2, v_rgb_acc.z, memory_order_relaxed);
            }
            atomic_fetch_add_explicit(v_conic + 3 * gaussian_id + 0, v_conic_acc.x, memory_order_relaxed);
            atomic_fetch_add_explicit(v_conic + 3 * gaussian_id + 1, v_conic_acc.y, memory_order_relaxed);
            atomic_fetch_add_explicit(v_conic + 3 * gaussian_id + 2, v_conic_acc.z, memory_order_relaxed);
            atomic_fetch_add_explicit(v_xy + 2 * gaussian_id + 0, v_xy_acc.x, memory_order_relaxed);
            atomic_fetch_add_explicit(v_xy + 2 * gaussian_id + 1, v_xy_acc.y, memory_order_relaxed);
            atomic_fetch_add_explicit(v_opacity + gaussian_id, v_opacity_acc, memory_order_relaxed);
            atomic_fetch_add_explicit(v_refine + gaussian_id, v_refine_acc, memory_order_relaxed);
        }
    }
}

kernel void nd_rasterize_backward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int32_t* gaussians_ids_sorted,
    constant int* tile_bins, // int2
    constant float* xys, // float2
    constant float* conics, // float3
    constant float* rgbs,
    constant float* opacities,
    constant float* background,
    constant float* final_Ts,
    constant int* final_index,
    constant float* v_output,
    device atomic_float* v_xy, // float2
    device atomic_float* v_conic, // float3
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    device float* workspace,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 blockDim [[threads_per_threadgroup]],
    uint3 threadIdx [[thread_position_in_threadgroup]]
) {
    if (channels > MAX_REGISTER_CHANNELS && workspace == nullptr) {
        return;
    }
    // Per-pixel backward pass (no shared-memory batching)
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians get gradients for this pixel
    int2 range = read_packed_int2(tile_bins, tile_id);
    // df/d_out for this pixel
    constant float *v_out = &(v_output[channels * pix_id]);
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    device float *S = &workspace[channels * pix_id];
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = read_packed_float3(conics, g);
        const float2 center = read_packed_float2(xys, g);
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        // Early-out: skip exp() when the result is guaranteed to be discarded.
        // alpha = min(0.99, opacity * exp(-sigma)), discarded when alpha < 1/255.
        // opacity = sigmoid(raw) ∈ [0,1], so alpha ≤ exp(-sigma).
        // exp(-5.55) ≈ 0.00389 < 1/255 ≈ 0.00392, so sigma ≥ 5.55 ⟹ alpha < 1/255.
        if (sigma < 0.f || sigma >= 5.55f) {
            continue;
        }
        const float opac = opacities[g];
        const float vis = exp(-sigma);
        const float alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        const float ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        const float fac = alpha * T;
        float v_alpha = 0.f;
        for (uint c = 0; c < channels; ++c) {
            // gradient wrt rgb
            atomic_fetch_add_explicit(v_rgb + channels * g + c, fac * v_out[c], memory_order_relaxed);
            // contribution from this pixel
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[channels * g + c] * fac;
        }
        // update v_opacity for this gaussian
        atomic_fetch_add_explicit(v_opacity + g, vis * v_alpha, memory_order_relaxed);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        const float v_sigma = -opac * vis * v_alpha;

        atomic_fetch_add_explicit(v_conic + 3*g + 0, 0.5f * v_sigma * delta.x * delta.x, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 1, 0.5f * v_sigma * delta.x * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(v_conic + 3*g + 2, 0.5f * v_sigma * delta.y * delta.y, memory_order_relaxed);
        atomic_fetch_add_explicit(
            v_xy + 2*g + 0, v_sigma * (conic.x * delta.x + conic.y * delta.y), memory_order_relaxed
        );
        atomic_fetch_add_explicit(
            v_xy + 2*g + 1, v_sigma * (conic.y * delta.x + conic.z * delta.y), memory_order_relaxed
        );
    }
}

