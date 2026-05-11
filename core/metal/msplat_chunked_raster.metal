#include "msplat_common.metal"

// ============================================================================
// Depth-chunked rasterization kernels
// ============================================================================

#define CHUNK_SIZE 512

// Reduce max tile count: find max(bin_end - bin_start) across all tiles.
kernel void reduce_max_tile_count_kernel(
    constant uint& num_tiles,
    constant int* tile_bins, // int2 packed
    device atomic_uint* max_count,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_tiles) return;
    int2 range = read_packed_int2(tile_bins, (int)idx);
    uint count = (uint)max(0, range.y - range.x);
    atomic_fetch_max_explicit(max_count, count, memory_order_relaxed);
}

// Forward chunked rasterization: each threadgroup processes one (tile, chunk) pair.
// Grid: (tile_x, tile_y, K_max). blockIdx.z = chunk index k.
kernel void rasterize_forward_chunked_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    constant int* tile_bins,
    constant float* packed_xy_opac,
    constant float* packed_conic,
    constant float* packed_rgb,
    constant float* packed_opacity_comp,
    device float* chunk_T,        // [K_max, H, W]
    device float* chunk_C,        // [K_max, H, W, 3]
    device int* chunk_final_idx,  // [K_max, H, W]
    constant uint& chunk_size,
    constant uint& K_max,
    constant uint2& blockDim,
    constant half* packed_conic_half,
    constant half* packed_rgb_half,
    constant half* packed_opacity_comp_half,
    constant uint& use_half_sorted_buffers,
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]]
) {
    uint k = blockIdx.z; // chunk index
    // Reconstruct 2D thread position from 1D thread index
    uint threadIdx_x = tr % RAST_BLOCK_X;
    uint threadIdx_y = tr / RAST_BLOCK_X;
    int32_t i = blockIdx.y * blockDim.y + threadIdx_y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx_x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    uint num_pixels = img_size.x * img_size.y;
    int32_t pix_id = i * (int)img_size.x + j;
    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    // Full tile range from tile_bins
    int2 full_range = read_packed_int2(tile_bins, tile_id);
    // Chunk sub-range
    int chunk_start = full_range.x + (int)(k * chunk_size);
    int chunk_end = min(full_range.x + (int)((k + 1) * chunk_size), full_range.y);

    // Output offset: k * num_pixels + pix_id
    uint out_offset = k * num_pixels + (uint)pix_id;

    if (chunk_start >= chunk_end) {
        // Empty chunk — write defaults
        if (inside) {
            chunk_T[out_offset] = 1.f;
            chunk_C[out_offset * 3 + 0] = 0.f;
            chunk_C[out_offset * 3 + 1] = 0.f;
            chunk_C[out_offset * 3 + 2] = 0.f;
            chunk_final_idx[out_offset] = -1;
        }
        return;
    }

    int num_batches = (chunk_end - chunk_start + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];
    threadgroup float opacity_comp_batch[RAST_BLOCK_SIZE];

    float T = 1.f;
    float3 pix_out = {0.f, 0.f, 0.f};
    int last_contributor = chunk_start - 1;
    bool done = false;

    for (int b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int batch_start = chunk_start + RAST_BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < chunk_end) {
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_sorted_float3(
                packed_conic, packed_conic_half, idx, use_half_sorted_buffers);
            const float3 raw_c = read_packed_sorted_float3(
                packed_rgb, packed_rgb_half, idx, use_half_sorted_buffers);
            rgbs_batch[tr] = max(raw_c + 0.5f, 0.0f);
            opacity_comp_batch[tr] = read_packed_sorted_float(
                packed_opacity_comp, packed_opacity_comp_half, idx, use_half_sorted_buffers);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done || !inside) continue;

        int batch_size = min(RAST_BLOCK_SIZE, chunk_end - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            const float3 conic_local = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = fma(0.5f,
                fma(conic_local.x, delta.x * delta.x, conic_local.z * delta.y * delta.y),
                conic_local.y * delta.x * delta.y);
            if (sigma < 0.f || sigma >= 5.55f) continue;
            const float alpha = min(0.999f, xy_opac.z * opacity_comp_batch[t] * exp(-sigma));
            if (alpha < 1.f / 255.f) continue;
            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                last_contributor = batch_start + t - 1;
                done = true;
                break;
            }
            const float vis = alpha * T;
            pix_out = fma(rgbs_batch[t], vis, pix_out);
            T = next_T;
            last_contributor = batch_start + t;
        }
    }

    if (inside) {
        chunk_T[out_offset] = T;
        chunk_C[out_offset * 3 + 0] = pix_out.x;
        chunk_C[out_offset * 3 + 1] = pix_out.y;
        chunk_C[out_offset * 3 + 2] = pix_out.z;
        chunk_final_idx[out_offset] = last_contributor;
    }
}

// Forward merge: scan K chunks per pixel, produce final out_img/final_Ts/final_idx.
// Also applies absolute transmittance cutoff: when T_running drops below 1e-4,
// zeros out chunk_final_idx for remaining chunks so the backward skips them.
kernel void rasterize_forward_merge_kernel(
    constant uint& num_pixels, // H * W
    constant uint& K_max,
    constant float* chunk_T,        // [K_max, H, W]
    constant float* chunk_C,        // [K_max, H, W, 3]
    device int* chunk_final_idx,    // [K_max, H, W] — device (not constant) for cutoff fixup
    device float* final_Ts,         // [H, W]
    device int* final_index,        // [H, W]
    device float* out_img,          // [H, W, 3]
    constant float* background,
    constant uint2& img_size,       // (W, H)
    uint2 gp [[thread_position_in_grid]]
) {
    uint px = gp.x;
    uint py = gp.y;
    if (px >= img_size.x || py >= img_size.y) return;
    uint pix_id = py * img_size.x + px;

    float T_running = 1.f;
    float3 C_running = {0.f, 0.f, 0.f};
    int last_idx = -1;
    uint cutoff_k = K_max; // chunk index where absolute T cutoff triggered

    for (uint k = 0; k < K_max; ++k) {
        uint offset = k * num_pixels + pix_id;
        int cfidx = chunk_final_idx[offset];
        if (cfidx < 0 && k > 0) break; // empty chunk after first real one = done
        // Even if cfidx == chunk_start-1 (no contribution), chunk_T=1 and chunk_C=0
        float cT = chunk_T[offset];
        float3 cC = {chunk_C[offset * 3 + 0], chunk_C[offset * 3 + 1], chunk_C[offset * 3 + 2]};
        C_running = fma(cC, T_running, C_running);
        T_running *= cT;
        if (cfidx >= 0) last_idx = cfidx;
        // Absolute transmittance cutoff: stop when pixel is fully opaque
        if (T_running <= 1e-4f) {
            cutoff_k = k + 1;
            break;
        }
    }

    // Zero out chunk_final_idx for chunks past the absolute cutoff
    // so the backward kernel skips them (bin_final < chunk_start → return)
    for (uint k = cutoff_k; k < K_max; ++k) {
        chunk_final_idx[k * num_pixels + pix_id] = -1;
    }

    final_Ts[pix_id] = T_running;
    final_index[pix_id] = last_idx;
    float3 bg = {background[0], background[1], background[2]};
    float3 final_rgb = saturate(fma(bg, T_running, C_running));
    out_img[CHANNELS * pix_id + 0] = final_rgb.x;
    out_img[CHANNELS * pix_id + 1] = final_rgb.y;
    out_img[CHANNELS * pix_id + 2] = final_rgb.z;
}

// Compute prefix transmittance and suffix color for backward chunked rasterization.
// prefix_T[k] = product of chunk_T[0..k-1] (transmittance before chunk k)
// after_C[k] = sum_{j>k} prefix_T[j] * chunk_C[j] (color contribution after chunk k)
kernel void compute_chunk_prefix_suffix_kernel(
    constant uint& num_pixels,
    constant uint& K_max,
    constant float* chunk_T,        // [K_max, H, W]
    constant float* chunk_C,        // [K_max, H, W, 3]
    constant int* chunk_final_idx,  // [K_max, H, W]
    device float* prefix_T,         // [K_max, H, W]
    device float* after_C,          // [K_max, H, W, 3]
    constant uint2& img_size,
    uint2 gp [[thread_position_in_grid]]
) {
    uint px = gp.x;
    uint py = gp.y;
    if (px >= img_size.x || py >= img_size.y) return;
    uint pix_id = py * img_size.x + px;

    // Forward scan: compute prefix transmittance products
    float pT = 1.f;
    for (uint k = 0; k < K_max; ++k) {
        uint offset = k * num_pixels + pix_id;
        prefix_T[offset] = pT;
        if (chunk_final_idx[offset] >= 0) {
            pT *= chunk_T[offset];
        }
    }

    // Backward scan: compute suffix color contribution
    // after_C[k] = sum_{j=k+1}^{K-1} prefix_T[j] * chunk_C[j]
    float3 aC = {0.f, 0.f, 0.f};
    for (int k = (int)K_max - 1; k >= 0; --k) {
        uint offset = (uint)k * num_pixels + pix_id;
        after_C[offset * 3 + 0] = aC.x;
        after_C[offset * 3 + 1] = aC.y;
        after_C[offset * 3 + 2] = aC.z;
        if (chunk_final_idx[offset] < 0) {
            continue;
        }
        float pT_k = prefix_T[offset];
        float3 cC = {chunk_C[offset * 3 + 0], chunk_C[offset * 3 + 1], chunk_C[offset * 3 + 2]};
        aC += pT_k * cC;
    }
}

// Backward chunked rasterization: each threadgroup processes one (tile, chunk) pair.
// Grid: (tile_x, tile_y, K_max). blockIdx.z = chunk index k.
kernel void rasterize_backward_chunked_kernel(
    constant uint3& tile_bounds,
    constant uint2& img_size,
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins,
    constant float* packed_xy_opac,
    constant float* packed_conic,
    constant float* packed_rgb,
    constant float* packed_opacity_comp,
    constant float* background,
    constant float* final_Ts,       // [H, W] — global final transmittance
    constant int* chunk_final_idx,  // [K_max, H, W]
    constant float* prefix_T_buf,   // [K_max, H, W]
    constant float* chunk_T_buf,    // [K_max, H, W]
    constant float* after_C_buf,    // [K_max, H, W, 3]
    constant float* v_output,
    device atomic_float* v_xy,
    device atomic_float* v_conic,
    device atomic_float* v_rgb,
    device atomic_float* v_opacity,
    device atomic_float* v_refine,
    constant uint& chunk_size,
    constant uint& K_max,
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
    uint k = blockIdx.z;
    uint i = gp.y;
    uint j = gp.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    const float px = (float)j;
    const float py = (float)i;
    const int32_t pix_id = min((int32_t)(i * img_size.x + j), (int32_t)(img_size.x * img_size.y - 1));
    const bool inside = (i < img_size.y && j < img_size.x);
    uint num_pixels = img_size.x * img_size.y;
    uint chunk_offset = k * num_pixels + (uint)pix_id;

    // Read per-pixel, per-chunk data
    float pT_k = prefix_T_buf[chunk_offset];       // transmittance before chunk k
    float cT_k = chunk_T_buf[chunk_offset];         // local transmittance of chunk k
    float3 aC_k = {after_C_buf[chunk_offset * 3 + 0],
                    after_C_buf[chunk_offset * 3 + 1],
                    after_C_buf[chunk_offset * 3 + 2]};

    // Initialize T and buffer as if monolithic backward just finished all chunks > k
    // T = prefix_T[k] * chunk_T[k] = absolute transmittance after chunk k
    float T = pT_k * cT_k;
    // buffer = after_C[k] = weighted color contribution from all chunks after k
    float3 buffer = aC_k;

    float T_final = final_Ts[pix_id];
    const float3 bg = {background[0], background[1], background[2]};
    const float3 T_final_bg = T_final * bg;
    const float3 v_out = read_packed_float3(v_output, pix_id);
    const float target_alpha = inside ? packed_gt_alpha(gt_packed, (uint)pix_id) : 0.0f;
    const uint use_alpha_loss_value = raster_use_alpha_loss(use_alpha_loss);
    const uint use_half_sorted_buffers_value = raster_use_half_sorted_buffers(use_half_sorted_buffers);
    const float alpha_loss_grad = (use_alpha_loss_value != 0 && inside)
        ? alpha_loss_grad_scale * (((1.0f - T_final) > target_alpha) ? 1.0f
            : (((1.0f - T_final) < target_alpha) ? -1.0f : 0.0f))
        : 0.0f;

    const int bin_final = inside ? chunk_final_idx[chunk_offset] : -1;

    // Chunk sub-range
    int2 full_range = read_packed_int2(tile_bins, tile_id);
    int chunk_start = full_range.x + (int)(k * chunk_size);
    int chunk_end = min(full_range.x + (int)((k + 1) * chunk_size), full_range.y);

    if (chunk_start >= chunk_end || bin_final < chunk_start) {
        return; // empty chunk or no contributors
    }

    const int num_batches = (chunk_end - chunk_start + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    threadgroup int32_t id_batch[RAST_BLOCK_SIZE];
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];
    threadgroup float opacity_comp_batch[RAST_BLOCK_SIZE];

    // Warp-level early exit
    const int warp_bin_final = warp_reduce_all_max(bin_final, warp_size);

    // Subtile-level early exit: skip leading batches beyond any pixel's bin_final
    const uint warp_id = tr / warp_size;
    constexpr uint NUM_WARPS = RAST_BLOCK_SIZE / 32;
    threadgroup int warp_max_finals[NUM_WARPS];
    if (wr == 0) warp_max_finals[warp_id] = warp_bin_final;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int tile_max_bin_final = warp_max_finals[0];
    for (uint w = 1; w < NUM_WARPS; w++)
        tile_max_bin_final = max(tile_max_bin_final, warp_max_finals[w]);
    int dead_count = max(0, (int)(chunk_end - 1) - tile_max_bin_final);
    int first_batch = dead_count / RAST_BLOCK_SIZE;

    for (int b = first_batch; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load batch from back to front within this chunk
        const int batch_end = chunk_end - 1 - RAST_BLOCK_SIZE * b;
        int batch_size = min(RAST_BLOCK_SIZE, batch_end + 1 - chunk_start);
        const int idx = batch_end - tr;
        if (idx >= chunk_start) {
            id_batch[tr] = gaussian_ids_sorted[idx];
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_sorted_float3(
                packed_conic, packed_conic_half, idx, use_half_sorted_buffers_value);
            rgbs_batch[tr] = read_packed_sorted_float3(
                packed_rgb, packed_rgb_half, idx, use_half_sorted_buffers_value);
            opacity_comp_batch[tr] = read_packed_sorted_float(
                packed_opacity_comp, packed_opacity_comp_half, idx, use_half_sorted_buffers_value);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
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
            if (batch_end - t > bin_final) valid = 0;

            float alpha;
            float opac;
            float2 delta;
            float vis;
            if (valid) {
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
                    if (alpha < 1.f / 255.f) valid = 0;
                }
            }

            if (!warp_reduce_all_or(valid, warp_size)) continue;

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            float v_refine_local = 0.f;

            if (valid && alpha < 0.999f) {
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = fac * v_out;

                const float3 rgb = max(b_rgb + 0.5f, 0.f);
                v_alpha += dot(fma(rgb, T, fma(-buffer, ra, -ra * T_final_bg)), v_out);
                v_alpha += alpha_loss_grad * T_final * ra;
                buffer = fma(rgb, fac, buffer);

                const float v_sigma = -alpha * v_alpha;
                v_conic_local = (0.5f * v_sigma) * float3(delta.x * delta.x,
                                                           delta.x * delta.y,
                                                           delta.y * delta.y);
                v_xy_local = v_sigma * float2(
                    fma(b_conic.x, delta.x, b_conic.y * delta.y),
                    fma(b_conic.y, delta.x, b_conic.z * delta.y));
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
