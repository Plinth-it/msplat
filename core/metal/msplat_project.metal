#include "msplat_common.metal"

kernel void copy_int_buffer_kernel(
    constant uint &count [[buffer(0)]],
    constant int *src [[buffer(1)]],
    device int *dst [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

kernel void project_gaussians_forward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* covs3d,
    device float* xys, // float2
    device float* depths,
    device int* radii,
    device float* conics, // float3
    device int32_t* num_tiles_hit,
    device float* aabb, // float2: per-axis pixel extents
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = read_packed_float3(means3d, idx);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        return;
    }

    // compute the projected covariance
    // scales are in log-space; exp() here to avoid a separate MPS dispatch
    float3 scale = exp(read_packed_float3(scales, idx));
    if (!finite_float3(scale)) {
        return;
    }
    float4 quat = read_packed_float4(quats, idx);
    if (!valid_quaternion(quat)) {
        return;
    }
    device float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view
    );

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return; // zero determinant
    }
    write_packed_float3(conics, idx, conic);

    float aabb_x = ceil(3.0f * sqrt(cov2d.x));
    float aabb_y = ceil(3.0f * sqrt(cov2d.z));

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, float2(aabb_x, aabb_y), (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    write_packed_float2(xys, idx, center);
    aabb[idx * 2] = aabb_x;
    aabb[idx * 2 + 1] = aabb_y;
}

kernel void nd_rasterize_forward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,
    device int* tile_bins, // int2
    constant float* packed_xy_opac, // float3: (x, y, sigmoid(opacity))
    constant float* packed_conic,   // float3
    constant float* packed_rgb,     // float3: raw SH (NOT clamped)
    constant float* packed_opacity_comp,
    device float* final_Ts,
    device int* final_index,
    device float* out_img,
    constant float* background,
    constant uint2& blockDim,
    constant half* packed_conic_half,
    constant half* packed_rgb_half,
    constant half* packed_opacity_comp_half,
    constant uint& use_half_sorted_buffers,
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]]
) {
    // Threadgroup-batched forward rasterization: all threads in a tile
    // cooperatively load Gaussian data into shared memory, then read from it.
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    // Map pixel coords back to parent 16x16 tile for tile_bins lookup
    int32_t tile_id = ((int)i / BLOCK_Y) * tile_bounds.x + ((int)j / BLOCK_X);
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * (int)img_size.x + j;

    const bool inside = (i < (int)img_size.y && j < (int)img_size.x);

    // Which gaussians to look through in this tile. Keep a local copy before
    // narrowing tile_bins.y for the backward pass.
    int2 range = int2(tile_bins[2 * tile_id], tile_bins[2 * tile_id + 1]);
    const int num_batches = (range.y - range.x + RAST_BLOCK_SIZE - 1) / RAST_BLOCK_SIZE;

    // threadgroup shared memory for batch loading
    threadgroup float3 xy_opacity_batch[RAST_BLOCK_SIZE];
    threadgroup float3 conic_batch[RAST_BLOCK_SIZE];
    threadgroup float3 rgbs_batch[RAST_BLOCK_SIZE];
    threadgroup float opacity_comp_batch[RAST_BLOCK_SIZE];

    float T = 1.f;
    float3 pix_out = {0.f, 0.f, 0.f};
    int last_contributor = range.x - 1;
    bool done = false;
    threadgroup atomic_int max_useful_isect;

    if (tr == 0) {
        atomic_store_explicit(&max_useful_isect, range.x, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int b = 0; b < num_batches; ++b) {
        // sync before loading next batch
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each thread loads one gaussian into shared memory
        int batch_start = range.x + RAST_BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            // Sequential reads from packed sorted-order buffers
            xy_opacity_batch[tr] = read_packed_float3(packed_xy_opac, idx);
            conic_batch[tr] = read_packed_sorted_float3(
                packed_conic, packed_conic_half, idx, use_half_sorted_buffers);
            // packed_rgb has raw SH output — clamp_min(raw + 0.5, 0)
            const float3 raw_c = read_packed_sorted_float3(
                packed_rgb, packed_rgb_half, idx, use_half_sorted_buffers);
            rgbs_batch[tr] = max(raw_c + 0.5f, 0.0f);
            opacity_comp_batch[tr] = read_packed_sorted_float(
                packed_opacity_comp, packed_opacity_comp_half, idx, use_half_sorted_buffers);
        }
        // wait for all threads to finish loading
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (done || !inside) continue;

        int batch_size = min(RAST_BLOCK_SIZE, range.y - batch_start);

        // process gaussians in this batch
        for (int t = 0; t < batch_size; ++t) {
            const float3 conic_local = conic_batch[t];
            const float3 xy_opac = xy_opacity_batch[t];
            const float2 delta = {xy_opac.x - px, xy_opac.y - py};

            const float sigma = fma(0.5f,
                fma(conic_local.x, delta.x * delta.x, conic_local.z * delta.y * delta.y),
                conic_local.y * delta.x * delta.y);
            // Early-out: skip exp() when the result is guaranteed to be discarded.
            // alpha = min(0.999, opacity * exp(-sigma)), discarded when alpha < 1/255.
            // opacity = sigmoid(raw) ∈ [0,1], so alpha ≤ exp(-sigma).
            // exp(-5.55) ≈ 0.00389 < 1/255 ≈ 0.00392, so sigma ≥ 5.55 ⟹ alpha < 1/255.
            // Empirically 94% of evaluations have sigma ≥ 5.55 (garden scene, mipnerf360).
            if (sigma < 0.f || sigma >= 5.55f) {
                continue;
            }

            const float alpha = min(0.999f, xy_opac.z * opacity_comp_batch[t] * exp(-sigma));
            if (alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) {
                last_contributor = batch_start + t - 1;
                done = true;
                break;
            }

            const float vis = alpha * T;
            const float3 rgb = rgbs_batch[t];
            pix_out = fma(rgb, vis, pix_out);
            T = next_T;
            last_contributor = batch_start + t;
        }
    }

    if (inside) {
        final_Ts[pix_id] = T;
        final_index[pix_id] = last_contributor;
        if (last_contributor >= range.x) {
            atomic_fetch_max_explicit(&max_useful_isect, last_contributor + 1, memory_order_relaxed);
        }
        // Fused clamp_max(output, 1.0) — saturate clamps to [0,1]
        float3 bg = {background[0], background[1], background[2]};
        float3 final_rgb = saturate(fma(bg, T, pix_out));
        out_img[CHANNELS * pix_id + 0] = final_rgb.x;
        out_img[CHANNELS * pix_id + 1] = final_rgb.y;
        out_img[CHANNELS * pix_id + 2] = final_rgb.z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tr == 0) {
        write_packed_int2y(tile_bins, tile_id, atomic_load_explicit(&max_useful_isect, memory_order_relaxed));
    }
}

kernel void compute_sh_forward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* means3d, // float3
    constant float4& cam_pos,
    constant float* coeffs,
    device float* colors,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    // Compute view direction from means and camera position (fused, avoids 3 MPS ops)
    float3 viewdir = normalize(read_packed_float3(means3d, idx) - cam_pos.xyz);

    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color(
        degrees_to_use, viewdir, &(coeffs[idx_sh]), &(coeffs[idx_sh + CHANNELS]), &(colors[idx_col])
    );
}

kernel void compute_sh_backward_kernel(
    constant uint& num_points,
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float* means3d, // float3
    constant float4& cam_pos,
    constant float* v_colors,
    device float* v_coeffs,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) {
        return;
    }
    // Recompute view direction (same as forward)
    float3 viewdir = normalize(read_packed_float3(means3d, idx) - cam_pos.xyz);

    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint idx_sh = num_bases * num_channels * idx;
    uint idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp(
        degrees_to_use, viewdir, &(v_colors[idx_col]), &(v_coeffs[idx_sh]), &(v_coeffs[idx_sh + CHANNELS])
    );
}

// Build (tile_id, depth) pairs for each gaussian-tile intersection.
kernel void map_gaussian_to_intersects_kernel(
    constant int& num_points,
    constant float* xys, // float2
    constant float* depths,
    constant int* radii,
    constant int32_t* num_tiles_hit,
    constant uint3& tile_bounds,
    constant uint& capacity,
    device int64_t* isect_ids,
    device int32_t* gaussian_ids,
    constant float* aabb, // float2: per-axis pixel extents
    device atomic_uint* overflow_flag, // set to 1 if any intersection exceeds capacity
    constant float* conics,
    constant float* opacities,
    constant float* opacity_comp,
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points)
        return;
    if (radii[idx] <= 0)
        return;
    float2 center = read_packed_float2(xys, idx);
    float3 conic = read_packed_float3(conics, idx);
    float opacity = (1.0f / (1.0f + exp(-opacities[idx]))) * opacity_comp[idx];
    if (!isfinite(opacity) || opacity < (1.0f / 255.0f))
        return;
    float power_threshold = log(255.0f * opacity);

    // get the tile bbox for gaussian using AABB extents
    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : num_tiles_hit[idx - 1];
    uint64_t depth_bits = (uint64_t)as_type<uint>(depths[idx]);
    for (uint i = tile_min.y; i < tile_max.y; ++i) {
        for (uint j = tile_min.x; j < tile_max.x; ++j) {
            if (!will_primitive_contribute(tile_rect(uint2(j, i)), center, conic, power_threshold)) {
                continue;
            }
            if ((uint)cur_idx >= capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }
            uint64_t tile_id = (uint64_t)(i * tile_bounds.x + j);
            isect_ids[cur_idx] = (int64_t)((tile_id << 32) | depth_bits);
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
}

kernel void map_gaussian_to_intersects_u32_kernel(
    constant int& num_points,
    constant float* xys, // float2
    constant float* depths,
    constant int* radii,
    constant int32_t* num_tiles_hit,
    constant uint3& tile_bounds,
    constant uint& capacity,
    device uint* isect_ids,
    device int32_t* gaussian_ids,
    constant float* aabb, // float2: per-axis pixel extents
    device atomic_uint* overflow_flag,
    constant float* conics,
    constant float* opacities,
    constant float* opacity_comp,
    uint3 gp [[thread_position_in_grid]]
) {
    uint idx = gp.x;
    if (idx >= (uint)num_points)
        return;
    if (radii[idx] <= 0)
        return;
    float2 center = read_packed_float2(xys, idx);
    float3 conic = read_packed_float3(conics, idx);
    float opacity = (1.0f / (1.0f + exp(-opacities[idx]))) * opacity_comp[idx];
    if (!isfinite(opacity) || opacity < (1.0f / 255.0f))
        return;
    float power_threshold = log(255.0f * opacity);

    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    int32_t cur_idx = (idx == 0) ? 0 : num_tiles_hit[idx - 1];
    // project_and_sh_forward_kernel culls non-contributing/behind-camera splats
    // before count prefixing, so positive float bits sort monotonically here.
    uint depth_q16 = as_type<uint>(depths[idx]) >> 16;
    for (uint i = tile_min.y; i < tile_max.y; ++i) {
        for (uint j = tile_min.x; j < tile_max.x; ++j) {
            if (!will_primitive_contribute(tile_rect(uint2(j, i)), center, conic, power_threshold)) {
                continue;
            }
            if ((uint)cur_idx >= capacity) {
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                return;
            }
            uint tile_id = i * tile_bounds.x + j;
            isect_ids[cur_idx] = (tile_id << 16) | depth_q16;
            gaussian_ids[cur_idx] = idx;
            ++cur_idx;
        }
    }
}

// Find start/end offsets for each tile in the sorted intersection array.
kernel void get_tile_bin_edges_kernel(
    constant uint& capacity,
    constant int64_t* isect_ids_sorted,
    device int* tile_bins, // int2
    device const int32_t* cum_tiles_hit,
    constant uint& num_points,
    uint idx [[thread_position_in_grid]]
) {
    // Read actual intersection count from GPU-resident prefix sum
    uint num_intersects = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    if (idx >= num_intersects)
        return;
    // Save the indices where the tile_id changes. The final element still has
    // to run the transition case; otherwise a last-element tile change leaves
    // the previous tile open and the final tile without a start offset.
    int32_t cur_tile_idx = (int32_t)(((uint64_t)isect_ids_sorted[idx]) >> 32);
    if (idx == 0) {
        write_packed_int2x(tile_bins, cur_tile_idx, 0);
    } else {
        int32_t prev_tile_idx = (int32_t)(((uint64_t)isect_ids_sorted[idx - 1]) >> 32);
        if (prev_tile_idx != cur_tile_idx) {
            write_packed_int2y(tile_bins, prev_tile_idx, idx);
            write_packed_int2x(tile_bins, cur_tile_idx, idx);
        }
    }
    if (idx == num_intersects - 1) {
        write_packed_int2y(tile_bins, cur_tile_idx, num_intersects);
    }
}

kernel void get_tile_bin_edges_u32_kernel(
    constant uint& capacity,
    constant uint* isect_ids_sorted,
    device int* tile_bins, // int2
    device const int32_t* cum_tiles_hit,
    constant uint& num_points,
    uint idx [[thread_position_in_grid]]
) {
    uint num_intersects = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    if (idx >= num_intersects)
        return;
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 16);
    if (idx == 0) {
        write_packed_int2x(tile_bins, cur_tile_idx, 0);
    } else {
        int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 16);
        if (prev_tile_idx != cur_tile_idx) {
            write_packed_int2y(tile_bins, prev_tile_idx, idx);
            write_packed_int2x(tile_bins, cur_tile_idx, idx);
        }
    }
    if (idx == num_intersects - 1) {
        write_packed_int2y(tile_bins, cur_tile_idx, num_intersects);
    }
}
