#include "msplat_common.metal"

// ===== Fused Projection + SH Kernels =====
// Combines project_gaussians_forward_kernel + compute_sh_forward_kernel into one dispatch.
// Saves 1 kernel dispatch + 1 read of means3d per direction. Skips SH for culled gaussians.

kernel void project_and_sh_forward_kernel(
    // Projection args
    constant int& num_points,
    constant float* means3d,
    constant float* scales,
    constant float& glob_scale,
    constant float* quats,
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant uint3& tile_bounds,
    constant float& clip_thresh,
    device float* xys,
    device float* depths,
    device int* radii,
    device float* conics,
    device int32_t* num_tiles_hit,
    // SH args
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float4& cam_pos,
    constant float* features_dc,
    constant float* features_rest,
    device float* colors,
    device float* aabb, // float2: per-axis pixel extents
    device float* opacity_comp,
    constant uint& use_mip_splatting,
    constant float* opacities,
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

    float3 scale = exp(read_packed_float3(scales, idx));
    if (!finite_float3(scale)) {
        return;
    }
    float4 quat = read_packed_float4(quats, idx);
    if (!valid_quaternion(quat)) {
        return;
    }
    // Compute cov3d in thread-local registers (no device memory round-trip)
    float local_cov3d[6];
    scale_rot_to_cov3d(scale, glob_scale, quat, local_cov3d);

    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5f * img_size.x / fx;
    float tan_fovy = 0.5f * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        local_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view
    );
    const bool use_mip_splatting_specialized = project_sh_use_mip_splatting(use_mip_splatting);
    float opacity_comp_value = use_mip_splatting_specialized ? mip_opacity_compensation(cov2d) : 1.0f;
    opacity_comp[idx] = opacity_comp_value;

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) {
        return;
    }
    write_packed_float3(conics, idx, conic);

    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});

    float opacity = (1.0f / (1.0f + exp(-opacities[idx]))) * opacity_comp_value;
    if (!isfinite(opacity) || opacity < (1.0f / 255.0f)) {
        return;
    }
    float power_threshold = log(255.0f * opacity);
    float2 extent = compute_bbox_extent(conic, power_threshold);
    if (!(extent.x >= 0.0f && extent.y >= 0.0f)) {
        return;
    }

    uint2 tile_min, tile_max;
    get_tile_bbox(center, extent, (int3)tile_bounds, tile_min, tile_max);
    int32_t tile_area = 0;
    for (uint i = tile_min.y; i < tile_max.y; i++) {
        for (uint j = tile_min.x; j < tile_max.x; j++) {
            if (will_primitive_contribute(tile_rect(uint2(j, i)), center, conic, power_threshold)) {
                tile_area++;
            }
        }
    }

    if (tile_area <= 0) {
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = max(1, (int)ceil(max(extent.x, extent.y)));
    write_packed_float2(xys, idx, center);
    aabb[idx * 2] = extent.x;
    aabb[idx * 2 + 1] = extent.y;

    // SH: compute colors for non-culled gaussians (reuse p_world from registers)
    float3 viewdir = normalize(p_world - cam_pos.xyz);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;
    const uint effective_degrees_to_use = project_sh_degrees_to_use(degrees_to_use);
    sh_coeffs_to_color(effective_degrees_to_use, viewdir, &(features_dc[dc_idx]), &(features_rest[rest_idx]), &(colors[idx_col]));
}

// Adam update helper — applies one Adam step to a single element.
// Computes in registers, writes param/exp_avg/exp_avg_sq back to device memory.
inline void adam_update_element(
    device float& param, device float& ea, device float& eas,
    float grad, float step_size, float beta1, float beta2, float bc2_sqrt, float eps
) {
    float m = fma(beta1, ea, (1.0f - beta1) * grad);
    float v = fma(beta2, eas, (1.0f - beta2) * grad * grad);
    param -= step_size * m / (sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

inline void adam_update_element_with_v(
    device float& param, device float& ea, device float& eas,
    float grad, float v, float step_size, float beta1, float bc2_sqrt, float eps
) {
    float m = fma(beta1, ea, (1.0f - beta1) * grad);
    param -= step_size * m / (sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

// Packed Adam hyperparameters for SH groups (passed via setBytes)
struct SHAdamParams {
    float dc_step_size;
    float dc_bc2_sqrt;
    float rest_step_size;
    float rest_bc2_sqrt;
    float beta1;
    float beta2;
    float eps;
    uint reduce_second_moment;
};

kernel void project_and_sh_backward_kernel(
    // Projection backward args
    constant int& num_points,
    constant float* means3d,
    constant float* scales,
    constant float& glob_scale,
    constant float* quats,
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant int* radii,
    constant float* conics,
    constant float* v_xy,
    constant float* v_depth,
    constant float* v_conic,
    device float* v_mean3d,
    device float* v_scale,
    device float* v_quat,
    // SH backward + fused Adam args
    constant uint& degree,
    constant uint& degrees_to_use,
    constant float4& cam_pos,
    constant float* v_colors,
    device float* features_dc,         // params (read-write for Adam)
    device float* features_rest,       // params (read-write for Adam)
    device float* dc_exp_avg,          // Adam state
    device float* dc_exp_avg_sq,
    device float* rest_exp_avg,
    device float* rest_exp_avg_sq,
    constant SHAdamParams& adam_hp,
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;

    // Projection backward: v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx,
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // z gradient contribution to mean3d
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx,
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // v_cov2d from v_conic (thread-local, no device memory round-trip)
    float local_v_cov2d[3];
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx),
        read_packed_float3(v_conic, idx),
        local_v_cov2d
    );

    // Recompute cov3d from scales+quats (avoids saving/reading 3.6MB tensor)
    float3 exp_scale = exp(read_packed_float3(scales, idx));
    float4 quat = read_packed_float4(quats, idx);
    float local_cov3d[6];
    scale_rot_to_cov3d(exp_scale, glob_scale, quat, local_cov3d);

    // v_cov3d (thread-local) and v_mean3d contribution
    float tan_fovx = 0.5f * (float)img_size.x / fx;
    float tan_fovy = 0.5f * (float)img_size.y / fy;
    float3 p_view = transform_4x3(viewmat, p_world);
    float local_v_cov3d[6];
    project_cov3d_ewa_vjp(
        local_cov3d,
        viewmat,
        fx,
        fy,
        tan_fovx,
        tan_fovy,
        float3(local_v_cov2d[0], local_v_cov2d[1], local_v_cov2d[2]),
        &(v_mean3d[3*idx]),
        local_v_cov3d,
        p_view
    );

    // v_scale and v_quat (reads v_cov3d from thread-local)
    scale_rot_to_cov3d_vjp(
        exp_scale,
        glob_scale,
        quat,
        local_v_cov3d,
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
    // Chain rule: dL/d(log_s) = dL/d(exp_s) * exp(log_s)
    v_scale[3*idx + 0] *= exp_scale.x;
    v_scale[3*idx + 1] *= exp_scale.y;
    v_scale[3*idx + 2] *= exp_scale.z;

    // ---- Fused SH backward + Adam ----
    // Compute SH gradients in registers and apply Adam inline.
    // Eliminates v_features_dc/v_features_rest write+read round-trip (~600 MB/iter at 1.6M gaussians).
    float3 viewdir = normalize(p_world - cam_pos.xyz);
    const uint num_channels = 3;
    uint num_bases = num_sh_bases(degree);
    uint dc_idx = num_channels * idx;
    uint rest_idx = (num_bases - 1) * num_channels * idx;
    uint idx_col = num_channels * idx;

    float vc[3] = { v_colors[idx_col], v_colors[idx_col + 1], v_colors[idx_col + 2] };
    float x = viewdir.x, y = viewdir.y, z = viewdir.z;
    float xx = x*x, xy = x*y, xz = x*z, yy = y*y, yz = y*z, zz = z*z;
    float sh1[3] = { -SH_C1 * y, SH_C1 * z, -SH_C1 * x };
    float sh2[5] = {
        SH_C2[0] * xy,
        SH_C2[1] * yz,
        SH_C2[2] * (2.f * zz - xx - yy),
        SH_C2[3] * xz,
        SH_C2[4] * (xx - yy)
    };
    float sh3[7] = {
        SH_C3[0] * y * (3.f * xx - yy),
        SH_C3[1] * xy * z,
        SH_C3[2] * y * (4.f * zz - xx - yy),
        SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy),
        SH_C3[4] * x * (4.f * zz - xx - yy),
        SH_C3[5] * z * (xx - yy),
        SH_C3[6] * x * (xx - 3.f * yy)
    };

    const uint effective_degrees_to_use = project_sh_degrees_to_use(degrees_to_use);
    bool reduce_v = project_sh_reduce_second_moment(adam_hp.reduce_second_moment);
    float shared_v = 0.0f;
    if (reduce_v) {
        float grad_sq_sum = 0.0f;
        float old_v_sum = 0.0f;
        uint total = 0;

        for (int c = 0; c < 3; c++) {
            float g = SH_C0 * vc[c];
            grad_sq_sum += g * g;
            old_v_sum += dc_exp_avg_sq[dc_idx + c];
            total++;
        }
        if (effective_degrees_to_use >= 1) {
            for (int b = 0; b < 3; b++) {
                for (int c = 0; c < 3; c++) {
                    uint i = rest_idx + b * 3 + c;
                    float g = sh1[b] * vc[c];
                    grad_sq_sum += g * g;
                    old_v_sum += rest_exp_avg_sq[i];
                    total++;
                }
            }
        }
        if (effective_degrees_to_use >= 2) {
            for (int b = 0; b < 5; b++) {
                for (int c = 0; c < 3; c++) {
                    uint i = rest_idx + (3 + b) * 3 + c;
                    float g = sh2[b] * vc[c];
                    grad_sq_sum += g * g;
                    old_v_sum += rest_exp_avg_sq[i];
                    total++;
                }
            }
        }
        if (effective_degrees_to_use >= 3) {
            for (int b = 0; b < 7; b++) {
                for (int c = 0; c < 3; c++) {
                    uint i = rest_idx + (8 + b) * 3 + c;
                    float g = sh3[b] * vc[c];
                    grad_sq_sum += g * g;
                    old_v_sum += rest_exp_avg_sq[i];
                    total++;
                }
            }
        }

        float inv_total = 1.0f / (float)total;
        shared_v = fma(adam_hp.beta2, old_v_sum * inv_total,
                       (1.0f - adam_hp.beta2) * grad_sq_sum * inv_total);
    }

    // DC: grad = SH_C0 * v_colors[c]
    for (int c = 0; c < 3; c++) {
        float g = SH_C0 * vc[c];
        if (reduce_v) {
            adam_update_element_with_v(features_dc[dc_idx + c], dc_exp_avg[dc_idx + c], dc_exp_avg_sq[dc_idx + c],
                                      g, shared_v, adam_hp.dc_step_size, adam_hp.beta1, adam_hp.dc_bc2_sqrt, adam_hp.eps);
        } else {
            adam_update_element(features_dc[dc_idx + c], dc_exp_avg[dc_idx + c], dc_exp_avg_sq[dc_idx + c],
                               g, adam_hp.dc_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.dc_bc2_sqrt, adam_hp.eps);
        }
    }

    if (effective_degrees_to_use < 1) return;

    // SH degree 1 (3 bases)
    for (int b = 0; b < 3; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + b * 3 + c;
            float g = sh1[b] * vc[c];
            if (reduce_v) {
                adam_update_element_with_v(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                          g, shared_v, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            } else {
                adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                   g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            }
        }
    }

    if (effective_degrees_to_use < 2) return;

    // SH degree 2 (5 bases)
    for (int b = 0; b < 5; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + (3 + b) * 3 + c;
            float g = sh2[b] * vc[c];
            if (reduce_v) {
                adam_update_element_with_v(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                          g, shared_v, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            } else {
                adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                   g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            }
        }
    }

    if (effective_degrees_to_use < 3) return;

    // SH degree 3 (7 bases)
    for (int b = 0; b < 7; b++) {
        for (int c = 0; c < 3; c++) {
            uint i = rest_idx + (8 + b) * 3 + c;
            float g = sh3[b] * vc[c];
            if (reduce_v) {
                adam_update_element_with_v(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                          g, shared_v, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            } else {
                adam_update_element(features_rest[i], rest_exp_avg[i], rest_exp_avg_sq[i],
                                   g, adam_hp.rest_step_size, adam_hp.beta1, adam_hp.beta2, adam_hp.rest_bc2_sqrt, adam_hp.eps);
            }
        }
    }
}
