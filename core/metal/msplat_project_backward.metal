#include "msplat_common.metal"

kernel void project_gaussians_backward_kernel(
    constant int& num_points,
    constant float* means3d, // float3
    constant float* scales, // float3
    constant float& glob_scale,
    constant float* quats, // float4
    constant float* viewmat,
    constant float* projmat,
    constant float4& intrins,
    constant uint2& img_size,
    constant float* cov3d,
    constant int* radii,
    constant float* conics, // float3
    constant float* v_xy, // float2
    constant float* v_depth,
    constant float* v_conic, // float3
    device float* v_cov2d, // float3
    device float* v_cov3d,
    device float* v_mean3d, // float3
    device float* v_scale, // float3
    device float* v_quat, // float4
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = read_packed_float3(means3d, idx);
    float fx = intrins.x;
    float fy = intrins.y;
    // get v_mean3d from v_xy
    write_packed_float3(
        v_mean3d, idx, 
        project_pix_vjp(projmat, p_world, img_size, read_packed_float2(v_xy, idx))
    );

    // get z gradient contribution to mean3d gradient
    // z = viemwat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] *
    // mean3d.z + viewmat[11]
    float v_z = v_depth[idx];
    write_packed_float3(
        v_mean3d, idx, 
        read_packed_float3(v_mean3d, idx) + float3(viewmat[8], viewmat[9], viewmat[10]) * v_z
    );

    // get v_cov2d
    cov2d_to_conic_vjp(
        read_packed_float3(conics, idx), 
        read_packed_float3(v_conic, idx), 
        &(v_cov2d[3*idx])
    );
    // get v_cov3d (and v_mean3d contribution)
    float tan_fovx = 0.5f * (float)img_size.x / fx;
    float tan_fovy = 0.5f * (float)img_size.y / fy;
    float3 p_view = transform_4x3(viewmat, p_world);
    project_cov3d_ewa_vjp(
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        tan_fovx,
        tan_fovy,
        read_packed_float3(v_cov2d, idx),
        &(v_mean3d[3*idx]),
        &(v_cov3d[6 * idx]),
        p_view
    );
    // get v_scale and v_quat
    // scales are in log-space; exp() here and apply chain rule for dL/d(log_scale)
    float3 exp_scale = exp(read_packed_float3(scales, idx));
    scale_rot_to_cov3d_vjp(
        exp_scale,
        glob_scale,
        read_packed_float4(quats, idx),
        &(v_cov3d[6 * idx]),
        &(v_scale[3*idx]),
        &(v_quat[4*idx])
    );
    // chain rule: dL/d(log_s) = dL/d(exp_s) * exp(log_s)
    v_scale[3*idx + 0] *= exp_scale.x;
    v_scale[3*idx + 1] *= exp_scale.y;
    v_scale[3*idx + 2] *= exp_scale.z;
}

kernel void compute_cov2d_bounds_kernel(
    constant uint& num_pts, 
    constant float* covs2d, 
    device float* conics, 
    device float* radii,
    uint row [[thread_index_in_threadgroup]]
) {
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;
    float3 conic;
    float radius;
    float3 cov2d{
        (float)covs2d[index], (float)covs2d[index + 1], (float)covs2d[index + 2]
    };
    compute_cov2d_bounds(cov2d, conic, radius);
    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

// Fused Adam optimizer kernel: single-pass update for params, exp_avg, exp_avg_sq.
// Precomputed on CPU: step_size = lr / (1 - beta1^t), bc2_sqrt = sqrt(1 - beta2^t)
// Fused per-step gradient accumulation for densification.
// Replaces ~8 MPS dispatches (boolean mask, vector_norm, index_put_, max) with 1 kernel.
kernel void accumulate_grad_stats_kernel(
    constant int& num_points,
    constant int* radii [[buffer(1)]],
    constant float* refine_weight_grad [[buffer(2)]],
    device float* vis_counts [[buffer(3)]],      // (N,) in-place
    device float* xys_grad_norm [[buffer(4)]],   // (N,) in-place
    device float* max_2d_size [[buffer(5)]],     // (N,) in-place
    constant float* aabb [[buffer(6)]],          // float2 extent per Gaussian
    constant float& inv_max_dim [[buffer(7)]],   // 1.0 / max(H, W)
    constant float& inv_width [[buffer(8)]],
    constant float& inv_height [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points) return;
    if (radii[idx] <= 0) return;

    vis_counts[idx] += 1.0f;

    float refine_weight = refine_weight_grad[idx] * (2.0f * inv_max_dim);
    if (isfinite(refine_weight)) {
        xys_grad_norm[idx] = max(xys_grad_norm[idx], refine_weight);
    }

    float2 extent = read_packed_float2(aabb, idx);
    float screen_size = max(extent.x * inv_width, extent.y * inv_height);
    if (!isfinite(screen_size)) {
        screen_size = (float)radii[idx] * inv_max_dim;
    }
    max_2d_size[idx] = max(max_2d_size[idx], screen_size);
}

kernel void accumulate_pup_hessian_kernel(
    constant int& num_points,
    constant int* radii [[buffer(1)]],
    constant float* v_mean3d [[buffer(2)]],
    constant float* v_scale [[buffer(3)]],
    device float* pup_hessian [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)num_points || radii[idx] <= 0) return;

    float j[6] = {
        v_mean3d[idx * 3 + 0],
        v_mean3d[idx * 3 + 1],
        v_mean3d[idx * 3 + 2],
        v_scale[idx * 3 + 0],
        v_scale[idx * 3 + 1],
        v_scale[idx * 3 + 2],
    };

    uint base = idx * 36;
    for (uint row = 0; row < 6; ++row) {
        float jr = isfinite(j[row]) ? j[row] : 0.0f;
        for (uint col = 0; col < 6; ++col) {
            float jc = isfinite(j[col]) ? j[col] : 0.0f;
            pup_hessian[base + row * 6 + col] += jr * jc;
        }
    }
}

kernel void fused_adam_kernel(
    device float * params [[buffer(0)]],
    device const float * grads [[buffer(1)]],
    device float * exp_avg [[buffer(2)]],
    device float * exp_avg_sq [[buffer(3)]],
    constant float & step_size [[buffer(4)]],
    constant float & beta1 [[buffer(5)]],
    constant float & beta2 [[buffer(6)]],
    constant float & bc2_sqrt [[buffer(7)]],
    constant float & eps [[buffer(8)]],
    constant uint & n [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    float g = grads[tid];
    float m = fma(beta1, exp_avg[tid], (1.0f - beta1) * g);
    float v = fma(beta2, exp_avg_sq[tid], (1.0f - beta2) * g * g);

    params[tid] -= step_size * m / (sqrt(v) / bc2_sqrt + eps);

    exp_avg[tid] = m;
    exp_avg_sq[tid] = v;
}

kernel void apply_mean_noise_kernel(
    device float* means3d [[buffer(0)]],
    constant float* opacities [[buffer(1)]],
    constant int* radii [[buffer(2)]],
    constant uint& num_points [[buffer(3)]],
    constant float& noise_scale [[buffer(4)]],
    constant float& max_noise [[buffer(5)]],
    constant uint& seed [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points || radii[idx] <= 0 || noise_scale <= 0.0f) {
        return;
    }

    float opacity = 1.0f / (1.0f + exp(-opacities[idx]));
    float weight = pow(clamp(1.0f - opacity, 0.0f, 1.0f), 150.0f) * noise_scale;
    if (weight <= 0.0f) {
        return;
    }

    means3d[idx * 3 + 0] += clamp(msplat_normal_sample(seed, idx, 0u) * weight, -max_noise, max_noise);
    means3d[idx * 3 + 1] += clamp(msplat_normal_sample(seed, idx, 1u) * weight, -max_noise, max_noise);
    means3d[idx * 3 + 2] += clamp(msplat_normal_sample(seed, idx, 2u) * weight, -max_noise, max_noise);
}
