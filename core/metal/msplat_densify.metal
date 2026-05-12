#include "msplat_common.metal"

// ============================================================
// GPU Densification Kernels (Phase 3)
// ============================================================

#define DENSIFY_NOTHING 0
#define DENSIFY_SPLIT   1
#define DENSIFY_DUP     2

// Classify each gaussian as split, dup, or nothing based on gradient and scale thresholds.
kernel void densify_classify_kernel(
    constant int& N,
    constant float* xys_grad_norm    [[buffer(1)]],
    constant float* vis_counts       [[buffer(2)]],
    constant float* scales           [[buffer(3)]],  // [N,3] log-space
    constant float* max_2d_size      [[buffer(4)]],
    constant float& half_max_dim     [[buffer(5)]],  // 0.5 * max(W,H)
    constant float& grad_thresh      [[buffer(6)]],
    constant float& size_thresh      [[buffer(7)]],
    constant float& screen_thresh    [[buffer(8)]],
    constant int& check_screen       [[buffer(9)]],
    device int* split_flag           [[buffer(10)]],
    device int* dup_flag             [[buffer(11)]],
    constant float& growth_select_fraction [[buffer(12)]],
    constant uint& growth_seed       [[buffer(13)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N) return;
    float vc = vis_counts[idx];
    if (vc <= 0.0f) { split_flag[idx] = 0; dup_flag[idx] = 0; return; }

    float refine_weight = xys_grad_norm[idx] * half_max_dim;
    bool high_grad = refine_weight > grad_thresh;

    float s0 = scales[idx*3], s1 = scales[idx*3+1], s2 = scales[idx*3+2];
    float max_scale = max(max(exp(s0), exp(s1)), exp(s2));
    bool is_large = max_scale > size_thresh;

    bool selected = growth_select_fraction >= 1.0f
        || msplat_hash_unit(growth_seed, idx, 17u) < growth_select_fraction;

    bool do_split = is_large;
    if (check_screen && max_2d_size[idx] > screen_thresh) do_split = true;
    do_split = do_split && high_grad && selected;

    bool do_dup = !is_large && high_grad && selected;

    split_flag[idx] = do_split ? 1 : 0;
    dup_flag[idx]   = do_dup   ? 1 : 0;
}

// Brush-style split. One thread per original Gaussian.
// Each selected Gaussian is kept as the first mixture component: its mean moves
// by -sample, its scale/opacity shrink, and one child is appended at [N + ord]
// with mean +sample and matching shrunken scale/opacity.
kernel void densify_append_split_kernel(
    constant int& N,
    constant int* split_flag         [[buffer(1)]],
    constant int* split_prefix       [[buffer(2)]],  // inclusive prefix sum
    constant uint& split_seed        [[buffer(3)]],
    constant float& log_scale_factor [[buffer(4)]],  // log(1/sqrt(2))
    device float* means_buf          [[buffer(5)]],
    device float* scales_buf         [[buffer(6)]],
    device float* quats_buf          [[buffer(7)]],
    device float* featuresDc_buf     [[buffer(8)]],
    device float* featuresRest_buf   [[buffer(9)]],
    device float* opacities_buf      [[buffer(10)]],
    constant int& fr_stride          [[buffer(11)]],  // featuresRest stride (e.g. 45)
    device float* adam_ea0           [[buffer(12)]],  // adam_exp_avg_buf[0..5]
    device float* adam_ea1           [[buffer(13)]],
    device float* adam_ea2           [[buffer(14)]],
    device float* adam_ea3           [[buffer(15)]],
    device float* adam_ea4           [[buffer(16)]],
    device float* adam_ea5           [[buffer(17)]],
    device float* adam_es0           [[buffer(18)]],  // adam_exp_avg_sq_buf[0..5]
    device float* adam_es1           [[buffer(19)]],
    device float* adam_es2           [[buffer(20)]],
    device float* adam_es3           [[buffer(21)]],
    device float* adam_es4           [[buffer(22)]],
    device float* adam_es5           [[buffer(23)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || split_flag[idx] == 0) return;

    int ord = split_prefix[idx] - 1;  // 0-based ordinal among splits
    int child = N + ord;

    // Read parent quaternion and normalize
    float qw = quats_buf[idx*4], qx = quats_buf[idx*4+1];
    float qy = quats_buf[idx*4+2], qz = quats_buf[idx*4+3];
    float qlen = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen;

    // Parent scale (exp)
    float mean_x = means_buf[idx*3], mean_y = means_buf[idx*3+1], mean_z = means_buf[idx*3+2];
    float log_sx = scales_buf[idx*3], log_sy = scales_buf[idx*3+1], log_sz = scales_buf[idx*3+2];
    float sx = exp(log_sx), sy = exp(log_sy), sz = exp(log_sz);

    float raw_opacity = opacities_buf[idx];
    float parent_opacity = 1.0f / (1.0f + exp(-raw_opacity));
    float new_opacity = 1.0f - sqrt(max(0.0f, 1.0f - parent_opacity));
    new_opacity = clamp(new_opacity, 1.0f / 255.0f, 1.0f - 1.0f / 255.0f);
    float new_raw_opacity = log(new_opacity / (1.0f - new_opacity));

    // Scale random sample by parent scale, then rotate by parent quaternion.
    const float split_offset_std = 0.7071067811865476f;
    float r0 = msplat_normal_sample(split_seed, (uint)ord, 0u) * split_offset_std * sx;
    float r1 = msplat_normal_sample(split_seed, (uint)ord, 1u) * split_offset_std * sy;
    float r2 = msplat_normal_sample(split_seed, (uint)ord, 2u) * split_offset_std * sz;
    float v0 = (1-2*(qy*qy+qz*qz))*r0 + 2*(qx*qy-qw*qz)*r1 + 2*(qx*qz+qw*qy)*r2;
    float v1 = 2*(qx*qy+qw*qz)*r0 + (1-2*(qx*qx+qz*qz))*r1 + 2*(qy*qz-qw*qx)*r2;
    float v2 = 2*(qx*qz-qw*qy)*r0 + 2*(qy*qz+qw*qx)*r1 + (1-2*(qx*qx+qy*qy))*r2;

    means_buf[idx*3]   = mean_x - v0;
    means_buf[idx*3+1] = mean_y - v1;
    means_buf[idx*3+2] = mean_z - v2;
    scales_buf[idx*3]   = log_sx + log_scale_factor;
    scales_buf[idx*3+1] = log_sy + log_scale_factor;
    scales_buf[idx*3+2] = log_sz + log_scale_factor;
    opacities_buf[idx] = new_raw_opacity;

    means_buf[child*3]   = mean_x + v0;
    means_buf[child*3+1] = mean_y + v1;
    means_buf[child*3+2] = mean_z + v2;
    scales_buf[child*3]   = log_sx + log_scale_factor;
    scales_buf[child*3+1] = log_sy + log_scale_factor;
    scales_buf[child*3+2] = log_sz + log_scale_factor;

    quats_buf[child*4] = qw;
    quats_buf[child*4+1] = qx;
    quats_buf[child*4+2] = qy;
    quats_buf[child*4+3] = qz;
    for (int j = 0; j < 3; j++) featuresDc_buf[child*3+j] = featuresDc_buf[idx*3+j];
    for (int j = 0; j < fr_stride; j++) featuresRest_buf[child*fr_stride+j] = featuresRest_buf[idx*fr_stride+j];
    opacities_buf[child] = new_raw_opacity;

    // Zero optimizer state for both split components. The existing parent is
    // structurally displaced/shrunk in-place, so stale Adam moments can push it
    // back along the pre-split trajectory.
    for (int j = 0; j < 3; j++) { adam_ea0[idx*3+j] = 0; adam_es0[idx*3+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea1[idx*3+j] = 0; adam_es1[idx*3+j] = 0; }
    for (int j = 0; j < 4; j++) { adam_ea2[idx*4+j] = 0; adam_es2[idx*4+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea3[idx*3+j] = 0; adam_es3[idx*3+j] = 0; }
    for (int j = 0; j < fr_stride; j++) { adam_ea4[idx*fr_stride+j] = 0; adam_es4[idx*fr_stride+j] = 0; }
    adam_ea5[idx] = 0; adam_es5[idx] = 0;

    for (int j = 0; j < 3; j++) { adam_ea0[child*3+j] = 0; adam_es0[child*3+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea1[child*3+j] = 0; adam_es1[child*3+j] = 0; }
    for (int j = 0; j < 4; j++) { adam_ea2[child*4+j] = 0; adam_es2[child*4+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea3[child*3+j] = 0; adam_es3[child*3+j] = 0; }
    for (int j = 0; j < fr_stride; j++) { adam_ea4[child*fr_stride+j] = 0; adam_es4[child*fr_stride+j] = 0; }
    adam_ea5[child] = 0; adam_es5[child] = 0;
}

// Append duplicate copies into backing buffers. One thread per original gaussian.
// Each dup produces 1 copy at [N + nSplits + dup_ord].
kernel void densify_append_dup_kernel(
    constant int& N,
    constant int* dup_flag           [[buffer(1)]],
    constant int* dup_prefix         [[buffer(2)]],  // inclusive prefix sum
    constant int* split_prefix       [[buffer(3)]],  // to read nSplits = split_prefix[N-1]
    device float* means_buf          [[buffer(4)]],
    device float* scales_buf         [[buffer(5)]],
    device float* quats_buf          [[buffer(6)]],
    device float* featuresDc_buf     [[buffer(7)]],
    device float* featuresRest_buf   [[buffer(8)]],
    device float* opacities_buf      [[buffer(9)]],
    constant int& fr_stride          [[buffer(10)]],
    device float* adam_ea0           [[buffer(11)]],
    device float* adam_ea1           [[buffer(12)]],
    device float* adam_ea2           [[buffer(13)]],
    device float* adam_ea3           [[buffer(14)]],
    device float* adam_ea4           [[buffer(15)]],
    device float* adam_ea5           [[buffer(16)]],
    device float* adam_es0           [[buffer(17)]],
    device float* adam_es1           [[buffer(18)]],
    device float* adam_es2           [[buffer(19)]],
    device float* adam_es3           [[buffer(20)]],
    device float* adam_es4           [[buffer(21)]],
    device float* adam_es5           [[buffer(22)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= (uint)N || dup_flag[idx] == 0) return;

    int nSplits = (N > 0) ? split_prefix[N - 1] : 0;
    int ord = dup_prefix[idx] - 1;
    int dst = N + nSplits + ord;

    // Copy all parent data
    for (int j = 0; j < 3; j++) means_buf[dst*3+j] = means_buf[idx*3+j];
    for (int j = 0; j < 3; j++) scales_buf[dst*3+j] = scales_buf[idx*3+j];
    for (int j = 0; j < 4; j++) quats_buf[dst*4+j] = quats_buf[idx*4+j];
    for (int j = 0; j < 3; j++) featuresDc_buf[dst*3+j] = featuresDc_buf[idx*3+j];
    for (int j = 0; j < fr_stride; j++) featuresRest_buf[dst*fr_stride+j] = featuresRest_buf[idx*fr_stride+j];
    opacities_buf[dst] = opacities_buf[idx];

    // Zero optimizer state
    for (int j = 0; j < 3; j++) { adam_ea0[dst*3+j] = 0; adam_es0[dst*3+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea1[dst*3+j] = 0; adam_es1[dst*3+j] = 0; }
    for (int j = 0; j < 4; j++) { adam_ea2[dst*4+j] = 0; adam_es2[dst*4+j] = 0; }
    for (int j = 0; j < 3; j++) { adam_ea3[dst*3+j] = 0; adam_es3[dst*3+j] = 0; }
    for (int j = 0; j < fr_stride; j++) { adam_ea4[dst*fr_stride+j] = 0; adam_es4[dst*fr_stride+j] = 0; }
    adam_ea5[dst] = 0; adam_es5[dst] = 0;
}

// Classify each post-growth gaussian as keep or cull.
// N_old = pre-growth count. N_new = N_old + nSplits + nDups (computed from prefix sums).
// Dispatch with grid_size = worst_case (e.g. 3*N_old).
kernel void densify_cull_classify_kernel(
    constant int& N_old,
    constant int* split_prefix       [[buffer(1)]],  // [N_old] inclusive
    constant int* dup_prefix         [[buffer(2)]],  // [N_old] inclusive
    constant int* split_flag         [[buffer(3)]],  // [N_old] — marks split parents
    constant float* opacities_buf    [[buffer(4)]],
    constant float* scales_buf       [[buffer(5)]],
    constant float* max_2d_size      [[buffer(6)]],  // [N_old] only valid for idx < N_old
    constant float& cull_alpha_thresh [[buffer(7)]],
    constant float& cull_scale_thresh [[buffer(8)]],
    constant float& cull_screen_size  [[buffer(9)]],
    constant int& check_huge         [[buffer(10)]],
    constant int& check_screen       [[buffer(11)]],
    device int* keep_flag            [[buffer(12)]],
    constant int& max_new_count      [[buffer(13)]],
    constant float* means_buf        [[buffer(14)]],
    constant float* quats_buf        [[buffer(15)]],
    constant float* featuresDc_buf   [[buffer(16)]],
    constant float* featuresRest_buf [[buffer(17)]],
    constant int& fr_stride          [[buffer(18)]],
    constant float4& cull_center     [[buffer(19)]],
    constant float& cull_bounds_thresh [[buffer(20)]],
    uint idx [[thread_position_in_grid]]
) {
    int nSplits = (N_old > 0) ? split_prefix[N_old - 1] : 0;
    int nDups   = (N_old > 0) ? dup_prefix[N_old - 1] : 0;
    int N_new = N_old + nSplits + nDups;

    N_new = min(N_new, max_new_count);
    if (idx >= (uint)N_new) { keep_flag[idx] = 0; return; }

    float opacity_sigmoid = 1.0f / (1.0f + exp(-opacities_buf[idx]));
    bool cull = !isfinite(opacities_buf[idx]) || opacity_sigmoid < cull_alpha_thresh;

    float3 log_scale = read_packed_float3(scales_buf, idx);
    float3 scale = exp(log_scale);
    bool scale_bad = !isfinite(log_scale.x) || !isfinite(log_scale.y) || !isfinite(log_scale.z)
        || !isfinite(scale.x) || !isfinite(scale.y) || !isfinite(scale.z)
        || min(min(scale.x, scale.y), scale.z) < 1e-10f
        || max(max(scale.x, scale.y), scale.z) > cull_scale_thresh;
    cull = cull || scale_bad;

    float3 mean = read_packed_float3(means_buf, idx);
    bool mean_bad = !isfinite(mean.x) || !isfinite(mean.y) || !isfinite(mean.z);
    if (cull_bounds_thresh < 3.0e38f) {
        float3 delta = abs(mean - cull_center.xyz);
        mean_bad = mean_bad || max(max(delta.x, delta.y), delta.z) > cull_bounds_thresh;
    }
    cull = cull || mean_bad;

    float4 quat = read_packed_float4(quats_buf, idx);
    cull = cull || !isfinite(quat.x) || !isfinite(quat.y)
        || !isfinite(quat.z) || !isfinite(quat.w);
    cull = cull || dot(quat, quat) < MIN_QUAT_NORM_SQR;

    float3 dc = read_packed_float3(featuresDc_buf, idx);
    cull = cull || !isfinite(dc.x) || !isfinite(dc.y) || !isfinite(dc.z);
    for (int j = 0; j < fr_stride; ++j) {
        cull = cull || !isfinite(featuresRest_buf[idx * fr_stride + j]);
    }

    (void)max_2d_size;
    (void)cull_screen_size;
    (void)check_huge;
    (void)check_screen;
    (void)split_flag;

    keep_flag[idx] = cull ? 0 : 1;
}

// Scatter kept elements from src to dst at compacted positions.
// One thread per float (elem * stride + sub).
kernel void compact_scatter_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant int* keep_flag          [[buffer(3)]],
    constant uint& N                 [[buffer(4)]],
    constant uint& stride            [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= N || keep_flag[elem] == 0) return;
    int dst_elem = keep_prefix[elem] - 1;
    dst[dst_elem * stride + sub] = src[elem * stride + sub];
}

// Copy compacted data from scratch back to original buffer.
// Reads new_count from keep_prefix to determine bounds.
kernel void compact_copy_back_kernel(
    constant float* src              [[buffer(0)]],
    device float* dst                [[buffer(1)]],
    constant int* keep_prefix        [[buffer(2)]],
    constant uint& last_prefix_idx   [[buffer(3)]],  // N_new - 1 (or worst_case - 1)
    constant uint& stride            [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int new_count = keep_prefix[last_prefix_idx];
    uint elem = tid / stride;
    uint sub  = tid % stride;
    if (elem >= (uint)new_count) return;
    dst[elem * stride + sub] = src[elem * stride + sub];
}

// ============================================================================
// Zero buffer kernel — replaces PyTorch .zero_() MPS dispatches.
// Writes 0 as uint32, which is the zero bit-pattern for float32, int32, etc.
// ============================================================================
kernel void zero_buffer_kernel(
    device uint* buf           [[buffer(0)]],
    constant uint& count       [[buffer(1)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < count) buf[idx] = 0;
}

kernel void apply_refine_decay_kernel(
    constant uint& N                 [[buffer(0)]],
    device float* opacities          [[buffer(1)]],
    device float* scales             [[buffer(2)]],
    constant float& minus_opacity    [[buffer(3)]],
    constant float& log_scale_delta  [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= N) return;

    if (minus_opacity > 0.0f) {
        float alpha = 1.0f / (1.0f + exp(-opacities[idx]));
        alpha = clamp(alpha - minus_opacity, 1.0e-12f, 1.0f - 1.0e-12f);
        opacities[idx] = log(alpha / (1.0f - alpha));
    }

    if (log_scale_delta != 0.0f) {
        scales[idx * 3] += log_scale_delta;
        scales[idx * 3 + 1] += log_scale_delta;
        scales[idx * 3 + 2] += log_scale_delta;
    }
}

kernel void reset_opacity_kernel(
    constant uint& N               [[buffer(0)]],
    device float* opacities        [[buffer(1)]],
    constant float& reset_logit    [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= N) return;
    if (opacities[idx] > reset_logit) {
        opacities[idx] = reset_logit;
    }
}
