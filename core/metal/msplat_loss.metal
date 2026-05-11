#include "msplat_common.metal"

// ===== Fused Loss Kernels =====
// Combines SSIM (with 11×11 Gaussian window) + L1 into a single forward pass,
// and computes the full backward (dL/d_rendered) in a single backward pass.
// Replaces ~15 MPS dispatches (5 conv2d + elementwise SSIM formula + conv2d_backward + L1 ops).

#define SSIM_WIN 11
#define SSIM_HALF_WIN 5
#define SSIM_C1 0.0001f
#define SSIM_C2 0.0009f

constant bool fc_loss_composite_gt [[function_constant(3)]];
constant bool fc_loss_use_mask [[function_constant(4)]];
constant bool fc_loss_use_alpha_loss [[function_constant(5)]];

static inline uint loss_composite_gt(const uint runtime_value) {
    return is_function_constant_defined(fc_loss_composite_gt)
        ? (fc_loss_composite_gt ? 1u : 0u)
        : runtime_value;
}

static inline uint loss_use_mask(const uint runtime_value) {
    return is_function_constant_defined(fc_loss_use_mask)
        ? (fc_loss_use_mask ? 1u : 0u)
        : runtime_value;
}

static inline uint loss_use_alpha_loss(const uint runtime_value) {
    return is_function_constant_defined(fc_loss_use_alpha_loss)
        ? (fc_loss_use_alpha_loss ? 1u : 0u)
        : runtime_value;
}

kernel void fused_loss_forward_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant float* gt,             // (H, W, 3) HWC
    constant float* window,         // (121,) precomputed 2D Gaussian window
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    device float* intermediates,    // (H, W, 15) — 5 values × 3 channels per pixel
    device atomic_float* loss_sum,  // scalar: atomic sum of all pixel losses
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    uint px = gid.x;
    uint py = gid.y;

    float pixel_loss = 0.0f;

    if (px < W && py < H) {
        float ssim_sum = 0.0f;
        float l1_sum = 0.0f;

        // Clamp loop bounds once to avoid per-iteration boundary checks
        int dy_min = max(-SSIM_HALF_WIN, -(int)py);
        int dy_max = min(SSIM_HALF_WIN, (int)(H - 1) - (int)py);
        int dx_min = max(-SSIM_HALF_WIN, -(int)px);
        int dx_max = min(SSIM_HALF_WIN, (int)(W - 1) - (int)px);

        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;

            for (int dy = dy_min; dy <= dy_max; dy++) {
                int ny = (int)py + dy;
                for (int dx = dx_min; dx <= dx_max; dx++) {
                    int nx = (int)px + dx;
                    float w = window[(dy + SSIM_HALF_WIN) * SSIM_WIN + (dx + SSIM_HALF_WIN)];
                    float x_val = gt[(ny * W + nx) * 3 + c];
                    float y_val = rendered[(ny * W + nx) * 3 + c];
                    mu_x += w * x_val;
                    mu_y += w * y_val;
                    sq_x += w * x_val * x_val;
                    sq_y += w * y_val * y_val;
                    cross_xy += w * x_val * y_val;
                }
            }

            float sigma_x_sq = max(0.0f, sq_x - mu_x * mu_x);
            float sigma_y_sq = max(0.0f, sq_y - mu_y * mu_y);
            float sigma_xy = cross_xy - mu_x * mu_y;

            uint iidx = (py * W + px) * 15 + c * 5;
            intermediates[iidx + 0] = mu_x;
            intermediates[iidx + 1] = mu_y;
            intermediates[iidx + 2] = sigma_x_sq;
            intermediates[iidx + 3] = sigma_y_sq;
            intermediates[iidx + 4] = sigma_xy;

            float A = 2.0f * mu_x * mu_y + SSIM_C1;
            float B = 2.0f * sigma_xy + SSIM_C2;
            float C_d = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            float D = sigma_x_sq + sigma_y_sq + SSIM_C2;

            ssim_sum += (A * B) / (C_d * D);

            float gt_val = gt[(py * W + px) * 3 + c];
            float rend_val = rendered[(py * W + px) * 3 + c];
            l1_sum += fabs(gt_val - rend_val);
        }

        pixel_loss = ssim_weight * (1.0f - ssim_sum / 3.0f) + (1.0f - ssim_weight) * l1_sum / 3.0f;
    }

    // Threadgroup reduction then single atomic add to device memory
    threadgroup float tg_sum[256];
    tg_sum[tid] = pixel_loss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tg_total = tg_size.x * tg_size.y;
    for (uint s = tg_total / 2; s > 0; s >>= 1) {
        if (tid < s) tg_sum[tid] += tg_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        atomic_fetch_add_explicit(loss_sum, tg_sum[0], memory_order_relaxed);
    }
}

kernel void fused_loss_backward_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant float* gt,             // (H, W, 3) HWC
    constant float* window,         // (121,) 2D Gaussian window
    constant uint2& img_size,       // (W, H)
    constant float* intermediates,  // (H, W, 15) from forward
    constant float& ssim_weight,
    constant float& inv_n,          // 1.0 / (H * W * 3)
    device float* v_rendered,       // (H, W, 3) output gradient
    uint2 gid [[thread_position_in_grid]]
) {
    uint W = img_size.x;
    uint H = img_size.y;
    uint px = gid.x;
    uint py = gid.y;
    if (px >= W || py >= H) return;

    // Clamp loop bounds once to avoid per-iteration boundary checks
    int dy_min = max(-SSIM_HALF_WIN, (int)py - (int)(H - 1));
    int dy_max = min(SSIM_HALF_WIN, (int)py);
    int dx_min = max(-SSIM_HALF_WIN, (int)px - (int)(W - 1));
    int dx_max = min(SSIM_HALF_WIN, (int)px);

    for (uint c = 0; c < 3; c++) {
        float rend_val = rendered[(py * W + px) * 3 + c];
        float gt_val = gt[(py * W + px) * 3 + c];

        // L1 gradient: d|gt-rend|/d(rend) = -sign(gt-rend)
        float v_l1 = (gt_val > rend_val) ? -1.0f : ((gt_val < rend_val) ? 1.0f : 0.0f);

        // SSIM gradient: sum contributions from all windows containing this pixel
        float v_ssim = 0.0f;

        for (int dy = dy_min; dy <= dy_max; dy++) {
            int cy = (int)py - dy;  // center of neighboring window
            for (int dx = dx_min; dx <= dx_max; dx++) {
                int cx = (int)px - dx;

                float w = window[(dy + SSIM_HALF_WIN) * SSIM_WIN + (dx + SSIM_HALF_WIN)];

                // Read intermediates at window center (cy, cx)
                uint iidx = (cy * W + cx) * 15 + c * 5;
                float mu_x = intermediates[iidx + 0];
                float mu_y = intermediates[iidx + 1];
                float sigma_x_sq = intermediates[iidx + 2];
                float sigma_y_sq = intermediates[iidx + 3];
                float sigma_xy = intermediates[iidx + 4];

                float A = 2.0f * mu_x * mu_y + SSIM_C1;
                float B = 2.0f * sigma_xy + SSIM_C2;
                float C_d = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
                float D = sigma_x_sq + sigma_y_sq + SSIM_C2;

                // Partial derivatives of SSIM w.r.t. mu_y, sigma_y_sq, sigma_xy
                float inv_CD = 1.0f / (C_d * D);
                float dSSIM_dmu_y = 2.0f * B * (mu_x * C_d - A * mu_y) / (C_d * C_d * D);
                float dSSIM_dsigma_y_sq = -A * B * inv_CD / D;
                float dSSIM_dsigma_xy = 2.0f * A * inv_CD;

                // Chain: d(rendered[py,px])/d(mu_y) = w, etc.
                v_ssim += w * (dSSIM_dmu_y
                    + 2.0f * (rend_val - mu_y) * dSSIM_dsigma_y_sq
                    + (gt_val - mu_x) * dSSIM_dsigma_xy);
            }
        }

        // Combined: loss = ssim_weight*(1-mean(ssim)) + (1-ssim_weight)*mean(l1)
        // dL/d(rendered) = -ssim_weight * inv_n * v_ssim + (1-ssim_weight) * inv_n * v_l1
        v_rendered[(py * W + px) * 3 + c] = inv_n * (
            -ssim_weight * v_ssim + (1.0f - ssim_weight) * v_l1
        );
    }
}

// ============================================================================
// Separable SSIM loss kernels (v29)
// Decompose 11×11 2D Gaussian convolution into 1D horizontal + vertical passes.
// Reduces per-pixel work from O(121) to O(22).
// ============================================================================

// 1D Gaussian window (sigma=1.5, size=11) — matches ssim.cpp:gaussian(1.5f)
// The 2D window is the outer product: w2d[i][j] = GAUSS_1D[i] * GAUSS_1D[j]
constant float GAUSS_1D[11] = {
    0.0010283801f, 0.0075987581f, 0.0360007721f, 0.1093606895f, 0.2130055377f,
    0.2660117249f,
    0.2130055377f, 0.1093606895f, 0.0360007721f, 0.0075987581f, 0.0010283801f
};

#define SSIM_TG 16   // threadgroup dimension (16×16 = 256 threads)

// Forward pass 1: horizontal convolution of rendered and gt.
// For each pixel, computes 5 horizontal partial sums per channel:
//   h_mu_x, h_mu_y, h_sq_x, h_sq_y, h_cross_xy
// Output: ssim_h_buf (H, W, 15) — 5 values × 3 channels
kernel void ssim_h_fwd_kernel(
    constant float* rendered,       // (H, W, 3) HWC
    constant uint* gt_packed,       // (H, W) packed RGBA8
    constant uint2& img_size,       // (W, H)
    device float* ssim_h_buf,       // (H, W, 15)
    constant float* background,
    constant uint& composite_gt,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const uint composite_gt_value = loss_composite_gt(composite_gt);
    const int base_gx = (int)(tgid.x * SSIM_TG) - SSIM_HALF_WIN;
    const int base_gy = (int)(tgid.y * SSIM_TG);
    constexpr uint TILE_W = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = SSIM_TG * TILE_W;         // 416

    // Load all 3 channels at once: 6 × 16×26 × 4B = 9.75KB shared memory
    threadgroup float tg_gt[3][SSIM_TG][TILE_W];
    threadgroup float tg_rd[3][SSIM_TG][TILE_W];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / TILE_W;
            uint sx = i % TILE_W;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float gv = 0.0f, rv = 0.0f;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint pixel = gy * W + gx;
                uint idx = pixel * 3 + c;
                gv = packed_gt_effective(gt_packed, pixel, c, background, composite_gt_value);
                rv = rendered[idx];
            }
            tg_gt[c][sy][sx] = gv;
            tg_rd[c][sy][sx] = rv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;
            for (uint dx = 0; dx < SSIM_WIN; dx++) {
                float w = GAUSS_1D[dx];
                float gv = tg_gt[c][lid.y][lid.x + dx];
                float rv = tg_rd[c][lid.y][lid.x + dx];
                mu_x += w * gv;
                mu_y += w * rv;
                sq_x += w * gv * gv;
                sq_y += w * rv * rv;
                cross_xy += w * gv * rv;
            }
            uint out = (py * W + px) * 15 + c * 5;
            ssim_h_buf[out + 0] = mu_x;
            ssim_h_buf[out + 1] = mu_y;
            ssim_h_buf[out + 2] = sq_x;
            ssim_h_buf[out + 3] = sq_y;
            ssim_h_buf[out + 4] = cross_xy;
        }
    }
}

kernel void l1_loss_fwd_bwd_kernel(
    constant float* rendered,
    constant uint* gt_packed,
    constant uint2& img_size,
    constant float& inv_n,
    device float* v_rendered,
    device atomic_float* loss_sum,
    constant float* background,
    constant uint& composite_gt,
    constant uint& use_loss_mask,
    constant float* final_Ts,
    constant uint& use_alpha_loss,
    constant float& alpha_loss_weight,
    uint2 gid [[thread_position_in_grid]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const uint composite_gt_value = loss_composite_gt(composite_gt);
    const uint use_loss_mask_value = loss_use_mask(use_loss_mask);
    const uint use_alpha_loss_value = loss_use_alpha_loss(use_alpha_loss);
    float pixel_loss = 0.0f;

    if (px < W && py < H) {
        const uint pixel = py * W + px;
        const float gt_alpha = packed_gt_alpha(gt_packed, pixel);
        const float mask_weight = use_loss_mask_value != 0 ? gt_alpha : 1.0f;
        float l1_sum = 0.0f;
        for (uint c = 0; c < 3; c++) {
            const uint idx = pixel * 3 + c;
            const float gt_val = packed_gt_effective(gt_packed, pixel, c, background, composite_gt_value);
            const float rend_val = rendered[idx];
            l1_sum += fabs(gt_val - rend_val);
            const float v_l1 = (gt_val > rend_val) ? -1.0f : ((gt_val < rend_val) ? 1.0f : 0.0f);
            v_rendered[idx] = mask_weight * inv_n * v_l1;
        }
        pixel_loss = mask_weight * l1_sum / 3.0f;
        if (use_alpha_loss_value != 0) {
            pixel_loss += alpha_loss_weight * fabs(gt_alpha - (1.0f - final_Ts[pixel]));
        }
    }

    threadgroup float sg_sums[32];
    float sg_sum = simd_sum(pixel_loss);
    if (sg_lane == 0) {
        sg_sums[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = tg_size.x * tg_size.y;
    uint num_sg = (tg_total + sg_size - 1) / sg_size;
    if (sg_id == 0) {
        float total = (sg_lane < num_sg) ? sg_sums[sg_lane] : 0.0f;
        total = simd_sum(total);
        if (sg_lane == 0) {
            atomic_fetch_add_explicit(loss_sum, total, memory_order_relaxed);
        }
    }
}

// Forward pass 2: vertical convolution + SSIM/L1 computation + loss reduction.
// Reads ssim_h_buf, processes 1 channel at a time for occupancy.
// Output: intermediates (H, W, 15) — same format as fused_loss_forward_kernel
kernel void ssim_v_fwd_kernel(
    constant float* rendered,       // (H, W, 3) for L1
    constant uint* gt_packed,       // (H, W) packed RGBA8
    constant float* ssim_h_buf,     // (H, W, 15)
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    device float* intermediates,    // (H, W, 15)
    device atomic_float* loss_sum,
    constant float* background,
    constant uint& composite_gt,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const uint composite_gt_value = loss_composite_gt(composite_gt);
    const int base_gx = (int)(tgid.x * SSIM_TG);
    const int base_gy = (int)(tgid.y * SSIM_TG) - SSIM_HALF_WIN;
    constexpr uint TILE_H = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = TILE_H * SSIM_TG;         // 416

    // Load all 3 channels at once: 3 × 26×16×5 × 4B = 24.96KB shared memory
    threadgroup float tg_hp[3][TILE_H][SSIM_TG][5];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / SSIM_TG;
            uint sx = i % SSIM_TG;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint hp = (gy * W + gx) * 15 + c * 5;
                for (uint f = 0; f < 5; f++) tg_hp[c][sy][sx][f] = ssim_h_buf[hp + f];
            } else {
                for (uint f = 0; f < 5; f++) tg_hp[c][sy][sx][f] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ssim_sum = 0.0f;
    float l1_sum = 0.0f;

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross_xy = 0;
            for (uint dy = 0; dy < SSIM_WIN; dy++) {
                float w = GAUSS_1D[dy];
                mu_x     += w * tg_hp[c][lid.y + dy][lid.x][0];
                mu_y     += w * tg_hp[c][lid.y + dy][lid.x][1];
                sq_x     += w * tg_hp[c][lid.y + dy][lid.x][2];
                sq_y     += w * tg_hp[c][lid.y + dy][lid.x][3];
                cross_xy += w * tg_hp[c][lid.y + dy][lid.x][4];
            }

            float sigma_x_sq = max(0.0f, sq_x - mu_x * mu_x);
            float sigma_y_sq = max(0.0f, sq_y - mu_y * mu_y);
            float sigma_xy = cross_xy - mu_x * mu_y;

            // Store intermediates (same format as fused_loss_forward_kernel)
            uint iidx = (py * W + px) * 15 + c * 5;
            intermediates[iidx + 0] = mu_x;
            intermediates[iidx + 1] = mu_y;
            intermediates[iidx + 2] = sigma_x_sq;
            intermediates[iidx + 3] = sigma_y_sq;
            intermediates[iidx + 4] = sigma_xy;

            // SSIM for this channel
            float A  = 2.0f * mu_x * mu_y + SSIM_C1;
            float B  = 2.0f * sigma_xy + SSIM_C2;
            float Cd = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            float D  = sigma_x_sq + sigma_y_sq + SSIM_C2;
            float raw_ssim = (A * B) / (Cd * D);
            ssim_sum += clamp(raw_ssim, -1.0f, 1.0f);

            // L1 for this channel
            float gt_v  = packed_gt_effective(gt_packed, py * W + px, c, background, composite_gt_value);
            float rd_v  = rendered[(py * W + px) * 3 + c];
            l1_sum += fabs(gt_v - rd_v);
        }
    }

    // Pixel loss
    float pixel_loss = 0.0f;
    if (px < W && py < H) {
        pixel_loss = ssim_weight * (1.0f - ssim_sum / 3.0f) + (1.0f - ssim_weight) * l1_sum / 3.0f;
    }

    // Threadgroup reduction → atomic add to loss_sum
    threadgroup float sg_sums[32];
    float sg_sum = simd_sum(pixel_loss);
    if (sg_lane == 0) {
        sg_sums[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = tg_size.x * tg_size.y;
    uint num_sg = (tg_total + sg_size - 1) / sg_size;
    if (sg_id == 0) {
        float total = (sg_lane < num_sg) ? sg_sums[sg_lane] : 0.0f;
        total = simd_sum(total);
        if (sg_lane == 0) {
            atomic_fetch_add_explicit(loss_sum, total, memory_order_relaxed);
        }
    }
}

// Fused V-forward + H-backward: recomputes SSIM stats from ssim_h_buf (V conv),
// computes loss + derivative fields, then H convs derivatives to output buffer.
// Eliminates loss_intermediates round-trip (130 MB/iter bandwidth saved).
kernel void ssim_fused_v_fwd_h_bwd_kernel(
    constant float* rendered, constant uint* gt_packed,
    constant float* ssim_h_buf, constant uint2& img_size,
    constant float& ssim_weight, constant float& inv_n,
    device float* deriv_h_buf, device atomic_float* loss_sum,
    constant float* background, constant uint& composite_gt,
    constant uint& use_loss_mask,
    constant float* final_Ts,
    constant uint& use_alpha_loss, constant float& alpha_loss_weight,
    uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    const uint W = img_size.x, H = img_size.y;
    const uint px = gid.x, py = gid.y;
    const uint composite_gt_value = loss_composite_gt(composite_gt);
    const uint use_loss_mask_value = loss_use_mask(use_loss_mask);
    const uint use_alpha_loss_value = loss_use_alpha_loss(use_alpha_loss);
    const int base_gx = (int)(tgid.x * SSIM_TG) - SSIM_HALF_WIN;
    const int base_gy = (int)(tgid.y * SSIM_TG) - SSIM_HALF_WIN;
    constexpr uint TILE_DIM = SSIM_TG + 2 * SSIM_HALF_WIN;
    constexpr uint TILE_PIXELS = TILE_DIM * TILE_DIM;
    float loss_accum = 0.0f;

    for (uint c = 0; c < 3; c++) {
        threadgroup float tg_hp[TILE_DIM][TILE_DIM][5];
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / TILE_DIM, sx = i % TILE_DIM;
            int gy = base_gy + (int)sy, gx = base_gx + (int)sx;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint hp = (gy * W + gx) * 15 + c * 5;
                for (uint f = 0; f < 5; f++) tg_hp[sy][sx][f] = ssim_h_buf[hp + f];
            } else {
                for (uint f = 0; f < 5; f++) tg_hp[sy][sx][f] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float tg_f1[SSIM_TG][TILE_DIM], tg_f2[SSIM_TG][TILE_DIM], tg_f3[SSIM_TG][TILE_DIM];
        constexpr uint DERIV_PIXELS = SSIM_TG * TILE_DIM;
        for (uint i = tr; i < DERIV_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint dy = i / TILE_DIM, dx = i % TILE_DIM;
            uint tile_y = dy + SSIM_HALF_WIN;
            float mu_x=0, mu_y=0, sq_x=0, sq_y=0, cross_xy=0;
            for (uint k = 0; k < SSIM_WIN; k++) {
                float w = GAUSS_1D[k];
                mu_x += w*tg_hp[tile_y-SSIM_HALF_WIN+k][dx][0];
                mu_y += w*tg_hp[tile_y-SSIM_HALF_WIN+k][dx][1];
                sq_x += w*tg_hp[tile_y-SSIM_HALF_WIN+k][dx][2];
                sq_y += w*tg_hp[tile_y-SSIM_HALF_WIN+k][dx][3];
                cross_xy += w*tg_hp[tile_y-SSIM_HALF_WIN+k][dx][4];
            }
            float sigma_x_sq = max(0.0f, sq_x - mu_x*mu_x);
            float sigma_y_sq = max(0.0f, sq_y - mu_y*mu_y);
            float sigma_xy = cross_xy - mu_x*mu_y;
            float A = 2.0f*mu_x*mu_y + SSIM_C1, B = 2.0f*sigma_xy + SSIM_C2;
            float Cd = mu_x*mu_x + mu_y*mu_y + SSIM_C1, D = sigma_x_sq + sigma_y_sq + SSIM_C2;
            float raw_ssim = (A * B) / (Cd * D);
            bool ssim_clamped = raw_ssim < -1.0f || raw_ssim > 1.0f;
            float iCD = 1.0f / (Cd * D);
            float dmu = 2.0f*B*(mu_x*Cd - A*mu_y) / (Cd*Cd*D);
            float dsyq = -A*B*iCD/D, dsxy = 2.0f*A*iCD;
            int gpx = base_gx + (int)dx;
            int gpy = base_gy + (int)(dy + SSIM_HALF_WIN);
            bool center_valid = gpx >= 0 && gpx < (int)W && gpy >= 0 && gpy < (int)H;
            uint center_pixel = (uint)gpy * W + (uint)gpx;
            float center_mask = center_valid
                ? ((use_loss_mask_value != 0) ? packed_gt_alpha(gt_packed, center_pixel) : 1.0f)
                : 0.0f;
            float deriv_scale = ssim_clamped ? 0.0f : center_mask;
            tg_f1[dy][dx] = deriv_scale * (dmu - 2.0f*mu_y*dsyq - mu_x*dsxy);
            tg_f2[dy][dx] = deriv_scale * (2.0f*dsyq);
            tg_f3[dy][dx] = deriv_scale * dsxy;
            if (dx >= SSIM_HALF_WIN && dx < SSIM_HALF_WIN + SSIM_TG) {
                if (center_valid) {
                    float gt_val = packed_gt_effective(
                        gt_packed, center_pixel, c, background, composite_gt_value);
                    float l1 = fabs(gt_val - rendered[(gpy*W+gpx)*3+c]);
                    loss_accum += center_mask * (
                        (c == 0 ? ssim_weight : 0.0f)
                        + ((1.0f - ssim_weight) * l1 - ssim_weight * clamp(raw_ssim, -1.0f, 1.0f)) / 3.0f
                    );
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (px < W && py < H) {
            float h1=0, h2=0, h3=0;
            for (uint dx = 0; dx < SSIM_WIN; dx++) {
                float w = GAUSS_1D[SSIM_WIN-1-dx];
                h1 += w*tg_f1[lid.y][lid.x+dx]; h2 += w*tg_f2[lid.y][lid.x+dx]; h3 += w*tg_f3[lid.y][lid.x+dx];
            }
            uint out = (py*W+px)*15 + c*5;
            deriv_h_buf[out+0]=h1; deriv_h_buf[out+1]=h2; deriv_h_buf[out+2]=h3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (use_alpha_loss_value != 0 && px < W && py < H) {
        uint pixel = py * W + px;
        loss_accum += alpha_loss_weight * fabs(packed_gt_alpha(gt_packed, pixel) - (1.0f - final_Ts[pixel]));
    }

    threadgroup float sg_sums[32];
    float sg_sum = simd_sum(loss_accum);
    if (sg_lane == 0) {
        sg_sums[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tg_total = tg_size.x * tg_size.y;
    uint num_sg = (tg_total + sg_size - 1) / sg_size;
    if (sg_id == 0) {
        float total = (sg_lane < num_sg) ? sg_sums[sg_lane] : 0.0f;
        total = simd_sum(total);
        if (sg_lane == 0) {
            atomic_fetch_add_explicit(loss_sum, total, memory_order_relaxed);
        }
    }
}

// Backward pass 1: compute derivative fields + horizontal convolution.
// For each pixel, computes F1, F2, F3 from intermediates, then convolves horizontally.
// Output: ssim_h_buf (H, W, 15) — 3 values per channel at stride 5
kernel void ssim_h_bwd_kernel(
    constant float* intermediates,  // (H, W, 15) from forward
    constant uint2& img_size,       // (W, H)
    device float* ssim_h_buf,       // (H, W, 15) — reused from forward
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const int base_gx = (int)(tgid.x * SSIM_TG) - SSIM_HALF_WIN;
    const int base_gy = (int)(tgid.y * SSIM_TG);
    constexpr uint TILE_W = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = SSIM_TG * TILE_W;         // 416

    // Compute derivative fields for all 3 channels: 9 × 16×26 × 4B = 14.6KB
    threadgroup float tg_f1[3][SSIM_TG][TILE_W];
    threadgroup float tg_f2[3][SSIM_TG][TILE_W];
    threadgroup float tg_f3[3][SSIM_TG][TILE_W];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / TILE_W;
            uint sx = i % TILE_W;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float f1 = 0, f2 = 0, f3 = 0;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint ii = (gy * W + gx) * 15 + c * 5;
                float mu_x  = intermediates[ii + 0];
                float mu_y  = intermediates[ii + 1];
                float sx_sq = max(0.0f, intermediates[ii + 2]);
                float sy_sq = max(0.0f, intermediates[ii + 3]);
                float sxy   = intermediates[ii + 4];

                float A   = 2.0f * mu_x * mu_y + SSIM_C1;
                float B   = 2.0f * sxy + SSIM_C2;
                float Cd  = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
                float D   = sx_sq + sy_sq + SSIM_C2;
                float raw_ssim = (A * B) / (Cd * D);
                bool ssim_clamped = raw_ssim < -1.0f || raw_ssim > 1.0f;
                float iCD = 1.0f / (Cd * D);

                float dmu  = 2.0f * B * (mu_x * Cd - A * mu_y) / (Cd * Cd * D);
                float dsyq = -A * B * iCD / D;
                float dsxy = 2.0f * A * iCD;

                if (!ssim_clamped) {
                    f1 = dmu - 2.0f * mu_y * dsyq - mu_x * dsxy;
                    f2 = 2.0f * dsyq;
                    f3 = dsxy;
                }
            }
            tg_f1[c][sy][sx] = f1;
            tg_f2[c][sy][sx] = f2;
            tg_f3[c][sy][sx] = f3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        for (uint c = 0; c < 3; c++) {
            float h1 = 0, h2 = 0, h3 = 0;
            for (uint dx = 0; dx < SSIM_WIN; dx++) {
                float w = GAUSS_1D[SSIM_WIN - 1 - dx];
                h1 += w * tg_f1[c][lid.y][lid.x + dx];
                h2 += w * tg_f2[c][lid.y][lid.x + dx];
                h3 += w * tg_f3[c][lid.y][lid.x + dx];
            }
            uint out = (py * W + px) * 15 + c * 5;
            ssim_h_buf[out + 0] = h1;
            ssim_h_buf[out + 1] = h2;
            ssim_h_buf[out + 2] = h3;
        }
    }
}

// Backward pass 2: vertical convolution + combine to produce v_rendered.
// Output: v_rendered (H, W, 3)
kernel void ssim_v_bwd_kernel(
    constant float* rendered,       // (H, W, 3)
    constant uint* gt_packed,       // (H, W) packed RGBA8
    constant float* ssim_h_buf,     // (H, W, 15)
    constant uint2& img_size,       // (W, H)
    constant float& ssim_weight,
    constant float& inv_n,          // 1.0 / (H * W * 3)
    device float* v_rendered,       // (H, W, 3)
    constant float* background,
    constant uint& composite_gt,
    constant uint& use_loss_mask,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tr [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint px = gid.x;
    const uint py = gid.y;
    const uint composite_gt_value = loss_composite_gt(composite_gt);
    const uint use_loss_mask_value = loss_use_mask(use_loss_mask);
    const int base_gx = (int)(tgid.x * SSIM_TG);
    const int base_gy = (int)(tgid.y * SSIM_TG) - SSIM_HALF_WIN;
    constexpr uint TILE_H = SSIM_TG + 2 * SSIM_HALF_WIN;  // 26
    constexpr uint TILE_PIXELS = TILE_H * SSIM_TG;         // 416

    // Load all 3 channels at once: 9 × 26×16 × 4B = 14.6KB
    threadgroup float tg_h1[3][TILE_H][SSIM_TG];
    threadgroup float tg_h2[3][TILE_H][SSIM_TG];
    threadgroup float tg_h3[3][TILE_H][SSIM_TG];

    for (uint c = 0; c < 3; c++) {
        for (uint i = tr; i < TILE_PIXELS; i += SSIM_TG * SSIM_TG) {
            uint sy = i / SSIM_TG;
            uint sx = i % SSIM_TG;
            int gy = base_gy + (int)sy;
            int gx = base_gx + (int)sx;
            float v1 = 0, v2 = 0, v3 = 0;
            if (gx >= 0 && gx < (int)W && gy >= 0 && gy < (int)H) {
                uint hp = (gy * W + gx) * 15 + c * 5;
                v1 = ssim_h_buf[hp + 0];
                v2 = ssim_h_buf[hp + 1];
                v3 = ssim_h_buf[hp + 2];
            }
            tg_h1[c][sy][sx] = v1;
            tg_h2[c][sy][sx] = v2;
            tg_h3[c][sy][sx] = v3;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (px < W && py < H) {
        uint pixel = py * W + px;
        float l1_mask_weight = use_loss_mask_value != 0 ? packed_gt_alpha(gt_packed, pixel) : 1.0f;
        for (uint c = 0; c < 3; c++) {
            float conv_f1 = 0, conv_f2 = 0, conv_f3 = 0;
            for (uint dy = 0; dy < SSIM_WIN; dy++) {
                float w = GAUSS_1D[SSIM_WIN - 1 - dy];
                conv_f1 += w * tg_h1[c][lid.y + dy][lid.x];
                conv_f2 += w * tg_h2[c][lid.y + dy][lid.x];
                conv_f3 += w * tg_h3[c][lid.y + dy][lid.x];
            }

            float rend_val = rendered[(py * W + px) * 3 + c];
            float gt_val = packed_gt_effective(gt_packed, pixel, c, background, composite_gt_value);

            float v_ssim = conv_f1 + rend_val * conv_f2 + gt_val * conv_f3;
            float v_l1 = (gt_val > rend_val) ? -1.0f : ((gt_val < rend_val) ? 1.0f : 0.0f);

            v_rendered[(py * W + px) * 3 + c] = inv_n * (
                -ssim_weight * v_ssim + (1.0f - ssim_weight) * l1_mask_weight * v_l1
            );
        }
    }
}

kernel void lpips_prepare_nchw_kernel(
    constant float* rendered [[buffer(0)]],
    constant uint* gt_packed [[buffer(1)]],
    constant uint2& img_size [[buffer(2)]],
    device float* rendered_nchw [[buffer(3)]],
    device float* gt_nchw [[buffer(4)]],
    constant float* background [[buffer(5)]],
    constant uint& composite_gt [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint pixels = W * H;
    if (idx >= pixels * 3) return;

    const uint c = idx / pixels;
    const uint p = idx - c * pixels;
    const uint y = p / W;
    const uint x = p - y * W;
    const uint hwc = (y * W + x) * 3 + c;
    rendered_nchw[idx] = rendered[hwc];
    gt_nchw[idx] = packed_gt_effective(gt_packed, p, c, background, composite_gt);
}

kernel void lpips_apply_grad_kernel(
    constant float* grad_nchw [[buffer(0)]],
    constant float* lpips_loss [[buffer(1)]],
    constant uint2& img_size [[buffer(2)]],
    constant float& lpips_weight [[buffer(3)]],
    device float* v_rendered [[buffer(4)]],
    device atomic_float* loss_sum [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    const uint W = img_size.x;
    const uint H = img_size.y;
    const uint pixels = W * H;
    if (idx == 0) {
        atomic_fetch_add_explicit(loss_sum, lpips_weight * lpips_loss[0] * (float)pixels,
                                  memory_order_relaxed);
    }
    if (idx >= pixels * 3) return;

    const uint c = idx / pixels;
    const uint p = idx - c * pixels;
    const uint y = p / W;
    const uint x = p - y * W;
    const uint hwc = (y * W + x) * 3 + c;
    v_rendered[hwc] += lpips_weight * grad_nchw[idx];
}
