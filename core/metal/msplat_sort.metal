#include "msplat_common.metal"

// ===== Pack Sorted Gaussians Kernel =====

kernel void pack_sorted_gaussians_kernel(
    constant int32_t* gaussian_ids_sorted [[buffer(0)]],
    constant float* xys              [[buffer(1)]],
    constant float* conics           [[buffer(2)]],
    constant float* colors           [[buffer(3)]],
    constant float* opacities        [[buffer(4)]],
    device float* packed_xy_opac     [[buffer(5)]],
    device float* packed_conic       [[buffer(6)]],
    device float* packed_rgb         [[buffer(7)]],
    constant float* opacity_comp     [[buffer(8)]],
    device float* packed_opacity_comp [[buffer(9)]],
    constant uint& N                 [[buffer(10)]],
    constant int32_t* cum_tiles_hit  [[buffer(11)]],
    constant uint& num_points        [[buffer(12)]],
    device half* packed_conic_half   [[buffer(13)]],
    device half* packed_rgb_half     [[buffer(14)]],
    device half* packed_opacity_comp_half [[buffer(15)]],
    constant uint& use_half_sorted_buffers [[buffer(16)]],
    uint idx [[thread_position_in_grid]]
) {
    uint actual_N = min(N, (uint)cum_tiles_hit[num_points - 1]);
    if (idx >= actual_N) return;
    int32_t g_id = gaussian_ids_sorted[idx];
    float2 xy = read_packed_float2(xys, g_id);
    float opac = 1.f / (1.f + exp(-opacities[g_id]));
    float3 conic = read_packed_float3(conics, g_id);
    float3 rgb = read_packed_float3(colors, g_id);
    write_packed_float3(packed_xy_opac, idx, {xy.x, xy.y, opac});
    write_packed_sorted_float3(
        packed_conic, packed_conic_half, idx, conic, use_half_sorted_buffers);
    write_packed_sorted_float3(
        packed_rgb, packed_rgb_half, idx, rgb, use_half_sorted_buffers);
    write_packed_sorted_float(
        packed_opacity_comp, packed_opacity_comp_half, idx,
        opacity_comp[g_id], use_half_sorted_buffers);
}

// ===== Tile-Local Sorting Kernels =====
// Pre-allocated per-tile bins: scatter directly to fixed-size bins, then sort in-place.
// Eliminates count→prefix_sum→scatter pipeline (3 dispatches + 3 barriers saved).

#define SORT_TG_SIZE 256
#define MAX_TILE_ELEMS 4096

// Scatter each gaussian's intersections directly into pre-allocated per-tile bins.
// Each tile gets MAX_TILE_ELEMS slots. Per-tile atomics track fill count.
kernel void scatter_to_prealloc_bins_kernel(
    constant uint& num_points               [[buffer(0)]],
    constant float* xys                     [[buffer(1)]],
    constant float* depths                  [[buffer(2)]],
    constant int* radii                     [[buffer(3)]],
    constant float* aabb                    [[buffer(4)]],
    constant uint3& tile_bounds             [[buffer(5)]],
    device atomic_uint* scatter_counters    [[buffer(6)]],
    device uint64_t* prealloc_bins          [[buffer(7)]],
    device atomic_uint* overflow_flag       [[buffer(8)]],
    constant float* conics                  [[buffer(9)]],
    constant float* opacities               [[buffer(10)]],
    constant float* opacity_comp            [[buffer(11)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= num_points) return;
    if (radii[idx] <= 0) return;

    float2 center = read_packed_float2(xys, idx);
    float3 conic = read_packed_float3(conics, idx);
    float opacity = (1.0f / (1.0f + exp(-opacities[idx]))) * opacity_comp[idx];
    if (!isfinite(opacity) || opacity < (1.0f / 255.0f)) return;
    float power_threshold = log(255.0f * opacity);

    uint2 tile_min, tile_max;
    get_tile_bbox(center, read_packed_float2(aabb, idx), (int3)tile_bounds, tile_min, tile_max);

    uint depth_bits = as_type<uint>(depths[idx]);

    for (uint i = tile_min.y; i < tile_max.y; i++) {
        for (uint j = tile_min.x; j < tile_max.x; j++) {
            if (!will_primitive_contribute(tile_rect(uint2(j, i)), center, conic, power_threshold)) {
                continue;
            }
            uint tile_id = i * tile_bounds.x + j;
            uint pos = atomic_fetch_add_explicit(&scatter_counters[tile_id], 1u, memory_order_relaxed);
            if (pos >= MAX_TILE_ELEMS) {
                // Clamp counter so prefix_sum sees at most MAX_TILE_ELEMS
                atomic_store_explicit(&scatter_counters[tile_id], MAX_TILE_ELEMS, memory_order_relaxed);
                atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
                continue;
            }
            prealloc_bins[(uint64_t)tile_id * MAX_TILE_ELEMS + pos] = ((uint64_t)depth_bits << 32) | (uint64_t)idx;
        }
    }
}

// Bitonic sort per tile in shared memory. Reads from pre-allocated bins.
// Writes sorted packed data to contiguous output (using tile_offsets from prefix sum).
// Also writes tile_bins for the rasterizer.
kernel void bitonic_sort_per_tile_kernel(
    constant int* tile_offsets          [[buffer(0)]],
    constant int* tile_counts_in        [[buffer(1)]],
    constant uint64_t* prealloc_bins    [[buffer(2)]],
    device int32_t* gaussian_ids_out    [[buffer(3)]],
    constant uint& num_tiles            [[buffer(4)]],
    // Pack buffers (fused sort+pack: eliminates separate pack dispatch)
    constant float* xys                 [[buffer(5)]],
    constant float* conics              [[buffer(6)]],
    constant float* colors              [[buffer(7)]],
    constant float* opacities           [[buffer(8)]],
    constant float* opacity_comp        [[buffer(9)]],
    device float* packed_xy_opac        [[buffer(10)]],
    device float* packed_conic          [[buffer(11)]],
    device float* packed_rgb            [[buffer(12)]],
    device float* packed_opacity_comp   [[buffer(13)]],
    device int* tile_bins               [[buffer(14)]],
    constant uint& pack_capacity        [[buffer(15)]],
    device atomic_uint* overflow_flag   [[buffer(16)]],
    device half* packed_conic_half      [[buffer(17)]],
    device half* packed_rgb_half        [[buffer(18)]],
    device half* packed_opacity_comp_half [[buffer(19)]],
    constant uint& use_half_sorted_buffers [[buffer(20)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tg_id >= num_tiles) return;

    int count_raw = tile_counts_in[tg_id];
    int count = min(count_raw, MAX_TILE_ELEMS);
    int end = tile_offsets[tg_id];
    int start = end - count;
    int capacity = (int)pack_capacity;
    int clamped_start = min(max(start, 0), capacity);
    int clamped_end = min(max(end, 0), capacity);

    // Write tile_bins for rasterizer
    if (tid == 0) {
        if (end > capacity) {
            atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
        }
        write_packed_int2(tile_bins, tg_id, int2(clamped_start, clamped_end));
    }

    if (count == 0 || start >= capacity || end <= 0) return;

    // Round up to next power of 2
    int n = 1;
    while (n < count) n <<= 1;

    threadgroup uint64_t data[MAX_TILE_ELEMS];

    // Load from pre-allocated bins
    uint64_t bin_base = (uint64_t)tg_id * MAX_TILE_ELEMS;
    for (int i = (int)tid; i < n; i += SORT_TG_SIZE) {
        data[i] = (i < count) ? prealloc_bins[bin_base + i] : 0xFFFFFFFFFFFFFFFFULL;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort (ascending on uint64 — depth in upper 32 bits)
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int)tid; i < (n >> 1); i += SORT_TG_SIZE) {
                int pos = 2 * i - (i & (j - 1));
                int partner = pos ^ j;
                bool ascending = ((pos & k) == 0);
                uint64_t a = data[pos];
                uint64_t b = data[partner];
                if ((a > b) == ascending) {
                    data[pos] = b;
                    data[partner] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Fused sort+pack: extract gaussian IDs, read per-gaussian data, write packed buffers
    for (int i = (int)tid; i < count; i += SORT_TG_SIZE) {
        int32_t g_id = (int32_t)(data[i] & 0xFFFFFFFF);
        int global_idx = start + i;
        if (global_idx < 0 || global_idx >= capacity) continue;
        gaussian_ids_out[global_idx] = g_id;
        float2 xy = read_packed_float2(xys, g_id);
        float opac = 1.f / (1.f + exp(-opacities[g_id]));
        write_packed_float3(packed_xy_opac, global_idx, {xy.x, xy.y, opac});
        write_packed_sorted_float3(
            packed_conic, packed_conic_half, global_idx,
            read_packed_float3(conics, g_id), use_half_sorted_buffers);
        write_packed_sorted_float3(
            packed_rgb, packed_rgb_half, global_idx,
            read_packed_float3(colors, g_id), use_half_sorted_buffers);
        write_packed_sorted_float(
            packed_opacity_comp, packed_opacity_comp_half, global_idx,
            opacity_comp[g_id], use_half_sorted_buffers);
    }
}

// ===== Dynamic Radix Sort Kernels =====
// 8-bit LSB radix sort for uint64/uint32 keys + int32 values.
// 3 kernels per pass: histogram, scan, scatter.
// TG_SIZE = 256 = RS_RADIX (one element per thread, one histogram bin per thread).

#define RS_RADIX 256
#define RS_TG_SIZE 256

kernel void radix_sort_histogram_kernel(
    constant uint& capacity        [[buffer(0)]],
    device const uint64_t* keys_in [[buffer(1)]],
    device uint* counts            [[buffer(2)]],
    constant uint& shift           [[buffer(3)]],
    device const int32_t* cum_tiles_hit [[buffer(4)]],
    constant uint& num_points      [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    // Read actual element count from GPU-resident prefix sum
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);

    // Early exit for threadgroups entirely beyond N — write zero and return
    if (bid * RS_TG_SIZE >= N) {
        counts[bid * RS_RADIX + tid] = 0;
        return;
    }

    threadgroup atomic_uint local_hist[RS_RADIX];

    // Each thread zeros one histogram bin
    atomic_store_explicit(&local_hist[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count digit for this thread's element
    uint global_idx = bid * RS_TG_SIZE + tid;
    if (global_idx < N) {
        uint digit = extract_bits(keys_in[global_idx], shift, 8);
        atomic_fetch_add_explicit(&local_hist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write histogram to global memory (row-major: counts[block_id * 256 + digit])
    counts[bid * RS_RADIX + tid] = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
}

kernel void radix_sort_histogram_u32_kernel(
    constant uint& capacity        [[buffer(0)]],
    device const uint* keys_in     [[buffer(1)]],
    device uint* counts            [[buffer(2)]],
    constant uint& shift           [[buffer(3)]],
    device const int32_t* cum_tiles_hit [[buffer(4)]],
    constant uint& num_points      [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);

    if (bid * RS_TG_SIZE >= N) {
        counts[bid * RS_RADIX + tid] = 0;
        return;
    }

    threadgroup atomic_uint local_hist[RS_RADIX];
    atomic_store_explicit(&local_hist[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint global_idx = bid * RS_TG_SIZE + tid;
    if (global_idx < N) {
        uint digit = extract_bits((uint64_t)keys_in[global_idx], shift, 8);
        atomic_fetch_add_explicit(&local_hist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    counts[bid * RS_RADIX + tid] = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
}

kernel void radix_sort_scan_kernel(
    device uint* counts                     [[buffer(0)]],
    constant uint& num_blocks               [[buffer(1)]],
    device const int32_t* cum_tiles_hit     [[buffer(2)]],
    constant uint& capacity                 [[buffer(3)]],
    constant uint& num_points               [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // Read actual element count and compute actual block count
    uint actual_N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    uint actual_num_blocks = (actual_N + RS_TG_SIZE - 1) / RS_TG_SIZE;

    // Each thread handles one digit value (tid = digit d)
    uint d = tid;

    // Phase 1: Exclusive prefix sum across actual blocks only
    uint running = 0;
    for (uint b = 0; b < actual_num_blocks; b++) {
        uint idx = b * RS_RADIX + d;
        uint count = counts[idx];
        counts[idx] = running;
        running += count;
    }

    // Phase 2: Cross-digit prefix sum to get global offsets
    threadgroup uint digit_totals[RS_RADIX];
    digit_totals[d] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 does sequential exclusive prefix sum of 256 digit totals
    if (d == 0) {
        uint accum = 0;
        for (uint i = 0; i < RS_RADIX; i++) {
            uint t = digit_totals[i];
            digit_totals[i] = accum;
            accum += t;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Add global digit offset to actual blocks only
    uint global_offset = digit_totals[d];
    for (uint b = 0; b < actual_num_blocks; b++) {
        counts[b * RS_RADIX + d] += global_offset;
    }
}

kernel void radix_sort_scatter_u32_kernel(
    constant uint& capacity            [[buffer(0)]],
    device const uint* keys_in         [[buffer(1)]],
    device const int32_t* vals_in      [[buffer(2)]],
    device uint* keys_out              [[buffer(3)]],
    device int32_t* vals_out           [[buffer(4)]],
    device const uint* counts          [[buffer(5)]],
    constant uint& shift               [[buffer(6)]],
    device const int32_t* cum_tiles_hit [[buffer(7)]],
    constant uint& num_points          [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);
    if (bid * RS_TG_SIZE >= N) return;

    threadgroup uchar shared_digits[RS_TG_SIZE];
    uint global_idx = bid * RS_TG_SIZE + tid;

    uint my_key = 0;
    int32_t my_val = 0;
    uint my_digit = 0;
    bool valid = (global_idx < N);

    if (valid) {
        my_key = keys_in[global_idx];
        my_val = vals_in[global_idx];
        my_digit = extract_bits((uint64_t)my_key, shift, 8);
    }

    shared_digits[tid] = (uchar)my_digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_rank = 0;
    for (ushort l = 0; l < 32; l++) {
        uint other = simd_broadcast(my_digit, l);
        if (l < sg_lane && other == my_digit) sg_rank++;
    }

    uint cross_rank = 0;
    uint preceding_end = sg_id * 32;
    uint full_words = preceding_end / 4;
    threadgroup uint* digits_u32 = (threadgroup uint*)shared_digits;
    for (uint i = 0; i < full_words; i++) {
        uint four = digits_u32[i];
        if (((four >>  0) & 0xFF) == my_digit) cross_rank++;
        if (((four >>  8) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 16) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 24) & 0xFF) == my_digit) cross_rank++;
    }
    uint rank = cross_rank + sg_rank;

    if (valid) {
        uint global_pos = counts[bid * RS_RADIX + my_digit] + rank;
        keys_out[global_pos] = my_key;
        vals_out[global_pos] = my_val;
    }
}

kernel void radix_sort_scatter_kernel(
    constant uint& capacity            [[buffer(0)]],
    device const uint64_t* keys_in     [[buffer(1)]],
    device const int32_t* vals_in      [[buffer(2)]],
    device uint64_t* keys_out          [[buffer(3)]],
    device int32_t* vals_out           [[buffer(4)]],
    device const uint* counts          [[buffer(5)]],
    constant uint& shift               [[buffer(6)]],
    device const int32_t* cum_tiles_hit [[buffer(7)]],
    constant uint& num_points          [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    // Read actual element count from GPU-resident prefix sum
    uint N = min(capacity, (uint)cum_tiles_hit[num_points - 1]);

    // Early exit for threadgroups entirely beyond N
    if (bid * RS_TG_SIZE >= N) return;

    threadgroup uchar shared_digits[RS_TG_SIZE];

    uint global_idx = bid * RS_TG_SIZE + tid;

    // Load element
    uint64_t my_key = 0;
    int32_t my_val = 0;
    uint my_digit = 0;
    bool valid = (global_idx < N);

    if (valid) {
        my_key = keys_in[global_idx];
        my_val = vals_in[global_idx];
        my_digit = extract_bits(my_key, shift, 8);
    }

    // Store digits for cross-simdgroup rank computation
    shared_digits[tid] = (uchar)my_digit;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hybrid rank: SIMD broadcast for intra-simdgroup, shared memory for cross-simdgroup.
    // Intra-simdgroup: all 32 lanes call simd_broadcast uniformly, each accumulates
    // only matches from lanes with lower index.
    uint sg_rank = 0;
    for (ushort l = 0; l < 32; l++) {
        uint other = simd_broadcast(my_digit, l);
        if (l < sg_lane && other == my_digit) sg_rank++;
    }

    // Cross-simdgroup: vectorized scan of preceding simdgroups via shared memory
    uint cross_rank = 0;
    uint preceding_end = sg_id * 32;
    uint full_words = preceding_end / 4;
    threadgroup uint* digits_u32 = (threadgroup uint*)shared_digits;
    for (uint i = 0; i < full_words; i++) {
        uint four = digits_u32[i];
        if (((four >>  0) & 0xFF) == my_digit) cross_rank++;
        if (((four >>  8) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 16) & 0xFF) == my_digit) cross_rank++;
        if (((four >> 24) & 0xFF) == my_digit) cross_rank++;
    }
    uint rank = cross_rank + sg_rank;

    // Write to global output at computed position
    if (valid) {
        uint global_pos = counts[bid * RS_RADIX + my_digit] + rank;
        keys_out[global_pos] = my_key;
        vals_out[global_pos] = my_val;
    }
}

// ===== Prefix Sum Kernel =====
// Single-dispatch inclusive prefix sum (cumsum) for int32 arrays.
// Uses one threadgroup: each thread serially sums its chunk, thread 0 scans
// block totals, then all threads write inclusive prefix sums.
// Used for small N (≤ PS_TG_SIZE) only; large N uses multi-threadgroup path.

#define PS_TG_SIZE 1024

kernel void prefix_sum_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    uint tg_tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    // Phase 1: Each thread serially sums its chunk
    uint chunk = (N + tg_size - 1) / tg_size;
    uint start = tg_tid * chunk;
    uint end = min(start + chunk, N);

    int my_sum = 0;
    for (uint i = start; i < end; i++) {
        my_sum += input[i];
    }

    // Phase 2: Two-level parallel prefix sum using SIMD
    // Level 1: intra-simdgroup exclusive prefix sum (hardware-accelerated)
    int sg_prefix = simd_prefix_exclusive_sum(my_sum);
    int sg_total = simd_sum(my_sum);

    // Level 2: cross-simdgroup scan (max 32 simdgroups for 1024 threads)
    uint num_sg = (tg_size + sg_size - 1) / sg_size;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    if (sg_lane == 0) {
        sg_totals[sg_id] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 scans simdgroup totals (max 32 iterations)
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (tg_tid == 0) {
        sg_offsets[0] = 0;
        for (uint i = 1; i < num_sg; i++) {
            sg_offsets[i] = sg_offsets[i - 1] + sg_totals[i - 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final prefix = cross-simdgroup offset + intra-simdgroup prefix
    int my_prefix = sg_offsets[sg_id] + sg_prefix;

    // Phase 3: Each thread writes inclusive prefix sum for its chunk
    int running = my_prefix;
    for (uint i = start; i < end; i++) {
        running += input[i];
        output[i] = running;
    }
}

// In-place inclusive prefix sum for compact intermediate arrays such as block
// totals. This avoids a separate scratch buffer before block propagation.
kernel void prefix_sum_inplace_kernel(
    constant uint& N,
    device int* values,
    uint tg_tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    uint chunk = (N + tg_size - 1) / tg_size;
    uint start = tg_tid * chunk;
    uint end = min(start + chunk, N);

    int my_sum = 0;
    for (uint i = start; i < end; i++) {
        my_sum += values[i];
    }

    int sg_prefix = simd_prefix_exclusive_sum(my_sum);
    int sg_total = simd_sum(my_sum);

    uint num_sg = (tg_size + sg_size - 1) / sg_size;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    if (sg_lane == 0) {
        sg_totals[sg_id] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (tg_tid == 0) {
        sg_offsets[0] = 0;
        for (uint i = 1; i < num_sg; i++) {
            sg_offsets[i] = sg_offsets[i - 1] + sg_totals[i - 1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int running = sg_offsets[sg_id] + sg_prefix;
    for (uint i = start; i < end; i++) {
        running += values[i];
        values[i] = running;
    }
}

// Multi-threadgroup prefix sum, pass 1: each threadgroup reduces its block of
// 1024 elements to a single total. Coalesced reads, 1 write per threadgroup.
kernel void block_reduce_kernel(
    constant uint& N,
    constant int* input,
    device int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]]
) {
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Two-level reduction: SIMD sum → cross-SIMD sum
    int sg_total = simd_sum(val);

    threadgroup int sg_sums[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_sums[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int total = 0;
        for (uint i = 0; i < PS_TG_SIZE / 32; i++) total += sg_sums[i];
        block_totals[tg_id] = total;
    }
}

// Multi-threadgroup prefix sum, pass 2: block_totals must already contain the
// inclusive prefix of per-block totals. Each threadgroup reads one offset and
// writes inclusive prefix sums with coalesced access.
kernel void block_scan_propagate_kernel(
    constant uint& N,
    constant int* input,
    device int* output,
    constant int* block_totals,
    uint tg_id [[threadgroup_position_in_grid]],
    uint tg_tid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint sg_lane [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]]
) {
    int block_offset = (tg_id == 0) ? 0 : block_totals[tg_id - 1];

    // Step 2: Load element (coalesced)
    uint idx = tg_id * PS_TG_SIZE + tg_tid;
    int val = (idx < N) ? input[idx] : 0;

    // Step 3: Intra-block inclusive prefix sum (SIMD + cross-SIMD)
    int sg_prefix = simd_prefix_exclusive_sum(val);
    int sg_total = simd_sum(val);

    uint num_sg = PS_TG_SIZE / 32;
    threadgroup int sg_totals[PS_TG_SIZE / 32];
    threadgroup int sg_offsets[PS_TG_SIZE / 32];
    if (sg_lane == 0) sg_totals[sg_id] = sg_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tg_tid == 0) {
        int acc = 0;
        for (uint i = 0; i < num_sg; i++) {
            sg_offsets[i] = acc;
            acc += sg_totals[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Write inclusive prefix sum (coalesced)
    int inclusive = block_offset + sg_offsets[sg_id] + sg_prefix + val;
    if (idx < N) output[idx] = inclusive;
}
