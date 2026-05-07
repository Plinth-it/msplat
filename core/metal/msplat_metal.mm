#import "bindings.h"
#define BLOCK_X 16
#define BLOCK_Y 16

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphAutomaticDifferentiation.h>
#import <chrono>
#import <dlfcn.h>
#import <unordered_map>
#import <functional>
#import <array>
#import <algorithm>
#import <mutex>
#import <memory>
#import <string>
#import <fstream>
#import <cstdlib>
#import <stdexcept>
#import <mach/mach_time.h>

// GPU profiling infrastructure.
// PROFILE_GPU=1: per-CB total GPU time via completion handlers.
// PROFILE_STAGES=1: per-stage GPU time via Metal timestamp counters + separate encoders.
static bool g_gpu_timing_enabled = false;
static bool g_gpu_timing_checked = false;
static std::mutex g_gpu_timing_mutex;
static std::vector<double> g_gpu_times_ms;

// Per-stage profiling
static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Stage names for training pipeline
static const char* g_train_stage_names[] = {
    "blit_zero", "proj_sh_fwd", "prefix_sort_pack", "rast_fwd",
    "loss_fwd_bwd", "rast_bwd", "proj_sh_bwd_adam", "grad_stats"
};
static constexpr int N_TRAIN_STAGES = 8;

static std::mutex g_stage_timing_mutex;
// Per-stage accumulated times (ms), indexed by stage
static std::vector<double> g_stage_times[N_TRAIN_STAGES];
static int g_stage_report_count = 0;

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    dispatch_queue_t d_queue;

    // Command buffer lifecycle using MPSCommandBuffer for commitAndContinue support.
    MPSCommandBuffer* _currentCB = nil;

    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_currentCB) {
            _currentCB = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [_currentCB retain];
        }
        return _currentCB;
    }
    void commitCB() {
        if (_currentCB) {
            if (g_gpu_timing_enabled) {
                [_currentCB addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
                    if (gpu_ms > 0) {
                        std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
                        g_gpu_times_ms.push_back(gpu_ms);
                    }
                }];
            }
            [_currentCB commitAndContinue];
        }
    }
    void syncCB() {
        if (_currentCB) {
            [_currentCB commit];
            [_currentCB waitUntilCompleted];
            [_currentCB release];
            _currentCB = nil;
        }
    }

    // Per-stage GPU timestamp profiling (Metal counter sample buffer)
    id<MTLCounterSampleBuffer> counterSampleBuffer;
    bool counterSamplingAvailable = false;
    double ticksToMs = 0.0;  // conversion factor from GPU ticks to milliseconds

    void initCounterSampling() {
        // Need 2 samples per stage (start + end)
        NSUInteger sampleCount = N_TRAIN_STAGES * 2;

        // Find timestamp counter set
        id<MTLCounterSet> timestampSet = nil;
        for (id<MTLCounterSet> cs in device.counterSets) {
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestampSet = cs;
                break;
            }
        }
        if (!timestampSet) {
            fprintf(stderr, "PROFILE_STAGES: MTLCommonCounterSetTimestamp not available\n");
            return;
        }

        // Check if stage boundary sampling is supported (guaranteed on Apple Silicon)
        if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            fprintf(stderr, "PROFILE_STAGES: AtStageBoundary sampling not supported\n");
            return;
        }

        MTLCounterSampleBufferDescriptor *desc = [MTLCounterSampleBufferDescriptor new];
        desc.counterSet = timestampSet;
        desc.sampleCount = sampleCount;
        desc.storageMode = MTLStorageModeShared;
        desc.label = @"msplat stage profiling";

        NSError *error = nil;
        counterSampleBuffer = [device newCounterSampleBufferWithDescriptor:desc error:&error];
        if (!counterSampleBuffer) {
            fprintf(stderr, "PROFILE_STAGES: Failed to create counter sample buffer: %s\n",
                    error.localizedDescription.UTF8String);
            return;
        }

        // Compute ticks-to-ms conversion (Apple Silicon: mach_absolute_time units)
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        ticksToMs = (double)tb.numer / (double)tb.denom / 1e6;

        counterSamplingAvailable = true;
        fprintf(stderr, "PROFILE_STAGES: GPU timestamp profiling enabled (%lu sample slots)\n",
                (unsigned long)sampleCount);
    }

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    // Tile-local sorting
    id<MTLComputePipelineState> scatter_to_prealloc_bins_kernel_cpso;
    id<MTLComputePipelineState> bitonic_sort_per_tile_kernel_cpso;
    // Prefix sum
    id<MTLComputePipelineState> prefix_sum_kernel_cpso;
    id<MTLComputePipelineState> prefix_sum_inplace_kernel_cpso;
    id<MTLComputePipelineState> block_reduce_kernel_cpso;
    id<MTLComputePipelineState> block_scan_propagate_kernel_cpso;
    // Depth-chunked rasterization
    id<MTLComputePipelineState> rasterize_forward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_forward_merge_kernel_cpso;
    id<MTLComputePipelineState> compute_chunk_prefix_suffix_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_chunked_kernel_cpso;
    id<MTLComputePipelineState> rasterize_backward_kernel_cpso;
    // Separable SSIM loss kernels
    id<MTLComputePipelineState> ssim_h_fwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_fwd_kernel_cpso;
    id<MTLComputePipelineState> l1_loss_fwd_bwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_fused_v_fwd_h_bwd_kernel_cpso;
    id<MTLComputePipelineState> ssim_v_bwd_kernel_cpso;
    id<MTLComputePipelineState> lpips_prepare_nchw_kernel_cpso;
    id<MTLComputePipelineState> lpips_apply_grad_kernel_cpso;
    // Backward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_backward_kernel_cpso;
    id<MTLComputePipelineState> fused_adam_kernel_cpso;
    id<MTLComputePipelineState> apply_mean_noise_kernel_cpso;
    id<MTLComputePipelineState> accumulate_grad_stats_kernel_cpso;
    id<MTLComputePipelineState> accumulate_pup_hessian_kernel_cpso;
    // GPU densification kernels
    id<MTLComputePipelineState> densify_classify_kernel_cpso;
    id<MTLComputePipelineState> densify_append_split_kernel_cpso;
    id<MTLComputePipelineState> densify_append_dup_kernel_cpso;
    id<MTLComputePipelineState> densify_cull_classify_kernel_cpso;
    id<MTLComputePipelineState> compact_scatter_kernel_cpso;
    id<MTLComputePipelineState> compact_copy_back_kernel_cpso;
};

// Explicit metallib path (set by Swift/Python wrappers before first use)
static char* g_metallib_path = NULL;
static char* g_lpips_weights_path = NULL;

extern "C" void msplat_set_metallib_path(const char* path) {
    free(g_metallib_path);
    g_metallib_path = path ? strdup(path) : NULL;
}

extern "C" void msplat_set_lpips_weights_path(const char* path) {
    free(g_lpips_weights_path);
    g_lpips_weights_path = path ? strdup(path) : NULL;
}

MetalContext* init_msplat_metal_context() {
    MetalContext* ctx = new MetalContext{};
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "msplat: Metal device not available\n");
        delete ctx;
        return NULL;
    }

    ctx->device = device;
    ctx->queue  = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "msplat: failed to create Metal command queue\n");
        delete ctx;
        return NULL;
    }
    ctx->d_queue = dispatch_queue_create("com.msplat.metal", DISPATCH_QUEUE_SERIAL);

    // Find precompiled metallib: explicit path (XCFramework/Python) or auto-discover
    NSError *error = nil;
    id<MTLLibrary> metal_library = nil;

    if (g_metallib_path) {
        // Explicit path (set by XCFramework / Swift wrapper)
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:g_metallib_path]];
        metal_library = [device newLibraryWithURL:url error:&error];
    } else {
        // Auto-discover default.metallib next to this library or the executable
        NSFileManager *fm = [NSFileManager defaultManager];

        // 1. Next to this shared library (Python .so / linked .a)
        Dl_info dl_info;
        if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
            NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if ([fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
        // 2. Next to the main executable (CLI build)
        if (!metal_library) {
            NSString *dir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"default.metallib"];
            if (dir && [fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
        // 3. Source-tree editable install / repo-local CLI usage.
        if (!metal_library) {
            NSString *path = [[[NSFileManager defaultManager] currentDirectoryPath]
                stringByAppendingPathComponent:@"build/default.metallib"];
            if ([fm fileExistsAtPath:path]) {
                metal_library = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            }
        }
    }

    if (!metal_library) {
        const char* detail = error ? [[error description] UTF8String] :
            (g_metallib_path ? g_metallib_path : "default.metallib not found");
        fprintf(stderr, "msplat: failed to load metallib: %s\n", detail);
        delete ctx;
        return NULL;
    }

    bool pipeline_load_failed = false;
    auto load = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [metal_library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "msplat: kernel not found: %s\n", [name UTF8String]);
            pipeline_load_failed = true;
            return nil;
        }
        NSError *pipelineError = nil;
        id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&pipelineError];
        [fn release];
        if (!pso || pipelineError) {
            fprintf(stderr, "msplat: failed to create pipeline for %s: %s\n",
                    [name UTF8String],
                    pipelineError ? [[pipelineError description] UTF8String] : "unknown error");
            pipeline_load_failed = true;
        }
        return pso;
    };

    // Forward pipeline
    ctx->project_and_sh_forward_kernel_cpso       = load(@"project_and_sh_forward_kernel");
    ctx->nd_rasterize_forward_kernel_cpso         = load(@"nd_rasterize_forward_kernel");
    // Tile-local sorting
    ctx->scatter_to_prealloc_bins_kernel_cpso      = load(@"scatter_to_prealloc_bins_kernel");
    ctx->bitonic_sort_per_tile_kernel_cpso        = load(@"bitonic_sort_per_tile_kernel");
    // Prefix sum
    ctx->prefix_sum_kernel_cpso                   = load(@"prefix_sum_kernel");
    ctx->prefix_sum_inplace_kernel_cpso           = load(@"prefix_sum_inplace_kernel");
    ctx->block_reduce_kernel_cpso                 = load(@"block_reduce_kernel");
    ctx->block_scan_propagate_kernel_cpso         = load(@"block_scan_propagate_kernel");
    // Depth-chunked rasterization
    ctx->rasterize_forward_chunked_kernel_cpso    = load(@"rasterize_forward_chunked_kernel");
    ctx->rasterize_forward_merge_kernel_cpso      = load(@"rasterize_forward_merge_kernel");
    ctx->compute_chunk_prefix_suffix_kernel_cpso  = load(@"compute_chunk_prefix_suffix_kernel");
    ctx->rasterize_backward_chunked_kernel_cpso   = load(@"rasterize_backward_chunked_kernel");
    ctx->rasterize_backward_kernel_cpso           = load(@"rasterize_backward_kernel");
    // Separable SSIM loss
    ctx->ssim_h_fwd_kernel_cpso                   = load(@"ssim_h_fwd_kernel");
    ctx->ssim_v_fwd_kernel_cpso                   = load(@"ssim_v_fwd_kernel");
    ctx->l1_loss_fwd_bwd_kernel_cpso              = load(@"l1_loss_fwd_bwd_kernel");
    ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso       = load(@"ssim_fused_v_fwd_h_bwd_kernel");
    ctx->ssim_v_bwd_kernel_cpso                   = load(@"ssim_v_bwd_kernel");
    ctx->lpips_prepare_nchw_kernel_cpso           = load(@"lpips_prepare_nchw_kernel");
    ctx->lpips_apply_grad_kernel_cpso             = load(@"lpips_apply_grad_kernel");
    // Backward pipeline
    ctx->project_and_sh_backward_kernel_cpso      = load(@"project_and_sh_backward_kernel");
    ctx->fused_adam_kernel_cpso                    = load(@"fused_adam_kernel");
    ctx->apply_mean_noise_kernel_cpso             = load(@"apply_mean_noise_kernel");
    ctx->accumulate_grad_stats_kernel_cpso        = load(@"accumulate_grad_stats_kernel");
    ctx->accumulate_pup_hessian_kernel_cpso       = load(@"accumulate_pup_hessian_kernel");
    // GPU densification
    ctx->densify_classify_kernel_cpso             = load(@"densify_classify_kernel");
    ctx->densify_append_split_kernel_cpso         = load(@"densify_append_split_kernel");
    ctx->densify_append_dup_kernel_cpso           = load(@"densify_append_dup_kernel");
    ctx->densify_cull_classify_kernel_cpso        = load(@"densify_cull_classify_kernel");
    ctx->compact_scatter_kernel_cpso              = load(@"compact_scatter_kernel");
    ctx->compact_copy_back_kernel_cpso            = load(@"compact_copy_back_kernel");

    auto requireThreadgroupSize = [&](id<MTLComputePipelineState> pso, NSString *name, NSUInteger required) {
        if (pso && pso.maxTotalThreadsPerThreadgroup < required) {
            fprintf(stderr, "msplat: kernel %s supports only %lu threads per threadgroup; need %lu\n",
                    [name UTF8String],
                    (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                    (unsigned long)required);
            pipeline_load_failed = true;
        }
    };
    requireThreadgroupSize(ctx->prefix_sum_kernel_cpso, @"prefix_sum_kernel", 1024);
    requireThreadgroupSize(ctx->prefix_sum_inplace_kernel_cpso, @"prefix_sum_inplace_kernel", 1024);
    requireThreadgroupSize(ctx->block_reduce_kernel_cpso, @"block_reduce_kernel", 1024);
    requireThreadgroupSize(ctx->block_scan_propagate_kernel_cpso, @"block_scan_propagate_kernel", 1024);

    [metal_library release];
    if (pipeline_load_failed) {
        delete ctx;
        return NULL;
    }

    // Initialize counter sampling if PROFILE_STAGES is set
    ctx->counterSampleBuffer = nil;
    ctx->counterSamplingAvailable = false;
    ctx->ticksToMs = 0.0;
    if (std::getenv("PROFILE_STAGES")) {
        g_profile_stages = true;
        g_profile_stages_checked = true;
        ctx->initCounterSampling();
    }

    return ctx;
}

static MetalContext* g_context = NULL;

MetalContext* get_global_context() {
    if (g_context == NULL) {
        g_context = init_msplat_metal_context();
    }
    if (g_context == NULL) {
        throw std::runtime_error("msplat: failed to initialize Metal context");
    }
    return g_context;
}



#define ENC_SCALAR(encoder, x, i) [encoder setBytes:&x length:sizeof(x) atIndex:i]
#define ENC_ARRAY(encoder, x, i) [encoder setBytes:x length:sizeof(x) atIndex:i]
#define ENC_BUF(encoder, x, i) [encoder setBuffer:x.buffer() offset:0 atIndex:i]

id<MTLDevice> msplat_device() {
    return get_global_context()->device;
}

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return mtensor_zeros(get_global_context()->device, std::move(shape), dtype);
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return mtensor_empty(get_global_context()->device, std::move(shape), dtype);
}

void msplat_commit() {
    if (!g_gpu_timing_checked) {
        g_gpu_timing_enabled = std::getenv("PROFILE_GPU") != nullptr;
        g_gpu_timing_checked = true;
    }
    get_global_context()->commitCB();
}

void msplat_gpu_sync() {
    get_global_context()->syncCB();
}

void msplat_enable_gpu_timing(bool enable) {
    g_gpu_timing_enabled = enable;
    g_gpu_timing_checked = true;
}

void msplat_drain_gpu_times(std::vector<double>& out) {
    std::lock_guard<std::mutex> lock(g_gpu_timing_mutex);
    out = std::move(g_gpu_times_ms);
    g_gpu_times_ms.clear();
}

void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names) {
    std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
    n_stages = std::min(max_stages, N_TRAIN_STAGES);
    for (int i = 0; i < n_stages; i++) {
        stage_times[i] = std::move(g_stage_times[i]);
        g_stage_times[i].clear();
        stage_names[i] = g_train_stage_names[i];
    }
}

void msplat_apply_mean_noise(
    int num_points, MTensor &means3d, MTensor &opacities, MTensor &radii,
    float noise_scale, float max_noise, uint32_t seed
) {
    if (num_points <= 0 || noise_scale <= 0.0f || max_noise <= 0.0f) {
        return;
    }

    MetalContext* ctx = get_global_context();
    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
    NSUInteger tpg = MIN(ctx->apply_mean_noise_kernel_cpso.maxTotalThreadsPerThreadgroup,
                         (NSUInteger)num_points);
    uint32_t n = (uint32_t)num_points;
    [enc setComputePipelineState:ctx->apply_mean_noise_kernel_cpso];
    ENC_BUF(enc, means3d, 0);
    ENC_BUF(enc, opacities, 1);
    ENC_BUF(enc, radii, 2);
    ENC_SCALAR(enc, n, 3);
    ENC_SCALAR(enc, noise_scale, 4);
    ENC_SCALAR(enc, max_noise, 5);
    ENC_SCALAR(enc, seed, 6);
    [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
}

#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8
static constexpr int kMaxTileElems = 4096;

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    int fwd_num_points = 0, capacity = 0, img_height = 0, img_width = 0, num_tiles = 0;
    int bwd_num_points = 0, features_rest_bases = 0;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, opacity_comp, num_tiles_hit, colors, aabb;
    MTensor gaussian_ids;
    MTensor packed_xy_opac, packed_conic, packed_rgb, packed_opacity_comp;
    MTensor out_img, final_Ts, final_idx;
    MTensor loss_intermediates;
    MTensor ssim_h_buf;
    MTensor tile_bins, loss_sum;
    MTensor lpips_rendered_nchw, lpips_gt_nchw, lpips_grad_nchw, lpips_loss;

    // Tile-local sorting buffers
    MTensor tile_offsets, tile_scatter_counters;
    MTensor prealloc_bins;  // [num_tiles × MAX_TILE_ELEMS] uint64 — pre-allocated per-tile bins

    // Multi-threadgroup prefix sum temp buffer
    MTensor block_totals;

    // Intersection overflow detection
    MTensor overflow_flag;
    int64_t capacity_multiplier = 16;

    // Depth-chunked rasterization buffers
    uint32_t current_K_max = 1;
    int chunk_K_max = 0;
    MTensor chunk_T, chunk_C, chunk_final_idx;
    MTensor prefix_T, after_C;

    // Backward gradient accumulators
    MTensor v_rendered;
    MTensor v_xy, v_conic, v_colors_rast, v_opacity, v_refine, v_depth;
    MTensor v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest;

    void ensure_forward(int np, int64_t cap, int ih, int iw, int nt,
                        bool needs_ssim_buffers, bool needs_lpips_buffers,
                        id<MTLDevice> dev) {
        if (np != fwd_num_points) {
            fwd_num_points = np;
            xys = mtensor_empty(dev, {np, 2}, DType::Float32);
            depths = mtensor_empty(dev, {np}, DType::Float32);
            radii_out = mtensor_empty(dev, {np}, DType::Int32);
            conics = mtensor_empty(dev, {np, 3}, DType::Float32);
            opacity_comp = mtensor_empty(dev, {np}, DType::Float32);
            num_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            colors = mtensor_empty(dev, {np, 3}, DType::Float32);
            aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
            block_totals = mtensor_empty(dev, {(np + 1023) / 1024}, DType::Int32);
        }
        if (cap != capacity) {
            capacity = cap;
            gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            packed_xy_opac = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_conic = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_rgb = mtensor_empty(dev, {cap, 3}, DType::Float32);
            packed_opacity_comp = mtensor_empty(dev, {cap}, DType::Float32);
        }
        bool image_size_changed = (ih != img_height || iw != img_width);
        if (image_size_changed) {
            img_height = ih; img_width = iw;
            out_img = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
            final_Ts = mtensor_empty(dev, {ih, iw}, DType::Float32);
            final_idx = mtensor_empty(dev, {ih, iw}, DType::Int32);
            v_rendered = mtensor_empty(dev, {ih, iw, 3}, DType::Float32);
        }
        if (needs_ssim_buffers) {
            if (image_size_changed || !loss_intermediates.defined() || !ssim_h_buf.defined()) {
                loss_intermediates = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
                ssim_h_buf = mtensor_empty(dev, {(int64_t)ih, (int64_t)iw, 15}, DType::Float32);
            }
        } else {
            if (loss_intermediates.defined()) loss_intermediates.reset();
            if (ssim_h_buf.defined()) ssim_h_buf.reset();
        }
        if (needs_lpips_buffers) {
            if (image_size_changed || !lpips_rendered_nchw.defined() ||
                !lpips_gt_nchw.defined() || !lpips_grad_nchw.defined() ||
                !lpips_loss.defined()) {
                lpips_rendered_nchw = mtensor_empty(dev, {1, 3, ih, iw}, DType::Float32);
                lpips_gt_nchw = mtensor_empty(dev, {1, 3, ih, iw}, DType::Float32);
                lpips_grad_nchw = mtensor_empty(dev, {1, 3, ih, iw}, DType::Float32);
                lpips_loss = mtensor_empty(dev, {1}, DType::Float32);
            }
        } else {
            if (lpips_rendered_nchw.defined()) lpips_rendered_nchw.reset();
            if (lpips_gt_nchw.defined()) lpips_gt_nchw.reset();
            if (lpips_grad_nchw.defined()) lpips_grad_nchw.reset();
            if (lpips_loss.defined()) lpips_loss.reset();
        }
        if (nt != num_tiles) {
            num_tiles = nt;
            tile_bins = mtensor_empty(dev, {nt, 2}, DType::Int32);
            tile_offsets = mtensor_empty(dev, {nt}, DType::Int32);
            tile_scatter_counters = mtensor_empty(dev, {nt}, DType::Int32);
            prealloc_bins = mtensor_empty(dev, {(int64_t)nt * kMaxTileElems}, DType::Int64);
        }
        if (!loss_sum.defined()) {
            loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    void ensure_chunks(int K, int ih, int iw, id<MTLDevice> dev) {
        if (K <= chunk_K_max && ih == img_height && iw == img_width) return;
        chunk_K_max = K;
        chunk_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        chunk_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
        chunk_final_idx = mtensor_empty(dev, {K, ih, iw}, DType::Int32);
        prefix_T = mtensor_empty(dev, {K, ih, iw}, DType::Float32);
        after_C = mtensor_empty(dev, {K, ih, iw, 3}, DType::Float32);
    }

    void ensure_backward(int np, int frb, id<MTLDevice> dev) {
        if (np != bwd_num_points || frb != features_rest_bases || !v_xy.defined()) {
            bwd_num_points = np;
            features_rest_bases = frb;
            v_xy = mtensor_empty(dev, {np, 2}, DType::Float32);
            v_conic = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_colors_rast = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_opacity = mtensor_empty(dev, {np, 1}, DType::Float32);
            v_refine = mtensor_empty(dev, {np}, DType::Float32);
            v_depth = mtensor_empty(dev, {np}, DType::Float32);
            v_mean3d = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_scale = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_quat = mtensor_empty(dev, {np, 4}, DType::Float32);
            v_features_dc = mtensor_empty(dev, {np, 3}, DType::Float32);
            v_features_rest = mtensor_empty(dev, {(int64_t)np, (int64_t)frb, 3}, DType::Float32);
        }
    }
};
static FusedTensorCache g_tcache;

void cleanup_msplat_metal() {
    if (g_context) {
        g_context->syncCB();
    }
    g_tcache = FusedTensorCache{};
}

struct LpipsTensorBlob {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> data;
};

struct LpipsWeights {
    std::vector<LpipsTensorBlob> tensors;
    const LpipsTensorBlob* tensorNamed(const std::string& name) const {
        for (const auto& t : tensors) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
};

struct LpipsGraphCache {
    int height = 0;
    int width = 0;
    MPSGraph* graph = nil;
    MPSGraphTensor* input_a = nil;
    MPSGraphTensor* input_b = nil;
    MPSGraphTensor* loss = nil;
    MPSGraphTensor* grad_a = nil;
};

static std::unique_ptr<LpipsWeights> g_lpips_weights;
static std::unique_ptr<LpipsGraphCache> g_lpips_graph;
static std::mutex g_lpips_mutex;

static NSString* discover_lpips_weights_path() {
    NSFileManager *fm = [NSFileManager defaultManager];
    if (g_lpips_weights_path) {
        NSString *path = [NSString stringWithUTF8String:g_lpips_weights_path];
        if ([fm fileExistsAtPath:path]) return path;
    }
    if (g_metallib_path) {
        NSString *dir = [[NSString stringWithUTF8String:g_metallib_path] stringByDeletingLastPathComponent];
        NSString *path = [dir stringByAppendingPathComponent:@"lpips_vgg.bin"];
        if ([fm fileExistsAtPath:path]) return path;
    }
    Dl_info dl_info;
    if (dladdr((void*)init_msplat_metal_context, &dl_info) && dl_info.dli_fname) {
        NSString *dir = [[NSString stringWithUTF8String:dl_info.dli_fname] stringByDeletingLastPathComponent];
        NSString *path = [dir stringByAppendingPathComponent:@"lpips_vgg.bin"];
        if ([fm fileExistsAtPath:path]) return path;
    }
    NSString *exeDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    NSString *exePath = [exeDir stringByAppendingPathComponent:@"lpips_vgg.bin"];
    if (exeDir && [fm fileExistsAtPath:exePath]) return exePath;
    return nil;
}

template <typename T>
static bool read_lpips_value(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return in.good();
}

static LpipsWeights* load_lpips_weights() {
    if (g_lpips_weights) return g_lpips_weights.get();

    NSString *path = discover_lpips_weights_path();
    if (!path) {
        fprintf(stderr, "msplat: LPIPS weights not found (lpips_vgg.bin)\n");
        return nullptr;
    }

    std::ifstream in([path fileSystemRepresentation], std::ios::binary);
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in.good() || std::string(magic, sizeof(magic)) != "MSLPIPS1") {
        fprintf(stderr, "msplat: invalid LPIPS weights file: %s\n", [path UTF8String]);
        return nullptr;
    }

    uint32_t count = 0;
    if (!read_lpips_value(in, count)) return nullptr;

    auto weights = std::make_unique<LpipsWeights>();
    weights->tensors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t name_len = 0, rank = 0;
        uint64_t data_len = 0;
        if (!read_lpips_value(in, name_len)) return nullptr;
        LpipsTensorBlob tensor;
        tensor.name.resize(name_len);
        in.read(tensor.name.data(), name_len);
        if (!read_lpips_value(in, rank)) return nullptr;
        tensor.shape.resize(rank);
        uint64_t expected = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            if (!read_lpips_value(in, tensor.shape[d])) return nullptr;
            expected *= tensor.shape[d];
        }
        if (!read_lpips_value(in, data_len) || data_len != expected) return nullptr;
        tensor.data.resize((size_t)data_len);
        in.read(reinterpret_cast<char*>(tensor.data.data()), (std::streamsize)(data_len * sizeof(float)));
        if (!in.good()) return nullptr;
        weights->tensors.push_back(std::move(tensor));
    }

    g_lpips_weights = std::move(weights);
    return g_lpips_weights.get();
}

static NSArray<NSNumber*>* shape4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return @[@(a), @(b), @(c), @(d)];
}

static MPSGraphTensor* constant_for_tensor(MPSGraph* graph, const LpipsTensorBlob& t) {
    NSMutableArray<NSNumber*> *shape = [NSMutableArray arrayWithCapacity:t.shape.size()];
    for (uint32_t dim : t.shape) [shape addObject:@(dim)];
    NSData *data = [NSData dataWithBytes:t.data.data() length:t.data.size() * sizeof(float)];
    return [graph constantWithData:data shape:shape dataType:MPSDataTypeFloat32];
}

static MPSGraphTensor* add_scalar(MPSGraph* graph, MPSGraphTensor* tensor, float value) {
    MPSGraphTensor *scalar = [graph constantWithScalar:value dataType:MPSDataTypeFloat32];
    return [graph additionWithPrimaryTensor:tensor secondaryTensor:scalar name:nil];
}

static MPSGraphTensor* lpips_conv_relu(MPSGraph* graph, MPSGraphTensor* x,
                                       const LpipsTensorBlob& weight,
                                       const LpipsTensorBlob* bias) {
    MPSGraphConvolution2DOpDescriptor *desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1
                                                   dilationRateInX:1 dilationRateInY:1
                                                            groups:1
                                                       paddingLeft:1 paddingRight:1
                                                        paddingTop:1 paddingBottom:1
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    x = [graph convolution2DWithSourceTensor:x weightsTensor:constant_for_tensor(graph, weight)
                                  descriptor:desc name:nil];
    if (bias) {
        MPSGraphTensor *b = constant_for_tensor(graph, *bias);
        b = [graph reshapeTensor:b withShape:shape4(1, bias->shape[0], 1, 1) name:nil];
        x = [graph additionWithPrimaryTensor:x secondaryTensor:b name:nil];
    }
    return [graph reLUWithTensor:x name:nil];
}

static MPSGraphTensor* lpips_head(MPSGraph* graph, MPSGraphTensor* x,
                                  const LpipsTensorBlob& weight) {
    MPSGraphConvolution2DOpDescriptor *desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1
                                                   dilationRateInX:1 dilationRateInY:1
                                                            groups:1
                                                       paddingLeft:0 paddingRight:0
                                                        paddingTop:0 paddingBottom:0
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    return [graph convolution2DWithSourceTensor:x weightsTensor:constant_for_tensor(graph, weight)
                                     descriptor:desc name:nil];
}

static MPSGraphTensor* lpips_norm(MPSGraph* graph, MPSGraphTensor* x, int height, int width) {
    MPSGraphTensor *sq = [graph squareWithTensor:x name:nil];
    MPSGraphTensor *sum = [graph reductionSumWithTensor:sq axes:@[@1] name:nil];
    sum = [graph reshapeTensor:sum withShape:shape4(1, 1, (uint32_t)height, (uint32_t)width) name:nil];
    MPSGraphTensor *norm = [graph squareRootWithTensor:add_scalar(graph, sum, 1e-10f) name:nil];
    return [graph divisionWithPrimaryTensor:x secondaryTensor:norm name:nil];
}

static MPSGraphTensor* lpips_preprocess(MPSGraph* graph, MPSGraphTensor* x) {
    MPSGraphTensor *two = [graph constantWithScalar:2.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    x = [graph multiplicationWithPrimaryTensor:x secondaryTensor:two name:nil];
    x = [graph subtractionWithPrimaryTensor:x secondaryTensor:one name:nil];

    const float shift_vals[3] = {-0.030f, -0.088f, -0.188f};
    const float scale_vals[3] = {0.458f, 0.448f, 0.450f};
    NSData *shift_data = [NSData dataWithBytes:shift_vals length:sizeof(shift_vals)];
    NSData *scale_data = [NSData dataWithBytes:scale_vals length:sizeof(scale_vals)];
    MPSGraphTensor *shift = [graph constantWithData:shift_data shape:shape4(1, 3, 1, 1) dataType:MPSDataTypeFloat32];
    MPSGraphTensor *scale = [graph constantWithData:scale_data shape:shape4(1, 3, 1, 1) dataType:MPSDataTypeFloat32];
    x = [graph subtractionWithPrimaryTensor:x secondaryTensor:shift name:nil];
    return [graph divisionWithPrimaryTensor:x secondaryTensor:scale name:nil];
}

static LpipsGraphCache* get_lpips_graph(int height, int width) {
    std::lock_guard<std::mutex> lock(g_lpips_mutex);
    if (height < 16 || width < 16) {
        fprintf(stderr, "msplat: LPIPS requires images at least 16x16 after downscaling\n");
        return nullptr;
    }
    if (g_lpips_graph && g_lpips_graph->height == height && g_lpips_graph->width == width) {
        return g_lpips_graph.get();
    }

    LpipsWeights *weights = load_lpips_weights();
    if (!weights) return nullptr;

    auto cache = std::make_unique<LpipsGraphCache>();
    cache->height = height;
    cache->width = width;
    cache->graph = [MPSGraph new];
    MPSGraph *graph = cache->graph;

    cache->input_a = [graph placeholderWithShape:shape4(1, 3, (uint32_t)height, (uint32_t)width)
                                        dataType:MPSDataTypeFloat32 name:@"lpips_rendered"];
    cache->input_b = [graph placeholderWithShape:shape4(1, 3, (uint32_t)height, (uint32_t)width)
                                        dataType:MPSDataTypeFloat32 name:@"lpips_gt"];

    MPSGraphTensor *a = lpips_preprocess(graph, cache->input_a);
    MPSGraphTensor *b = lpips_preprocess(graph, cache->input_b);
    MPSGraphTensor *total = [graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];

    const int block_convs[5] = {2, 2, 3, 3, 3};
    int h = height, w = width;
    for (int block = 0; block < 5; ++block) {
        if (block != 0) {
            MPSGraphPooling2DOpDescriptor *pool =
                [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:2 kernelHeight:2
                                                                strideInX:2 strideInY:2
                                                             paddingStyle:MPSGraphPaddingStyleExplicit
                                                               dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
            [pool setExplicitPaddingWithPaddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0];
            a = [graph maxPooling2DWithSourceTensor:a descriptor:pool name:nil];
            b = [graph maxPooling2DWithSourceTensor:b descriptor:pool name:nil];
            h /= 2;
            w /= 2;
        }
        for (int conv = 0; conv < block_convs[block]; ++conv) {
            std::string base = "blocks." + std::to_string(block) + ".convs." + std::to_string(conv);
            const LpipsTensorBlob *conv_w = weights->tensorNamed(base + ".weight");
            const LpipsTensorBlob *conv_b = weights->tensorNamed(base + ".bias");
            if (!conv_w || !conv_b) {
                fprintf(stderr, "msplat: missing LPIPS tensor %s\n", base.c_str());
                return nullptr;
            }
            a = lpips_conv_relu(graph, a, *conv_w, conv_b);
            b = lpips_conv_relu(graph, b, *conv_w, conv_b);
        }

        MPSGraphTensor *na = lpips_norm(graph, a, h, w);
        MPSGraphTensor *nb = lpips_norm(graph, b, h, w);
        MPSGraphTensor *diff = [graph subtractionWithPrimaryTensor:na secondaryTensor:nb name:nil];
        diff = [graph squareWithTensor:diff name:nil];

        const LpipsTensorBlob *head_w = weights->tensorNamed("heads." + std::to_string(block) + ".weight");
        if (!head_w) {
            fprintf(stderr, "msplat: missing LPIPS head %d\n", block);
            return nullptr;
        }
        MPSGraphTensor *head = lpips_head(graph, diff, *head_w);
        MPSGraphTensor *mean = [graph meanOfTensor:head axes:@[@2, @3] name:nil];
        mean = [graph reshapeTensor:mean withShape:@[@1] name:nil];
        total = [graph additionWithPrimaryTensor:total secondaryTensor:mean name:nil];
    }

    cache->loss = total;
    NSDictionary<MPSGraphTensor*, MPSGraphTensor*> *grads =
        [graph gradientForPrimaryTensor:cache->loss withTensors:@[cache->input_a] name:nil];
    cache->grad_a = grads[cache->input_a];
    if (!cache->grad_a) {
        fprintf(stderr, "msplat: failed to create LPIPS gradient graph\n");
        return nullptr;
    }

    g_lpips_graph = std::move(cache);
    return g_lpips_graph.get();
}

// Internal forward pipeline — used by both msplat_render and msplat_train_step.
// When compute_loss=false, gt/window2d/ssim_weight are ignored.
static void forward_pipeline(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    int use_mip_splatting,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    bool compute_loss
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    // --- Overflow check: detect per-tile overflow (> kMaxTileElems gaussians in a tile) ---
    // Only warn once to avoid noisy output (per-tile overflow is common at >1M gaussians)
    static bool overflow_warned = false;
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (!overflow_warned && g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            fprintf(stderr, "WARNING: tile intersection overflow. "
                    "Some gaussians were dropped from overfull tiles or packed tile output.\n");
            overflow_warned = true;
        }
    }
    int64_t capacity = (int64_t)num_points * g_tcache.capacity_multiplier;
    uint32_t channels = 3;
    uint32_t use_mip_splatting_u32 = use_mip_splatting ? 1u : 0u;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                            compute_loss, false, ctx->device);
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &opacity_comp = g_tcache.opacity_comp;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &packed_opacity_comp = g_tcache.packed_opacity_comp;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    uint32_t composite_gt_u32 = 0;

    // --- Constants (heap-allocated for Obj-C block) ---
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});

    // Periodic diagnostic: print key dimensions for roofline analysis
    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") && (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
            fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
            fprintf(stderr, "  num_points:     %d\n", num_points);
            fprintf(stderr, "  capacity:       %lld (= num_points * %lld)\n", (long long)capacity, (long long)g_tcache.capacity_multiplier);
            fprintf(stderr, "  img:            %u x %u = %u pixels\n", img_width, img_height, img_width * img_height);
            fprintf(stderr, "  tiles:          %d x %d = %d\n", tile_bounds_x, tile_bounds_y, num_tiles);
            fprintf(stderr, "  SH degree:      %u (bases: %u)\n", degree, (degree + 1) * (degree + 1));
            fprintf(stderr, "  features_rest:  [%lld x %lld x %lld]\n",
                (long long)features_rest.size(0), (long long)features_rest.size(1), (long long)features_rest.size(2));
            fprintf(stderr, "  sort:           tile-local (bitonic, max %d/tile)\n", kMaxTileElems);
            fprintf(stderr, "  sort buffer:    %.1f MB (sort_pairs)\n", (double)capacity * 8.0 / 1e6);
            fprintf(stderr, "  opacities:      [%lld]\n", (long long)opacities.size(0));
            fprintf(stderr, "===========================\n\n");
    }

    // Helper lambdas to encode each stage onto a given encoder
    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, opacity_comp, 23); ENC_SCALAR(enc, use_mip_splatting_u32, 24);
        ENC_BUF(enc, opacities, 25);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        // 1. scatter_to_prealloc_bins (replaces count_intersections + scatter_to_tiles)
        {
            NSUInteger tpg = MIN(ctx->scatter_to_prealloc_bins_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_prealloc_bins_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3); ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 6);
            ENC_BUF(enc, g_tcache.prealloc_bins, 7);
            ENC_BUF(enc, g_tcache.overflow_flag, 8);
            ENC_BUF(enc, conics, 9);
            ENC_BUF(enc, opacities, 10);
            ENC_BUF(enc, opacity_comp, 11);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2. prefix_sum(tile_scatter_counters -> tile_offsets)
        {
            NSUInteger tg2 = MIN(ctx->prefix_sum_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
            [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
            ENC_SCALAR(enc, num_tiles_u32, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1); ENC_BUF(enc, g_tcache.tile_offsets, 2);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 3. bitonic_sort_per_tile (reads prealloc_bins, writes packed + tile_bins)
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1);
            ENC_BUF(enc, g_tcache.prealloc_bins, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5); ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7); ENC_BUF(enc, opacities, 8); ENC_BUF(enc, opacity_comp, 9);
            ENC_BUF(enc, packed_xy_opac, 10); ENC_BUF(enc, packed_conic, 11); ENC_BUF(enc, packed_rgb, 12);
            ENC_BUF(enc, packed_opacity_comp, 13); ENC_BUF(enc, tile_bins, 14);
            ENC_SCALAR(enc, capacity_u32, 15); ENC_BUF(enc, g_tcache.overflow_flag, 16);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    // K_max for chunked rasterization — set after GPU readback
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;

    auto encode_rast_fwd_monolithic = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, packed_opacity_comp, 7);
        ENC_BUF(enc, final_Ts, 8); ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, out_img, 10);
        ENC_BUF(enc, background, 11);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    auto encode_rast_fwd_chunked = [&](id<MTLComputeCommandEncoder> enc) {
        // Phase 1: dispatch forward chunked kernel — grid (tile_x, tile_y, K_max)
        uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
        uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
        uint32_t num_pix = img_width * img_height;
        auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
        MTLSize chunked_tg = MTLSizeMake(tile_x, tile_y, K_max);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, packed_opacity_comp, 7);
        ENC_BUF(enc, g_tcache.chunk_T, 8); ENC_BUF(enc, g_tcache.chunk_C, 9); ENC_BUF(enc, g_tcache.chunk_final_idx, 10);
        ENC_SCALAR(enc, CHUNK_SIZE, 11); ENC_SCALAR(enc, K_max, 12);
        [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:13];
        [enc dispatchThreadgroups:chunked_tg threadsPerThreadgroup:tg_size];

        // Phase 2: merge kernel — one thread per pixel
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger merge_tpg = 256;
        [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
        ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
        ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
        ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
        ENC_BUF(enc, background, 8);
        [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
        [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            encode_rast_fwd_monolithic(enc);
        } else {
            encode_rast_fwd_chunked(enc);
        }
    };

    auto encode_loss_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        // Separable SSIM forward: H conv → barrier → V conv + SSIM + reduction
        MTLSize loss_tg_count = MTLSizeMake((img_width + 15) / 16, (img_height + 15) / 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);

        // Pass 1: horizontal convolution
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        ENC_BUF(enc, background, 4); ENC_SCALAR(enc, composite_gt_u32, 5);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Pass 2: vertical convolution + SSIM/L1 + loss reduction
        [enc setComputePipelineState:ctx->ssim_v_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4);
        ENC_BUF(enc, loss_intermediates, 5); ENC_BUF(enc, loss_sum, 6);
        ENC_BUF(enc, background, 7); ENC_SCALAR(enc, composite_gt_u32, 8);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
    };

    // Zero cached buffers will be done inside the command encoder via blit.
    // (CPU memset would race with previous CB's GPU reads.)

    // Compute K_max conservatively from CPU-side data (no GPU sync needed).
    // avg_per_tile = capacity / num_tiles. Use 6x average to cover heavy-tailed
    // tile distributions. Overestimate is cheap: empty chunks early-exit immediately.
    // If the densest tile exceeds K_max * CHUNK_SIZE, those gaussians are silently
    // skipped — but 6x covers typical skew (measured max/avg ratio: ~1.5-2.5x).
    if (num_tiles >= 400) {
        // High-res: enough tiles for good GPU occupancy, skip chunking
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;  // 6x average — covers heavy-tailed distributions
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
        uint32_t tile_cap_chunks = (uint32_t)((kMaxTileElems + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > tile_cap_chunks) K_max = tile_cap_chunks;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            // Blit-zero buffers that accumulate across gaussians (must be GPU-side
            // to avoid racing with previous CB's reads on pipelined execution)
            id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
            // tile_bins written by sort kernel, tile_counts no longer used
            [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");

            encode_proj_sh(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_map(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(encoder);
            if (compute_loss) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_loss_fwd(encoder);
            }

            [encoder endEncoding];
        });
    }

    // All outputs are in g_tcache — no return needed
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    int use_mip_splatting
) {
    MTensor dummyGt, dummyWindow;
    forward_pipeline(num_points, means3d, scales, glob_scale,
        quats, viewmat, projmat, fx, fy, cx, cy,
        img_height, img_width, tile_bounds, clip_thresh,
        degree, degrees_to_use, cam_pos, features_dc, features_rest,
        opacities, background, use_mip_splatting, dummyGt, dummyWindow, 0.0f, false);
    return g_tcache.out_img;
}

std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    int use_mip_splatting,
    MTensor &gt_packed, int use_loss_mask,
    int use_alpha_loss, float alpha_loss_weight, int composite_gt,
    MTensor &window2d, float ssim_weight, float lpips_loss_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    int reduce_second_moment,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim, float inv_width, float inv_height,
    MTensor *pup_hessian
) {
    MetalContext* ctx = get_global_context();
    int tile_bounds_x = std::get<0>(tile_bounds);
    int tile_bounds_y = std::get<1>(tile_bounds);
    int num_tiles = tile_bounds_x * tile_bounds_y;

    // --- Overflow check: detect per-tile overflow (> kMaxTileElems gaussians in a tile) ---
    // Only warn once to avoid noisy output (per-tile overflow is common at >1M gaussians)
    static bool overflow_warned = false;
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (!overflow_warned && g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            fprintf(stderr, "WARNING: tile intersection overflow. "
                    "Some gaussians were dropped from overfull tiles or packed tile output.\n");
            overflow_warned = true;
        }
    }
    int64_t capacity = (int64_t)num_points * g_tcache.capacity_multiplier;
    uint32_t channels = 3;
    uint32_t use_mip_splatting_u32 = use_mip_splatting ? 1u : 0u;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                            ssim_weight > 0.0f, lpips_loss_weight > 0.0f,
                            ctx->device);
    g_tcache.ensure_backward(num_points, features_rest_bases, ctx->device);

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &opacity_comp = g_tcache.opacity_comp;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &loss_sum = g_tcache.loss_sum;
    MTensor &packed_xy_opac = g_tcache.packed_xy_opac;
    MTensor &packed_conic = g_tcache.packed_conic;
    MTensor &packed_rgb = g_tcache.packed_rgb;
    MTensor &packed_opacity_comp = g_tcache.packed_opacity_comp;
    MTensor &out_img = g_tcache.out_img;
    MTensor &final_Ts = g_tcache.final_Ts;
    MTensor &final_idx = g_tcache.final_idx;
    MTensor &loss_intermediates = g_tcache.loss_intermediates;

    MTensor &v_rendered = g_tcache.v_rendered;
    MTensor &v_xy = g_tcache.v_xy;
    MTensor &v_conic = g_tcache.v_conic;
    MTensor &v_colors_rast = g_tcache.v_colors_rast;
    MTensor &v_opacity = g_tcache.v_opacity;
    MTensor &v_refine = g_tcache.v_refine;
    MTensor &v_depth = g_tcache.v_depth;
    MTensor &v_mean3d = g_tcache.v_mean3d;
    MTensor &v_scale = g_tcache.v_scale;
    MTensor &v_quat = g_tcache.v_quat;
    MTensor &v_features_dc = g_tcache.v_features_dc;
    MTensor &v_features_rest = g_tcache.v_features_rest;

    // Wire backward outputs as Adam grads (MTensor references for gradient buffers)
    auto adam_grads = std::make_shared<std::array<MTensor, 6>>(
        std::array<MTensor, 6>{v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest, v_opacity});

    // --- Constants (heap-allocated for Obj-C block capture) ---
    auto loss_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    uint32_t lpips_numel = img_width * img_height * 3;
    uint32_t use_loss_mask_u32 = use_loss_mask ? 1u : 0u;
    uint32_t use_alpha_loss_u32 = use_alpha_loss ? 1u : 0u;
    uint32_t composite_gt_u32 = composite_gt ? 1u : 0u;
    float alpha_loss_grad_scale = use_alpha_loss ? (alpha_loss_weight / (float)(img_height * img_width)) : 0.0f;
    auto proj_intrins = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_img_size = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto tile_bounds_arr = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    });
    auto cam_pos_arr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{cam_pos[0], cam_pos[1], cam_pos[2], 0.0f});
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    auto img_size_dim3 = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{img_width, img_height, 1, 0xDEAD});
    auto block_size_dim2 = std::make_shared<std::array<int32_t, 2>>(std::array<int32_t, 2>{RAST_BLOCK_X, RAST_BLOCK_Y});
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    auto rast_tb = std::make_shared<std::array<uint32_t, 4>>(std::array<uint32_t, 4>{
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD});
    auto rast_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
    auto proj_bwd_intr = std::make_shared<std::array<float, 4>>(std::array<float, 4>{fx, fy, cx, cy});
    auto proj_bwd_isz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});

    // --- K_max for chunked rasterization ---
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;
    if (num_tiles >= 400) {
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
        uint32_t tile_cap_chunks = (uint32_t)((kMaxTileElems + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > tile_cap_chunks) K_max = tile_cap_chunks;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }

    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;

    // ========================== FORWARD ENCODE LAMBDAS ==========================

    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_forward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_forward_kernel_cpso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins->data() length:sizeof(*proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size->data() length:sizeof(*proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, opacity_comp, 23); ENC_SCALAR(enc, use_mip_splatting_u32, 24);
        ENC_BUF(enc, opacities, 25);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        // 1. scatter_to_prealloc_bins
        {
            NSUInteger tpg = MIN(ctx->scatter_to_prealloc_bins_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_prealloc_bins_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0); ENC_BUF(enc, xys, 1); ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3); ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 6);
            ENC_BUF(enc, g_tcache.prealloc_bins, 7);
            ENC_BUF(enc, g_tcache.overflow_flag, 8);
            ENC_BUF(enc, conics, 9);
            ENC_BUF(enc, opacities, 10);
            ENC_BUF(enc, opacity_comp, 11);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 2. prefix_sum(tile_scatter_counters -> tile_offsets)
        {
            NSUInteger tg2 = MIN(ctx->prefix_sum_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)1024);
            [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
            ENC_SCALAR(enc, num_tiles_u32, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1); ENC_BUF(enc, g_tcache.tile_offsets, 2);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // 3. bitonic_sort_per_tile (reads prealloc_bins, writes packed + tile_bins)
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0); ENC_BUF(enc, g_tcache.tile_scatter_counters, 1);
            ENC_BUF(enc, g_tcache.prealloc_bins, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5); ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7); ENC_BUF(enc, opacities, 8); ENC_BUF(enc, opacity_comp, 9);
            ENC_BUF(enc, packed_xy_opac, 10); ENC_BUF(enc, packed_conic, 11); ENC_BUF(enc, packed_rgb, 12);
            ENC_BUF(enc, packed_opacity_comp, 13); ENC_BUF(enc, tile_bins, 14);
            ENC_SCALAR(enc, capacity_u32, 15); ENC_BUF(enc, g_tcache.overflow_flag, 16);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, final_Ts, 8); ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, out_img, 10);
            ENC_BUF(enc, background, 11);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:12];
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto img_sz_2 = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
            [enc setBytes:tile_bounds_arr->data() length:sizeof(*tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3->data() length:sizeof(*img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, g_tcache.chunk_T, 8); ENC_BUF(enc, g_tcache.chunk_C, 9); ENC_BUF(enc, g_tcache.chunk_final_idx, 10);
            ENC_SCALAR(enc, CHUNK_SIZE, 11); ENC_SCALAR(enc, K_max, 12);
            [enc setBytes:block_size_dim2->data() length:sizeof(*block_size_dim2) atIndex:13];
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Merge
            [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
            ENC_BUF(enc, background, 8);
            [enc setBytes:img_sz_2->data() length:sizeof(*img_sz_2) atIndex:9];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
    };

    // Fused loss: ssim_h_fwd → fused_v_fwd_h_bwd → ssim_v_bwd
    // Eliminates loss_intermediates round-trip (130 MB/iter bandwidth saved).
    auto encode_loss_fwd_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize loss_tg_count = MTLSizeMake((img_width + 15) / 16, (img_height + 15) / 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        if (ssim_weight <= 0.0f) {
            [enc setComputePipelineState:ctx->l1_loss_fwd_bwd_kernel_cpso];
            ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
            [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
            ENC_SCALAR(enc, loss_inv_n, 3);
            ENC_BUF(enc, v_rendered, 4); ENC_BUF(enc, loss_sum, 5);
            ENC_BUF(enc, background, 6); ENC_SCALAR(enc, composite_gt_u32, 7);
            ENC_SCALAR(enc, use_loss_mask_u32, 8);
            ENC_BUF(enc, final_Ts, 9);
            ENC_SCALAR(enc, use_alpha_loss_u32, 10); ENC_SCALAR(enc, alpha_loss_weight, 11);
            [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
            return;
        }
        // Pass 1: H conv on images → ssim_h_buf
        [enc setComputePipelineState:ctx->ssim_h_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        ENC_BUF(enc, background, 4); ENC_SCALAR(enc, composite_gt_u32, 5);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 2: Fused V fwd + H bwd
        [enc setComputePipelineState:ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, loss_intermediates, 6); ENC_BUF(enc, loss_sum, 7);
        ENC_BUF(enc, background, 8); ENC_SCALAR(enc, composite_gt_u32, 9);
        ENC_SCALAR(enc, use_loss_mask_u32, 10);
        ENC_BUF(enc, final_Ts, 11);
        ENC_SCALAR(enc, use_alpha_loss_u32, 12); ENC_SCALAR(enc, alpha_loss_weight, 13);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 3: V bwd
        [enc setComputePipelineState:ctx->ssim_v_bwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        ENC_BUF(enc, loss_intermediates, 2);
        [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, v_rendered, 6);
        ENC_BUF(enc, background, 7); ENC_SCALAR(enc, composite_gt_u32, 8);
        ENC_SCALAR(enc, use_loss_mask_u32, 9);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
    };

    auto encode_lpips = [&](MPSCommandBuffer *command_buffer) {
        if (lpips_loss_weight <= 0.0f) return;
        LpipsGraphCache *lpips = get_lpips_graph((int)img_height, (int)img_width);
        if (!lpips) std::abort();

        {
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setComputePipelineState:ctx->lpips_prepare_nchw_kernel_cpso];
            ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
            [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
            ENC_BUF(enc, g_tcache.lpips_rendered_nchw, 3);
            ENC_BUF(enc, g_tcache.lpips_gt_nchw, 4);
            ENC_BUF(enc, background, 5); ENC_SCALAR(enc, composite_gt_u32, 6);
            [enc dispatchThreads:MTLSizeMake(lpips_numel, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(ctx->lpips_prepare_nchw_kernel_cpso.maxTotalThreadsPerThreadgroup,
                                                  (NSUInteger)lpips_numel), 1, 1)];
            [enc endEncoding];
        }

        MPSGraphTensorData *inputA = [[[MPSGraphTensorData alloc] initWithMTLBuffer:g_tcache.lpips_rendered_nchw.buffer()
                                                                              shape:shape4(1, 3, img_height, img_width)
                                                                           dataType:MPSDataTypeFloat32] autorelease];
        MPSGraphTensorData *inputB = [[[MPSGraphTensorData alloc] initWithMTLBuffer:g_tcache.lpips_gt_nchw.buffer()
                                                                              shape:shape4(1, 3, img_height, img_width)
                                                                           dataType:MPSDataTypeFloat32] autorelease];
        MPSGraphTensorData *grad = [[[MPSGraphTensorData alloc] initWithMTLBuffer:g_tcache.lpips_grad_nchw.buffer()
                                                                            shape:shape4(1, 3, img_height, img_width)
                                                                         dataType:MPSDataTypeFloat32] autorelease];
        MPSGraphTensorData *lossData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:g_tcache.lpips_loss.buffer()
                                                                                shape:@[@1]
                                                                             dataType:MPSDataTypeFloat32] autorelease];
        NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithCapacity:2];
        feeds[lpips->input_a] = inputA;
        feeds[lpips->input_b] = inputB;
        NSMutableDictionary *results = [NSMutableDictionary dictionaryWithCapacity:2];
        results[lpips->grad_a] = grad;
        results[lpips->loss] = lossData;
        [lpips->graph encodeToCommandBuffer:command_buffer
                                      feeds:feeds
                           targetOperations:nil
                          resultsDictionary:results
                        executionDescriptor:nil];

        {
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            [enc setComputePipelineState:ctx->lpips_apply_grad_kernel_cpso];
            ENC_BUF(enc, g_tcache.lpips_grad_nchw, 0);
            ENC_BUF(enc, g_tcache.lpips_loss, 1);
            [enc setBytes:loss_img_size->data() length:sizeof(*loss_img_size) atIndex:2];
            ENC_SCALAR(enc, lpips_loss_weight, 3);
            ENC_BUF(enc, v_rendered, 4);
            ENC_BUF(enc, loss_sum, 5);
            [enc dispatchThreads:MTLSizeMake(lpips_numel, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(MIN(ctx->lpips_apply_grad_kernel_cpso.maxTotalThreadsPerThreadgroup,
                                                  (NSUInteger)lpips_numel), 1, 1)];
            [enc endEncoding];
        }
    };

    auto encode_rast_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->rasterize_backward_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, background, 8); ENC_BUF(enc, final_Ts, 9);
            ENC_BUF(enc, final_idx, 10); ENC_BUF(enc, v_rendered, 11);
            ENC_BUF(enc, v_xy, 12); ENC_BUF(enc, v_conic, 13);
            ENC_BUF(enc, v_colors_rast, 14); ENC_BUF(enc, v_opacity, 15);
            ENC_BUF(enc, v_refine, 16);
            ENC_BUF(enc, gt_packed, 17); ENC_SCALAR(enc, use_alpha_loss_u32, 18);
            ENC_SCALAR(enc, alpha_loss_grad_scale, 19);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked backward
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            auto bwd_img_sz = std::make_shared<std::array<uint32_t, 2>>(std::array<uint32_t, 2>{img_width, img_height});
            // Phase 1: prefix_T and after_C
            [enc setComputePipelineState:ctx->compute_chunk_prefix_suffix_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, bwd_K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, g_tcache.prefix_T, 5); ENC_BUF(enc, g_tcache.after_C, 6);
            [enc setBytes:bwd_img_sz->data() length:sizeof(*bwd_img_sz) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Phase 2: backward chunked
            [enc setComputePipelineState:ctx->rasterize_backward_chunked_kernel_cpso];
            [enc setBytes:rast_tb->data() length:sizeof(*rast_tb) atIndex:0];
            [enc setBytes:rast_isz->data() length:sizeof(*rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, background, 8); ENC_BUF(enc, final_Ts, 9);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 10);
            ENC_BUF(enc, g_tcache.prefix_T, 11); ENC_BUF(enc, g_tcache.chunk_T, 12);
            ENC_BUF(enc, g_tcache.after_C, 13);
            ENC_BUF(enc, v_rendered, 14);
            ENC_BUF(enc, v_xy, 15); ENC_BUF(enc, v_conic, 16);
            ENC_BUF(enc, v_colors_rast, 17); ENC_BUF(enc, v_opacity, 18);
            ENC_BUF(enc, v_refine, 19);
            ENC_SCALAR(enc, BWD_CHUNK_SIZE, 20); ENC_SCALAR(enc, bwd_K_max, 21);
            ENC_BUF(enc, gt_packed, 22); ENC_SCALAR(enc, use_alpha_loss_u32, 23);
            ENC_SCALAR(enc, alpha_loss_grad_scale, 24);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, bwd_K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        }
    };

    // Packed SH Adam hyperparameters (must match SHAdamParams in .metal)
    struct SHAdamParams {
        float dc_step_size, dc_bc2_sqrt;
        float rest_step_size, rest_bc2_sqrt;
        float beta1, beta2, eps;
        uint32_t reduce_second_moment;
    };
    auto sh_adam_hp = std::make_shared<SHAdamParams>();
    if (num_adam_groups >= 5) {
        sh_adam_hp->dc_step_size = adam_step_sizes[3];
        sh_adam_hp->dc_bc2_sqrt = adam_bc2_sqrts[3];
        sh_adam_hp->rest_step_size = adam_step_sizes[4];
        sh_adam_hp->rest_bc2_sqrt = adam_bc2_sqrts[4];
        sh_adam_hp->beta1 = adam_beta1;
        sh_adam_hp->beta2 = adam_beta2;
        sh_adam_hp->eps = adam_eps;
        sh_adam_hp->reduce_second_moment = reduce_second_moment ? 1u : 0u;
    }

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->project_and_sh_backward_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->project_and_sh_backward_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0); ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr->data() length:sizeof(*proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz->data() length:sizeof(*proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_depth, 12); ENC_BUF(enc, v_conic, 13);
        ENC_BUF(enc, v_mean3d, 14); ENC_BUF(enc, v_scale, 15); ENC_BUF(enc, v_quat, 16);
        ENC_SCALAR(enc, degree, 17); ENC_SCALAR(enc, degrees_to_use, 18);
        [enc setBytes:cam_pos_arr->data() length:sizeof(*cam_pos_arr) atIndex:19];
        ENC_BUF(enc, v_colors_rast, 20);
        // Fused SH backward + Adam: pass params + optimizer state instead of gradient buffers
        [enc setBuffer:adam_params[3].buffer() offset:0 atIndex:21];  // features_dc params
        [enc setBuffer:adam_params[4].buffer() offset:0 atIndex:22];  // features_rest params
        [enc setBuffer:adam_exp_avg[3].buffer() offset:0 atIndex:23]; // dc exp_avg
        [enc setBuffer:adam_exp_avg_sq[3].buffer() offset:0 atIndex:24]; // dc exp_avg_sq
        [enc setBuffer:adam_exp_avg[4].buffer() offset:0 atIndex:25]; // rest exp_avg
        [enc setBuffer:adam_exp_avg_sq[4].buffer() offset:0 atIndex:26]; // rest exp_avg_sq
        [enc setBytes:sh_adam_hp.get() length:sizeof(SHAdamParams) atIndex:27];
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        // Adam for remaining groups (skip 3=featuresDc, 4=featuresRest — fused above)
        if (num_adam_groups > 0) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            for (int g = 0; g < num_adam_groups; ++g) {
                if (g == 3 || g == 4) continue;  // fused into backward kernel
                uint32_t n = adam_params[g].numel();
                if (n == 0) continue;
                NSUInteger atpg = MIN(ctx->fused_adam_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)n);
                [enc setComputePipelineState:ctx->fused_adam_kernel_cpso];
                [enc setBuffer:adam_params[g].buffer() offset:0 atIndex:0];
                [enc setBuffer:(*adam_grads)[g].buffer() offset:0 atIndex:1];
                [enc setBuffer:adam_exp_avg[g].buffer() offset:0 atIndex:2];
                [enc setBuffer:adam_exp_avg_sq[g].buffer() offset:0 atIndex:3];
                ENC_SCALAR(enc, adam_step_sizes[g], 4);
                ENC_SCALAR(enc, adam_beta1, 5);
                ENC_SCALAR(enc, adam_beta2, 6);
                ENC_SCALAR(enc, adam_bc2_sqrts[g], 7);
                ENC_SCALAR(enc, adam_eps, 8);
                ENC_SCALAR(enc, n, 9);
                [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(atpg, 1, 1)];
            }
        }
    };

    // ========================== DISPATCH ==========================

    // Encode accumulate_grad_stats as a lambda (shared by both paths)
    auto encode_grad_stats = [&](id<MTLComputeCommandEncoder> enc) {
        NSUInteger tpg = MIN(ctx->accumulate_grad_stats_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->accumulate_grad_stats_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0);
        ENC_BUF(enc, radii_out, 1);
        ENC_BUF(enc, v_refine, 2);
        ENC_BUF(enc, vis_counts, 3);
        ENC_BUF(enc, xys_grad_norm, 4);
        ENC_BUF(enc, max_2d_size, 5);
        ENC_BUF(enc, aabb, 6);
        ENC_SCALAR(enc, inv_max_dim, 7);
        ENC_SCALAR(enc, inv_width, 8);
        ENC_SCALAR(enc, inv_height, 9);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_pup_hessian = [&](id<MTLComputeCommandEncoder> enc) {
        if (!pup_hessian) return;
        NSUInteger tpg = MIN(ctx->accumulate_pup_hessian_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:ctx->accumulate_pup_hessian_kernel_cpso];
        ENC_SCALAR(enc, num_points, 0);
        ENC_BUF(enc, radii_out, 1);
        ENC_BUF(enc, v_mean3d, 2);
        ENC_BUF(enc, v_scale, 3);
        [enc setBuffer:pup_hessian->buffer() offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    // Blit-zero helper (shared by both paths)
    auto do_blit_zero = [&](id<MTLCommandBuffer> cb) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        // tile_bins written by sort kernel, tile_counts no longer used
        [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
        [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
        [blit fillBuffer:v_xy.buffer() range:NSMakeRange(0, v_xy.nbytes()) value:0];
        [blit fillBuffer:v_conic.buffer() range:NSMakeRange(0, v_conic.nbytes()) value:0];
        [blit fillBuffer:v_colors_rast.buffer() range:NSMakeRange(0, v_colors_rast.nbytes()) value:0];
        [blit fillBuffer:v_opacity.buffer() range:NSMakeRange(0, v_opacity.nbytes()) value:0];
        [blit fillBuffer:v_refine.buffer() range:NSMakeRange(0, v_refine.nbytes()) value:0];
        [blit fillBuffer:v_depth.buffer() range:NSMakeRange(0, v_depth.nbytes()) value:0];
        [blit fillBuffer:v_mean3d.buffer() range:NSMakeRange(0, v_mean3d.nbytes()) value:0];
        [blit fillBuffer:v_scale.buffer() range:NSMakeRange(0, v_scale.nbytes()) value:0];
        [blit fillBuffer:v_quat.buffer() range:NSMakeRange(0, v_quat.nbytes()) value:0];
        // v_features_dc and v_features_rest no longer needed — SH grads fused into Adam
        [blit endEncoding];
    };

    if (!g_profile_stages_checked) {
        g_profile_stages = std::getenv("PROFILE_STAGES") != nullptr;
        g_profile_stages_checked = true;
    }

    if (g_profile_stages && ctx->counterSamplingAvailable) {
        // Per-stage profiling: separate encoders on the SAME command buffer,
        // each with start/end timestamp sampling via the pass descriptor.
        // Metal handles inter-encoder resource hazard tracking automatically.
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        id<MTLCounterSampleBuffer> csb = ctx->counterSampleBuffer;
        double ticksToMs = ctx->ticksToMs;

        // Encode each stage in its own encoder with timestamp bookends
        typedef void (^encode_fn_t)(id<MTLComputeCommandEncoder>);
        struct StageInfo {
            const char* name;
            bool isBlit;  // true = blit encoder, false = compute encoder
        };

        dispatch_sync(ctx->d_queue, ^(){
            // Stage 0: blit_zero (use blit encoder, no timestamp — blit pass descriptors
            // don't support counter sampling the same way, so we just wrap it)
            // Use a compute pass with start/end timestamps for the blit stage
            // Actually, blit must use blit encoder. We'll measure it via a dummy compute
            // encoder with timestamps before and after.

            // -- Blit zero (no direct timestamp, included in first compute stage overhead) --
            do_blit_zero(command_buffer);

            // Stage encoders with timestamps
            // We have 8 compute stages (indices 0-7 in counter sample buffer)
            // Each stage uses sample indices: start = stage*2, end = stage*2+1
            auto make_profiled_encoder = [&](int stage_idx) -> id<MTLComputeCommandEncoder> {
                MTLComputePassDescriptor *passDesc = [MTLComputePassDescriptor computePassDescriptor];
                passDesc.sampleBufferAttachments[0].sampleBuffer = csb;
                passDesc.sampleBufferAttachments[0].startOfEncoderSampleIndex = stage_idx * 2;
                passDesc.sampleBufferAttachments[0].endOfEncoderSampleIndex = stage_idx * 2 + 1;
                return [command_buffer computeCommandEncoderWithDescriptor:passDesc];
            };

            id<MTLComputeCommandEncoder> enc;

            // Stage 1: proj_sh_fwd
            enc = make_profiled_encoder(0);
            encode_proj_sh(enc);
            [enc endEncoding];

            // Stage 2: prefix_sort_pack
            enc = make_profiled_encoder(1);
            encode_prefix_map(enc);
            [enc endEncoding];

            // Stage 3: rast_fwd
            enc = make_profiled_encoder(2);
            encode_rast_fwd(enc);
            [enc endEncoding];

            // Stage 4+5: loss_fwd_bwd (fused)
            enc = make_profiled_encoder(3);
            encode_loss_fwd_bwd(enc);
            [enc endEncoding];
            encode_lpips(ctx->_currentCB);

            // Stage 5: rast_bwd
            enc = make_profiled_encoder(4);
            encode_rast_bwd(enc);
            [enc endEncoding];

            // Stage 6: proj_sh_bwd + Adam
            enc = make_profiled_encoder(5);
            encode_proj_sh_bwd_adam(enc);
            [enc endEncoding];

            // Stage 7: grad_stats
            enc = make_profiled_encoder(6);
            encode_grad_stats(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_pup_hessian(enc);
            [enc endEncoding];
        });

        // Add completion handler to read timestamps after GPU finishes
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            @autoreleasepool {
                NSData *data = [csb resolveCounterRange:NSMakeRange(0, (N_TRAIN_STAGES - 1) * 2)];
                if (!data) return;
                const MTLCounterResultTimestamp *samples =
                    (const MTLCounterResultTimestamp *)[data bytes];

                std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
                for (int i = 0; i < N_TRAIN_STAGES - 1; i++) {
                    uint64_t start = samples[i * 2].timestamp;
                    uint64_t end = samples[i * 2 + 1].timestamp;
                    if (start == MTLCounterErrorValue || end == MTLCounterErrorValue) continue;
                    // stage_idx 0-7 maps to g_train_stage_names[1-8] (skip blit_zero)
                    g_stage_times[i + 1].push_back((double)(end - start) * ticksToMs);
                }
                g_stage_report_count++;

                if (g_stage_report_count % 500 == 0) {
                    fprintf(stderr, "\n  === GPU Stage Profile (n=%d) ===\n", g_stage_report_count);
                    double total_median = 0;
                    for (int i = 1; i < N_TRAIN_STAGES; i++) {
                        auto &v = g_stage_times[i];
                        if (v.empty()) continue;
                        auto sorted = v;
                        std::sort(sorted.begin(), sorted.end());
                        double med = sorted[sorted.size() / 2];
                        double sum = 0;
                        for (auto x : sorted) sum += x;
                        total_median += med;
                        fprintf(stderr, "  %-20s median=%.3fms  mean=%.3fms\n",
                                g_train_stage_names[i], med, sum / sorted.size());
                    }
                    fprintf(stderr, "  %-20s %.3fms\n", "TOTAL (sum medians)", total_median);
                }
            }
        }];

    } else {
        // Production: single encoder for everything
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");

        dispatch_sync(ctx->d_queue, ^(){
            do_blit_zero(command_buffer);

            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            assert(enc && "Failed to create compute command encoder");

            // --- Forward: proj_sh → prefix_map → rast_fwd → loss_fwd ---
            encode_proj_sh(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_prefix_map(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_rast_fwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // --- Fused loss forward + backward ---
            encode_loss_fwd_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (lpips_loss_weight > 0.0f) {
                [enc endEncoding];
                encode_lpips(ctx->_currentCB);
                enc = [ctx->getCommandBuffer() computeCommandEncoder];
                assert(enc && "Failed to create post-LPIPS compute command encoder");
            }
            encode_rast_bwd(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_proj_sh_bwd_adam(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // --- Accumulate grad stats ---
            encode_grad_stats(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_pup_hessian(enc);

            [enc endEncoding];
        });
    }

    // Callers currently use the returned radii only. Reading loss_sum here would
    // force a stale shared-memory read before this command buffer is committed.
    return std::make_tuple(radii_out, 0.0f);
}

// ============================================================================
// GPU-native densification (v34 Phase 3)
// Entire classify → grow → cull → compact pipeline in one compute encoder.
// Returns new num_active after densification.
// ============================================================================
int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float growth_select_fraction, uint32_t growth_seed, int max_splats,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    const float *cull_center, float cull_bounds_thresh, int use_precomputed_flags,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    MTensor &random_samples
) {
    MetalContext* ctx = get_global_context();

    // Worst case: each of N gaussians splits (2 children) + dups (1 copy) = 3N
    int worst_case = 3 * N;
    assert(worst_case <= buf_capacity && "gpu_densify: 3*N exceeds buf_capacity");

    float split_log_scale_factor = std::log(1.0f / std::sqrt(2.0f));

    // Strides for each of the 18 buffers (6 params + 12 optimizer states)
    // Order: means(3), scales(3), quats(4), featuresDc(3), featuresRest(fr_stride), opacities(1)
    int strides[6] = {3, 3, 4, 3, fr_stride, 1};
    int max_stride = fr_stride;  // featuresRest has the largest stride

    // Collect all 18 buffers in order for compact loops (std::array for block capture)
    std::array<MTensor*, 18> all_bufs = {{
        &means_buf, &scales_buf, &quats_buf, &featuresDc_buf, &featuresRest_buf, &opacities_buf,
        &adam_exp_avg_buf[0], &adam_exp_avg_buf[1], &adam_exp_avg_buf[2],
        &adam_exp_avg_buf[3], &adam_exp_avg_buf[4], &adam_exp_avg_buf[5],
        &adam_exp_avg_sq_buf[0], &adam_exp_avg_sq_buf[1], &adam_exp_avg_sq_buf[2],
        &adam_exp_avg_sq_buf[3], &adam_exp_avg_sq_buf[4], &adam_exp_avg_sq_buf[5]
    }};
    std::array<int, 18> all_strides = {{
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1,
        3, 3, 4, 3, fr_stride, 1
    }};

    uint32_t N_u32 = (uint32_t)N;
    uint32_t K = (uint32_t)((N + 1023) / 1024);  // threadgroups for prefix sum on N elements
    int check_screen_int = check_screen;
    int check_huge_int = check_huge;
    float growth_fraction = std::clamp(growth_select_fraction, 0.0f, 1.0f);
    uint32_t growth_seed_u32 = growth_seed;
    int max_new_count = std::max(0, std::min(max_splats, worst_case));
    int use_precomputed_flags_int = use_precomputed_flags;
    std::array<float, 4> cull_center4 = {
        cull_center ? cull_center[0] : 0.0f,
        cull_center ? cull_center[1] : 0.0f,
        cull_center ? cull_center[2] : 0.0f,
        0.0f
    };

    id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
    assert(command_buffer && "Failed to retrieve command buffer reference");

    auto encode_block_totals_prefix = [&](id<MTLComputeCommandEncoder> enc, uint32_t block_count) {
        [enc setComputePipelineState:ctx->prefix_sum_inplace_kernel_cpso];
        ENC_SCALAR(enc, block_count, 0);
        ENC_BUF(enc, block_totals, 1);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    };

    dispatch_sync(ctx->d_queue, ^(){
        id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
        assert(enc && "Failed to create compute command encoder");

        // ---- Stage 1: Classify (split/dup) ----
        if (!use_precomputed_flags_int) {
            NSUInteger tpg = MIN(ctx->densify_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, xys_grad_norm, 1);
            ENC_BUF(enc, vis_counts, 2);
            ENC_BUF(enc, scales_buf, 3);
            ENC_BUF(enc, max_2d_size, 4);
            ENC_SCALAR(enc, half_max_dim, 5);
            ENC_SCALAR(enc, grad_thresh, 6);
            ENC_SCALAR(enc, size_thresh, 7);
            ENC_SCALAR(enc, screen_thresh, 8);
            ENC_SCALAR(enc, check_screen_int, 9);
            ENC_BUF(enc, split_flag, 10);
            ENC_BUF(enc, dup_flag, 11);
            ENC_SCALAR(enc, growth_fraction, 12);
            ENC_SCALAR(enc, growth_seed_u32, 13);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 2: Prefix sum on split_flag → split_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_block_totals_prefix(enc, K);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 3: Prefix sum on dup_flag → dup_prefix ----
        {
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_block_totals_prefix(enc, K);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0); ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 4: Append split children ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_split_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_split_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_flag, 1);
            ENC_BUF(enc, split_prefix, 2);
            ENC_BUF(enc, random_samples, 3);
            ENC_SCALAR(enc, split_log_scale_factor, 4);
            ENC_BUF(enc, means_buf, 5);
            ENC_BUF(enc, scales_buf, 6);
            ENC_BUF(enc, quats_buf, 7);
            ENC_BUF(enc, featuresDc_buf, 8);
            ENC_BUF(enc, featuresRest_buf, 9);
            ENC_BUF(enc, opacities_buf, 10);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 11);
            ENC_BUF(enc, adam_exp_avg_buf[0], 12);
            ENC_BUF(enc, adam_exp_avg_buf[1], 13);
            ENC_BUF(enc, adam_exp_avg_buf[2], 14);
            ENC_BUF(enc, adam_exp_avg_buf[3], 15);
            ENC_BUF(enc, adam_exp_avg_buf[4], 16);
            ENC_BUF(enc, adam_exp_avg_buf[5], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 22);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 23);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 5: Append duplicates ----
        {
            NSUInteger tpg = MIN(ctx->densify_append_dup_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)N);
            [enc setComputePipelineState:ctx->densify_append_dup_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, dup_flag, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_prefix, 3);
            ENC_BUF(enc, means_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, quats_buf, 6);
            ENC_BUF(enc, featuresDc_buf, 7);
            ENC_BUF(enc, featuresRest_buf, 8);
            ENC_BUF(enc, opacities_buf, 9);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 10);
            ENC_BUF(enc, adam_exp_avg_buf[0], 11);
            ENC_BUF(enc, adam_exp_avg_buf[1], 12);
            ENC_BUF(enc, adam_exp_avg_buf[2], 13);
            ENC_BUF(enc, adam_exp_avg_buf[3], 14);
            ENC_BUF(enc, adam_exp_avg_buf[4], 15);
            ENC_BUF(enc, adam_exp_avg_buf[5], 16);
            ENC_BUF(enc, adam_exp_avg_sq_buf[0], 17);
            ENC_BUF(enc, adam_exp_avg_sq_buf[1], 18);
            ENC_BUF(enc, adam_exp_avg_sq_buf[2], 19);
            ENC_BUF(enc, adam_exp_avg_sq_buf[3], 20);
            ENC_BUF(enc, adam_exp_avg_sq_buf[4], 21);
            ENC_BUF(enc, adam_exp_avg_sq_buf[5], 22);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 6: Cull classify (on post-growth population) ----
        // Dispatch worst_case threads; kernel reads N_new from prefix sums
        {
            uint32_t wc = (uint32_t)worst_case;
            NSUInteger tpg = MIN(ctx->densify_cull_classify_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)worst_case);
            [enc setComputePipelineState:ctx->densify_cull_classify_kernel_cpso];
            ENC_SCALAR(enc, N_u32, 0);
            ENC_BUF(enc, split_prefix, 1);
            ENC_BUF(enc, dup_prefix, 2);
            ENC_BUF(enc, split_flag, 3);
            ENC_BUF(enc, opacities_buf, 4);
            ENC_BUF(enc, scales_buf, 5);
            ENC_BUF(enc, max_2d_size, 6);
            ENC_SCALAR(enc, cull_alpha_thresh, 7);
            ENC_SCALAR(enc, cull_scale_thresh, 8);
            ENC_SCALAR(enc, cull_screen_size, 9);
            ENC_SCALAR(enc, check_huge_int, 10);
            ENC_SCALAR(enc, check_screen_int, 11);
            ENC_BUF(enc, keep_flag, 12);
            ENC_SCALAR(enc, max_new_count, 13);
            ENC_BUF(enc, means_buf, 14);
            ENC_BUF(enc, quats_buf, 15);
            ENC_BUF(enc, featuresDc_buf, 16);
            ENC_BUF(enc, featuresRest_buf, 17);
            int fr_stride_val = fr_stride;
            ENC_SCALAR(enc, fr_stride_val, 18);
            [enc setBytes:cull_center4.data() length:sizeof(cull_center4) atIndex:19];
            ENC_SCALAR(enc, cull_bounds_thresh, 20);
            [enc dispatchThreads:MTLSizeMake(worst_case, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 7: Prefix sum on keep_flag → keep_prefix ----
        // Over worst_case elements (includes padding zeros for unused slots)
        uint32_t K2 = (uint32_t)((worst_case + 1023) / 1024);
        {
            uint32_t wc = (uint32_t)worst_case;
            [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, block_totals, 2);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_block_totals_prefix(enc, K2);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            uint32_t wc = (uint32_t)worst_case;
            [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
            ENC_SCALAR(enc, wc, 0); ENC_BUF(enc, keep_flag, 1);
            ENC_BUF(enc, keep_prefix, 2); ENC_BUF(enc, block_totals, 3);
            [enc dispatchThreadgroups:MTLSizeMake(K2, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // ---- Stage 8: Compact scatter (18 buffers → scratch) ----
        // For each buffer: scatter kept elements into compact_scratch
        // Then copy back. We reuse compact_scratch at different offsets per stride.
        for (int b = 0; b < 18; b++) {
            uint32_t wc = (uint32_t)worst_case;
            uint32_t stride_u32 = (uint32_t)all_strides[b];
            uint32_t total_threads = wc * stride_u32;
            NSUInteger tpg = MIN(ctx->compact_scatter_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)total_threads);
            [enc setComputePipelineState:ctx->compact_scatter_kernel_cpso];
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:0];
            ENC_BUF(enc, compact_scratch, 1);
            ENC_BUF(enc, keep_prefix, 2);
            ENC_BUF(enc, keep_flag, 3);
            ENC_SCALAR(enc, wc, 4);
            ENC_SCALAR(enc, stride_u32, 5);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // Copy back from scratch to buffer
            uint32_t last_idx = wc - 1;
            [enc setComputePipelineState:ctx->compact_copy_back_kernel_cpso];
            ENC_BUF(enc, compact_scratch, 0);
            [enc setBuffer:all_bufs[b]->buffer() offset:0 atIndex:1];
            ENC_BUF(enc, keep_prefix, 2);
            ENC_SCALAR(enc, last_idx, 3);
            ENC_SCALAR(enc, stride_u32, 4);
            [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        [enc endEncoding];
    });

    // Single GPU→CPU sync: read new_count from keep_prefix[worst_case - 1]
    ctx->syncCB();
    int new_count = keep_prefix.data<int32_t>()[worst_case - 1];
    return new_count;
}
