#import "bindings.h"
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

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
#import <cstdint>
#import <mutex>
#import <memory>
#import <string>
#import <vector>
#import <cstring>
#import <fstream>
#import <cstdlib>
#import <stdexcept>

// GPU profiling infrastructure.
// PROFILE_GPU=1: per-CB total GPU time via completion handlers.
// PROFILE_STAGES=1: per-stage GPU time via synchronized command buffers.
static bool g_gpu_timing_enabled = false;
static bool g_gpu_timing_checked = false;
static std::mutex g_gpu_timing_mutex;
static std::vector<double> g_gpu_times_ms;

static std::mutex g_forced_sync_mutex;
static uint64_t g_forced_sync_count = 0;

static void record_forced_sync(const char *reason) {
    (void)reason;
    std::lock_guard<std::mutex> lock(g_forced_sync_mutex);
    g_forced_sync_count++;
}

// Per-stage profiling
static bool g_profile_stages = false;
static bool g_profile_stages_checked = false;

// Stage names for training pipeline
static const char* g_train_stage_names[] = {
    "blit_zero", "proj_count", "map_intersects", "radix_sort",
    "tile_edges", "pack_sorted", "rast_fwd", "loss_fwd_bwd",
    "rast_bwd", "proj_sh_bwd_adam", "grad_stats"
};
static constexpr int N_TRAIN_STAGES = 11;

static std::mutex g_stage_timing_mutex;
// Per-stage accumulated times (ms), indexed by stage
static std::vector<double> g_stage_times[N_TRAIN_STAGES];
static int g_stage_report_count = 0;

static int stage_profile_report_interval() {
    static int interval = [] {
        const char *env = std::getenv("PROFILE_STAGES_REPORT_EVERY");
        if (!env) return 50;
        int parsed = std::atoi(env);
        return parsed > 0 ? parsed : 50;
    }();
    return interval;
}

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLLibrary>      metal_library;
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

    // Forward pipeline kernels
    id<MTLComputePipelineState> project_and_sh_forward_kernel_cpso;
    id<MTLComputePipelineState> nd_rasterize_forward_kernel_cpso;
    id<MTLComputePipelineState> copy_int_buffer_kernel_cpso;
    // Brush-style dynamic intersection sorting
    id<MTLComputePipelineState> map_gaussian_to_intersects_kernel_cpso;
    id<MTLComputePipelineState> get_tile_bin_edges_kernel_cpso;
    id<MTLComputePipelineState> pack_sorted_gaussians_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_histogram_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_scan_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_scatter_kernel_cpso;
    id<MTLComputePipelineState> map_gaussian_to_intersects_u32_kernel_cpso;
    id<MTLComputePipelineState> get_tile_bin_edges_u32_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_histogram_u32_kernel_cpso;
    id<MTLComputePipelineState> radix_sort_scatter_u32_kernel_cpso;
    // Legacy tile-local sorting
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
    id<MTLComputePipelineState> rasterize_backward_persplat_kernel_cpso;
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
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> project_sh_forward_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> project_sh_backward_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> loss_l1_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> loss_ssim_h_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> loss_ssim_fused_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> loss_ssim_v_bwd_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> raster_backward_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> raster_backward_persplat_specializations;
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> raster_backward_chunked_specializations;
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
    ctx->metal_library = metal_library;

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
    auto loadWithEmptyConstants = [&](NSString* name) -> id<MTLComputePipelineState> {
        MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
        NSError *functionError = nil;
        id<MTLFunction> fn = [metal_library newFunctionWithName:name
                                                 constantValues:constants
                                                          error:&functionError];
        [constants release];
        if (!fn || functionError) {
            fprintf(stderr, "msplat: kernel not found: %s (%s)\n",
                    [name UTF8String],
                    functionError ? [[functionError description] UTF8String] : "unknown error");
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
    ctx->project_and_sh_forward_kernel_cpso       = loadWithEmptyConstants(@"project_and_sh_forward_kernel");
    ctx->nd_rasterize_forward_kernel_cpso         = load(@"nd_rasterize_forward_kernel");
    ctx->copy_int_buffer_kernel_cpso              = load(@"copy_int_buffer_kernel");
    // Brush-style dynamic intersection sorting
    ctx->map_gaussian_to_intersects_kernel_cpso   = load(@"map_gaussian_to_intersects_kernel");
    ctx->get_tile_bin_edges_kernel_cpso           = load(@"get_tile_bin_edges_kernel");
    ctx->pack_sorted_gaussians_kernel_cpso        = load(@"pack_sorted_gaussians_kernel");
    ctx->radix_sort_histogram_kernel_cpso         = load(@"radix_sort_histogram_kernel");
    ctx->radix_sort_scan_kernel_cpso              = load(@"radix_sort_scan_kernel");
    ctx->radix_sort_scatter_kernel_cpso           = load(@"radix_sort_scatter_kernel");
    ctx->map_gaussian_to_intersects_u32_kernel_cpso = load(@"map_gaussian_to_intersects_u32_kernel");
    ctx->get_tile_bin_edges_u32_kernel_cpso       = load(@"get_tile_bin_edges_u32_kernel");
    ctx->radix_sort_histogram_u32_kernel_cpso     = load(@"radix_sort_histogram_u32_kernel");
    ctx->radix_sort_scatter_u32_kernel_cpso       = load(@"radix_sort_scatter_u32_kernel");
    // Legacy tile-local sorting
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
    ctx->rasterize_backward_chunked_kernel_cpso   = loadWithEmptyConstants(@"rasterize_backward_chunked_kernel");
    ctx->rasterize_backward_persplat_kernel_cpso  = loadWithEmptyConstants(@"rasterize_backward_persplat_kernel");
    ctx->rasterize_backward_kernel_cpso           = loadWithEmptyConstants(@"rasterize_backward_kernel");
    // Separable SSIM loss
    ctx->ssim_h_fwd_kernel_cpso                   = loadWithEmptyConstants(@"ssim_h_fwd_kernel");
    ctx->ssim_v_fwd_kernel_cpso                   = loadWithEmptyConstants(@"ssim_v_fwd_kernel");
    ctx->l1_loss_fwd_bwd_kernel_cpso              = loadWithEmptyConstants(@"l1_loss_fwd_bwd_kernel");
    ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso       = loadWithEmptyConstants(@"ssim_fused_v_fwd_h_bwd_kernel");
    ctx->ssim_v_bwd_kernel_cpso                   = loadWithEmptyConstants(@"ssim_v_bwd_kernel");
    ctx->lpips_prepare_nchw_kernel_cpso           = load(@"lpips_prepare_nchw_kernel");
    ctx->lpips_apply_grad_kernel_cpso             = load(@"lpips_apply_grad_kernel");
    // Backward pipeline
    ctx->project_and_sh_backward_kernel_cpso      = loadWithEmptyConstants(@"project_and_sh_backward_kernel");
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
    auto requireStaticThreadgroupMemoryFits = [&](id<MTLComputePipelineState> pso, NSString *name) {
        if (pso && pso.staticThreadgroupMemoryLength > ctx->device.maxThreadgroupMemoryLength) {
            fprintf(stderr, "msplat: kernel %s uses %lu bytes of static threadgroup memory; device supports %lu\n",
                    [name UTF8String],
                    (unsigned long)pso.staticThreadgroupMemoryLength,
                    (unsigned long)ctx->device.maxThreadgroupMemoryLength);
            pipeline_load_failed = true;
        }
    };
    requireThreadgroupSize(ctx->prefix_sum_kernel_cpso, @"prefix_sum_kernel", 1024);
    requireThreadgroupSize(ctx->prefix_sum_inplace_kernel_cpso, @"prefix_sum_inplace_kernel", 1024);
    requireThreadgroupSize(ctx->block_reduce_kernel_cpso, @"block_reduce_kernel", 1024);
    requireThreadgroupSize(ctx->block_scan_propagate_kernel_cpso, @"block_scan_propagate_kernel", 1024);
    requireStaticThreadgroupMemoryFits(ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso, @"ssim_fused_v_fwd_h_bwd_kernel");

    if (pipeline_load_failed) {
        delete ctx;
        return NULL;
    }

    // PROFILE_STAGES deliberately uses synchronized command buffers. This is
    // slower than production dispatch, but GPUEndTime/GPUStartTime gives stable
    // per-stage timings without timestamp counter unit ambiguity.
    if (std::getenv("PROFILE_STAGES")) {
        g_profile_stages = true;
        g_profile_stages_checked = true;
        fprintf(stderr, "PROFILE_STAGES: synchronized command-buffer profiling enabled\n");
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

static bool should_use_half_sorted_buffers() {
    static const bool enabled = [] {
        const char *mode = std::getenv("MSPLAT_HALF_SORTED_BUFFERS");
        return mode && std::strcmp(mode, "1") == 0;
    }();
    return enabled;
}

static void bind_sorted_half_buffers(id<MTLComputeCommandEncoder> enc,
                                     MTensor &packed_conic,
                                     MTensor &packed_rgb,
                                     MTensor &packed_opacity_comp,
                                     uint32_t use_half_sorted_buffers,
                                     NSUInteger conic_index,
                                     NSUInteger rgb_index,
                                     NSUInteger opacity_comp_index,
                                     NSUInteger flag_index) {
    [enc setBuffer:packed_conic.buffer() offset:0 atIndex:conic_index];
    [enc setBuffer:packed_rgb.buffer() offset:0 atIndex:rgb_index];
    [enc setBuffer:packed_opacity_comp.buffer() offset:0 atIndex:opacity_comp_index];
    [enc setBytes:&use_half_sorted_buffers length:sizeof(use_half_sorted_buffers) atIndex:flag_index];
}

static bool project_sh_specialization_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool can_specialize_project_sh(uint32_t degrees_to_use) {
    return project_sh_specialization_enabled() && degrees_to_use <= 3u;
}

static uint32_t project_sh_specialization_key(uint32_t degrees_to_use,
                                             bool use_mip_splatting,
                                             bool reduce_second_moment) {
    return (degrees_to_use & 0xffu)
        | (use_mip_splatting ? (1u << 8) : 0u)
        | (reduce_second_moment ? (1u << 9) : 0u);
}

static id<MTLComputePipelineState> make_project_sh_specialization(
    MetalContext *ctx,
    NSString *function_name,
    uint32_t degrees_to_use,
    bool use_mip_splatting,
    bool reduce_second_moment,
    bool specialize_reduce_second_moment
) {
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    uint32_t degrees = degrees_to_use;
    bool mip = use_mip_splatting;
    bool reduce = reduce_second_moment;
    [constants setConstantValue:&degrees type:MTLDataTypeUInt atIndex:0];
    [constants setConstantValue:&mip type:MTLDataTypeBool atIndex:1];
    if (specialize_reduce_second_moment) {
        [constants setConstantValue:&reduce type:MTLDataTypeBool atIndex:2];
    }

    NSError *function_error = nil;
    id<MTLFunction> fn = [ctx->metal_library newFunctionWithName:function_name
                                                   constantValues:constants
                                                            error:&function_error];
    [constants release];
    if (!fn || function_error) {
        fprintf(stderr, "msplat: failed to specialize %s: %s\n",
                [function_name UTF8String],
                function_error ? [[function_error description] UTF8String] : "unknown error");
        return nil;
    }

    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&pipeline_error];
    [fn release];
    if (!pso || pipeline_error) {
        fprintf(stderr, "msplat: failed to create specialized pipeline for %s: %s\n",
                [function_name UTF8String],
                pipeline_error ? [[pipeline_error description] UTF8String] : "unknown error");
        return nil;
    }
    return pso;
}

static id<MTLComputePipelineState> project_sh_forward_pipeline(MetalContext *ctx,
                                                               uint32_t degrees_to_use,
                                                               bool use_mip_splatting) {
    if (!can_specialize_project_sh(degrees_to_use)) {
        return ctx->project_and_sh_forward_kernel_cpso;
    }
    uint32_t key = project_sh_specialization_key(degrees_to_use, use_mip_splatting, false);
    auto it = ctx->project_sh_forward_specializations.find(key);
    if (it != ctx->project_sh_forward_specializations.end()) {
        return it->second;
    }
    id<MTLComputePipelineState> pso = make_project_sh_specialization(
        ctx, @"project_and_sh_forward_kernel", degrees_to_use, use_mip_splatting, false, false);
    if (!pso) {
        return ctx->project_and_sh_forward_kernel_cpso;
    }
    ctx->project_sh_forward_specializations.emplace(key, pso);
    return pso;
}

static id<MTLComputePipelineState> project_sh_backward_pipeline(MetalContext *ctx,
                                                                uint32_t degrees_to_use,
                                                                bool reduce_second_moment) {
    if (!can_specialize_project_sh(degrees_to_use)) {
        return ctx->project_and_sh_backward_kernel_cpso;
    }
    uint32_t key = project_sh_specialization_key(degrees_to_use, false, reduce_second_moment);
    auto it = ctx->project_sh_backward_specializations.find(key);
    if (it != ctx->project_sh_backward_specializations.end()) {
        return it->second;
    }
    id<MTLComputePipelineState> pso = make_project_sh_specialization(
        ctx, @"project_and_sh_backward_kernel", degrees_to_use, false,
        reduce_second_moment, true);
    if (!pso) {
        return ctx->project_and_sh_backward_kernel_cpso;
    }
    ctx->project_sh_backward_specializations.emplace(key, pso);
    return pso;
}

static bool loss_specialization_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("MSPLAT_ENABLE_LOSS_SPECIALIZATION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static uint32_t loss_specialization_key(uint32_t composite_gt,
                                        uint32_t use_loss_mask,
                                        uint32_t use_alpha_loss) {
    return (composite_gt ? 1u : 0u)
        | (use_loss_mask ? (1u << 1) : 0u)
        | (use_alpha_loss ? (1u << 2) : 0u);
}

static id<MTLComputePipelineState> make_loss_specialization(
    MetalContext *ctx,
    NSString *function_name,
    uint32_t composite_gt,
    uint32_t use_loss_mask,
    uint32_t use_alpha_loss,
    bool specialize_mask,
    bool specialize_alpha
) {
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    bool composite = composite_gt != 0u;
    bool mask = use_loss_mask != 0u;
    bool alpha = use_alpha_loss != 0u;
    [constants setConstantValue:&composite type:MTLDataTypeBool atIndex:3];
    if (specialize_mask) {
        [constants setConstantValue:&mask type:MTLDataTypeBool atIndex:4];
    }
    if (specialize_alpha) {
        [constants setConstantValue:&alpha type:MTLDataTypeBool atIndex:5];
    }

    NSError *function_error = nil;
    id<MTLFunction> fn = [ctx->metal_library newFunctionWithName:function_name
                                                   constantValues:constants
                                                            error:&function_error];
    [constants release];
    if (!fn || function_error) {
        fprintf(stderr, "msplat: failed to specialize %s: %s\n",
                [function_name UTF8String],
                function_error ? [[function_error description] UTF8String] : "unknown error");
        return nil;
    }

    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&pipeline_error];
    [fn release];
    if (!pso || pipeline_error) {
        fprintf(stderr, "msplat: failed to create specialized pipeline for %s: %s\n",
                [function_name UTF8String],
                pipeline_error ? [[pipeline_error description] UTF8String] : "unknown error");
        return nil;
    }
    return pso;
}

static id<MTLComputePipelineState> loss_pipeline(
    MetalContext *ctx,
    id<MTLComputePipelineState> default_pso,
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> &cache,
    NSString *function_name,
    uint32_t composite_gt,
    uint32_t use_loss_mask,
    uint32_t use_alpha_loss,
    bool specialize_mask,
    bool specialize_alpha
) {
    if (!loss_specialization_enabled()) {
        return default_pso;
    }
    uint32_t key = loss_specialization_key(
        composite_gt,
        specialize_mask ? use_loss_mask : 0u,
        specialize_alpha ? use_alpha_loss : 0u);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    id<MTLComputePipelineState> pso = make_loss_specialization(
        ctx, function_name, composite_gt, use_loss_mask, use_alpha_loss,
        specialize_mask, specialize_alpha);
    if (!pso) {
        return default_pso;
    }
    cache.emplace(key, pso);
    return pso;
}

static bool raster_backward_specialization_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool raster_backward_warp_merge_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

static uint32_t raster_backward_specialization_key(uint32_t use_alpha_loss,
                                                   uint32_t use_half_sorted_buffers,
                                                   bool use_warp_merge) {
    return (use_alpha_loss ? 1u : 0u)
        | (use_half_sorted_buffers ? (1u << 1) : 0u)
        | (use_warp_merge ? (1u << 2) : 0u);
}

static id<MTLComputePipelineState> make_raster_backward_specialization(
    MetalContext *ctx,
    NSString *function_name,
    uint32_t use_alpha_loss,
    uint32_t use_half_sorted_buffers,
    bool use_warp_merge
) {
    MTLFunctionConstantValues *constants = [[MTLFunctionConstantValues alloc] init];
    bool alpha = use_alpha_loss != 0u;
    bool half = use_half_sorted_buffers != 0u;
    [constants setConstantValue:&alpha type:MTLDataTypeBool atIndex:6];
    [constants setConstantValue:&half type:MTLDataTypeBool atIndex:7];
    [constants setConstantValue:&use_warp_merge type:MTLDataTypeBool atIndex:8];

    NSError *function_error = nil;
    id<MTLFunction> fn = [ctx->metal_library newFunctionWithName:function_name
                                                   constantValues:constants
                                                            error:&function_error];
    [constants release];
    if (!fn || function_error) {
        fprintf(stderr, "msplat: failed to specialize %s: %s\n",
                [function_name UTF8String],
                function_error ? [[function_error description] UTF8String] : "unknown error");
        return nil;
    }

    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pso = [ctx->device newComputePipelineStateWithFunction:fn error:&pipeline_error];
    [fn release];
    if (!pso || pipeline_error) {
        fprintf(stderr, "msplat: failed to create specialized pipeline for %s: %s\n",
                [function_name UTF8String],
                pipeline_error ? [[pipeline_error description] UTF8String] : "unknown error");
        return nil;
    }
    return pso;
}

static id<MTLComputePipelineState> raster_backward_pipeline(
    MetalContext *ctx,
    id<MTLComputePipelineState> default_pso,
    std::unordered_map<uint32_t, id<MTLComputePipelineState>> &cache,
    NSString *function_name,
    uint32_t use_alpha_loss,
    uint32_t use_half_sorted_buffers,
    bool use_warp_merge
) {
    if (!raster_backward_specialization_enabled() && !use_warp_merge) {
        return default_pso;
    }
    uint32_t key = raster_backward_specialization_key(use_alpha_loss, use_half_sorted_buffers, use_warp_merge);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    id<MTLComputePipelineState> pso = make_raster_backward_specialization(
        ctx, function_name, use_alpha_loss, use_half_sorted_buffers, use_warp_merge);
    if (!pso) {
        return default_pso;
    }
    cache.emplace(key, pso);
    return pso;
}

static uint32_t prefix_sum_block_count(uint32_t count) {
    return std::max<uint32_t>(1, (count + 1023u) / 1024u);
}

static void encode_int32_prefix_sum(MetalContext *ctx, id<MTLComputeCommandEncoder> enc,
                                    uint32_t count, MTensor &input, MTensor &output,
                                    MTensor &block_totals) {
    if (count <= 1024u) {
        [enc setComputePipelineState:ctx->prefix_sum_kernel_cpso];
        ENC_SCALAR(enc, count, 0);
        ENC_BUF(enc, input, 1);
        ENC_BUF(enc, output, 2);
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        return;
    }

    uint32_t block_count = prefix_sum_block_count(count);
    [enc setComputePipelineState:ctx->block_reduce_kernel_cpso];
    ENC_SCALAR(enc, count, 0);
    ENC_BUF(enc, input, 1);
    ENC_BUF(enc, block_totals, 2);
    [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ctx->prefix_sum_inplace_kernel_cpso];
    ENC_SCALAR(enc, block_count, 0);
    ENC_BUF(enc, block_totals, 1);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    [enc setComputePipelineState:ctx->block_scan_propagate_kernel_cpso];
    ENC_SCALAR(enc, count, 0);
    ENC_BUF(enc, input, 1);
    ENC_BUF(enc, output, 2);
    ENC_BUF(enc, block_totals, 3);
    [enc dispatchThreadgroups:MTLSizeMake(block_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

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

uint64_t msplat_drain_forced_sync_count() {
    std::lock_guard<std::mutex> lock(g_forced_sync_mutex);
    uint64_t count = g_forced_sync_count;
    g_forced_sync_count = 0;
    return count;
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
static constexpr int64_t kFixedTileCapacityMultiplier = 16;

// Cached buffer pool — all intermediate GPU buffers are reused across iterations.
// Sizes only change at densification (every 100 steps); between densifications
// this eliminates all per-iteration GPU allocations.
struct FusedTensorCache {
    int fwd_num_points = 0, capacity = 0, img_height = 0, img_width = 0, num_tiles = 0;
    int bwd_num_points = 0, features_rest_bases = 0;
    bool packed_sorted_half = false;

    // Forward intermediates
    MTensor xys, depths, radii_out, conics, opacity_comp, num_tiles_hit, cum_tiles_hit, colors, aabb;
    MTensor gaussian_ids;
    MTensor gaussian_ids_tmp;
    MTensor isect_ids, isect_ids_tmp;
    MTensor isect_ids_u32, isect_ids_u32_tmp;
    MTensor radix_counts;
    MTensor packed_xy_opac, packed_conic, packed_rgb, packed_opacity_comp;
    MTensor out_img, final_Ts, final_idx;
    MTensor loss_intermediates;
    MTensor ssim_h_buf;
    MTensor tile_bins, loss_sum;
    MTensor debug_tile_bins_before_raster;
    MTensor lpips_rendered_nchw, lpips_gt_nchw, lpips_grad_nchw, lpips_loss;

    // Tile-local sorting buffers
    MTensor tile_offsets, tile_scatter_counters;
    MTensor prealloc_bins;  // legacy tile-local path only

    // Multi-threadgroup prefix sum temp buffer
    MTensor block_totals;

    // Intersection overflow detection
    MTensor overflow_flag;
    int radix_block_capacity = 0;
    bool dynamic_capacity_valid = false;
    int dynamic_capacity_num_points = 0;
    int dynamic_capacity_img_height = 0;
    int dynamic_capacity_img_width = 0;
    int dynamic_capacity_num_tiles = 0;
    bool force_dynamic_intersections = false;
    bool last_sort_path_dynamic = true;

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
                        bool needs_fixed_tile_bins, bool needs_dynamic_u32_keys,
                        bool use_half_sorted_buffers,
                        id<MTLDevice> dev) {
        if (np != fwd_num_points) {
            fwd_num_points = np;
            xys = mtensor_empty(dev, {np, 2}, DType::Float32);
            depths = mtensor_empty(dev, {np}, DType::Float32);
            radii_out = mtensor_empty(dev, {np}, DType::Int32);
            conics = mtensor_empty(dev, {np, 3}, DType::Float32);
            opacity_comp = mtensor_empty(dev, {np}, DType::Float32);
            num_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            cum_tiles_hit = mtensor_empty(dev, {np}, DType::Int32);
            colors = mtensor_empty(dev, {np, 3}, DType::Float32);
            aabb = mtensor_empty(dev, {np, 2}, DType::Float32);
        }
        int64_t prefix_count_capacity = std::max<int64_t>(np, nt);
        int64_t prefix_blocks = std::max<int64_t>(1, (prefix_count_capacity + 1023) / 1024);
        if (!block_totals.defined() || block_totals.size(0) < prefix_blocks) {
            block_totals = mtensor_empty(dev, {prefix_blocks}, DType::Int32);
        }
        if (cap != capacity || use_half_sorted_buffers != packed_sorted_half) {
            capacity = cap;
            packed_sorted_half = use_half_sorted_buffers;
            gaussian_ids = mtensor_empty(dev, {cap}, DType::Int32);
            gaussian_ids_tmp = mtensor_empty(dev, {cap}, DType::Int32);
            isect_ids = mtensor_empty(dev, {cap}, DType::UInt64);
            isect_ids_tmp = mtensor_empty(dev, {cap}, DType::UInt64);
            if (needs_dynamic_u32_keys) {
                isect_ids_u32 = mtensor_empty(dev, {cap}, DType::UInt32);
                isect_ids_u32_tmp = mtensor_empty(dev, {cap}, DType::UInt32);
            } else {
                isect_ids_u32.reset();
                isect_ids_u32_tmp.reset();
            }
            packed_xy_opac = mtensor_empty(dev, {cap, 3}, DType::Float32);
            DType sorted_dtype = use_half_sorted_buffers ? DType::Float16 : DType::Float32;
            packed_conic = mtensor_empty(dev, {cap, 3}, sorted_dtype);
            packed_rgb = mtensor_empty(dev, {cap, 3}, sorted_dtype);
            packed_opacity_comp = mtensor_empty(dev, {cap}, sorted_dtype);

            int blocks = (int)((cap + 255) / 256);
            if (blocks != radix_block_capacity) {
                radix_block_capacity = blocks;
                radix_counts = mtensor_empty(dev, {(int64_t)blocks * 256}, DType::UInt32);
            }
        }
        if (needs_dynamic_u32_keys && (!isect_ids_u32.defined() || !isect_ids_u32_tmp.defined())) {
            isect_ids_u32 = mtensor_empty(dev, {cap}, DType::UInt32);
            isect_ids_u32_tmp = mtensor_empty(dev, {cap}, DType::UInt32);
        } else if (!needs_dynamic_u32_keys && (isect_ids_u32.defined() || isect_ids_u32_tmp.defined())) {
            isect_ids_u32.reset();
            isect_ids_u32_tmp.reset();
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
            prealloc_bins.reset();
        }
        if (needs_fixed_tile_bins && !prealloc_bins.defined()) {
            prealloc_bins = mtensor_empty(dev, {(int64_t)nt * kMaxTileElems}, DType::UInt64);
        }
        if (!loss_sum.defined()) {
            loss_sum = mtensor_empty(dev, {1}, DType::Float32);
        }
        if (!overflow_flag.defined()) {
            overflow_flag = mtensor_empty(dev, {1}, DType::Int32);
        }
    }

    void ensure_backward_debug(int nt, id<MTLDevice> dev) {
        if (!debug_tile_bins_before_raster.defined() || debug_tile_bins_before_raster.size(0) != nt) {
            debug_tile_bins_before_raster = mtensor_empty(dev, {nt, 2}, DType::Int32);
        }
    }

    bool has_dynamic_capacity(int np, int ih, int iw, int nt) const {
        return dynamic_capacity_valid
            && np == dynamic_capacity_num_points
            && ih == dynamic_capacity_img_height
            && iw == dynamic_capacity_img_width
            && nt == dynamic_capacity_num_tiles
            && capacity > 0;
    }

    void mark_dynamic_capacity(int np, int ih, int iw, int nt) {
        dynamic_capacity_valid = true;
        dynamic_capacity_num_points = np;
        dynamic_capacity_img_height = ih;
        dynamic_capacity_img_width = iw;
        dynamic_capacity_num_tiles = nt;
    }

    void invalidate_dynamic_capacity() {
        dynamic_capacity_valid = false;
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

static uint32_t ceil_log2_u32(uint32_t v) {
    if (v <= 1) return 0;
    uint32_t bits = 0;
    --v;
    while (v > 0) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

static uint32_t radix_pass_count_for_tiles(int num_tiles) {
    uint32_t tile_bits = ceil_log2_u32((uint32_t)std::max(1, num_tiles));
    uint32_t passes = (32u + tile_bits + 7u) / 8u;
    return std::max(2u, std::min(8u, passes));
}

enum class IntersectionKeyBitsMode {
    Force64,
    Force32,
    Auto32WhenSafe,
};

static IntersectionKeyBitsMode intersection_key_bits_mode() {
    static const IntersectionKeyBitsMode mode = [] {
        const char *mode = std::getenv("MSPLAT_INTERSECTION_KEY_BITS");
        if (!mode || std::strcmp(mode, "auto") == 0) {
            return IntersectionKeyBitsMode::Auto32WhenSafe;
        }
        if (std::strcmp(mode, "64") == 0) {
            return IntersectionKeyBitsMode::Force64;
        }
        if (std::strcmp(mode, "32") == 0) {
            return IntersectionKeyBitsMode::Force32;
        }
        fprintf(stderr, "WARNING: unknown MSPLAT_INTERSECTION_KEY_BITS=%s; using auto keys.\n", mode);
        return IntersectionKeyBitsMode::Auto32WhenSafe;
    }();
    return mode;
}

static bool should_use_32_bit_intersection_keys(int num_tiles) {
    IntersectionKeyBitsMode mode = intersection_key_bits_mode();
    if (mode == IntersectionKeyBitsMode::Force64) {
        return false;
    }
    if (num_tiles <= 65536) {
        return true;
    }
    if (mode == IntersectionKeyBitsMode::Auto32WhenSafe) {
        return false;
    }
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "WARNING: MSPLAT_INTERSECTION_KEY_BITS=32 requested, "
                "but %d tiles require 64-bit intersection keys.\n", num_tiles);
        warned = true;
    }
    return false;
}

static int64_t read_dynamic_intersection_count(const MTensor &cum_tiles_hit, int num_points) {
    if (num_points <= 0) return 0;
    int32_t count = cum_tiles_hit.data<int32_t>()[num_points - 1];
    return std::max<int64_t>(0, count);
}

static int64_t padded_dynamic_intersection_capacity(int64_t exact_count) {
    if (exact_count <= 0) return 1;
    int64_t padding = std::max<int64_t>(4096, exact_count / 2);
    return exact_count + padding;
}

static bool should_use_dynamic_intersections(unsigned img_width, unsigned img_height,
                                             int num_tiles, bool force_dynamic) {
    const char *mode = std::getenv("MSPLAT_INTERSECTION_SORT");
    if (mode) {
        if (std::strcmp(mode, "fixed") == 0) return false;
        if (std::strcmp(mode, "dynamic") == 0) return true;
    }
    if (force_dynamic) return true;
    return std::max(img_width, img_height) > 2560 || num_tiles > 25000;
}

static bool should_use_persplat_backward(unsigned img_width, unsigned img_height, int num_tiles) {
    const char *mode = std::getenv("MSPLAT_BACKWARD_RASTERIZER");
    if (!mode || std::strcmp(mode, "auto") == 0) {
        return std::max(img_width, img_height) > 2560 || num_tiles > 25000;
    }
    if (std::strcmp(mode, "persplat") == 0 || std::strcmp(mode, "brush") == 0) return true;
    if (std::strcmp(mode, "pixel") == 0 || std::strcmp(mode, "perpixel") == 0
        || std::strcmp(mode, "chunked") == 0) {
        return false;
    }
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "WARNING: unknown MSPLAT_BACKWARD_RASTERIZER=%s; using auto.\n", mode);
        warned = true;
    }
    return false;
}

static bool should_collect_backward_debug() {
    static bool checked = false;
    static bool enabled = false;
    if (!checked) {
        enabled = std::getenv("MSPLAT_BACKWARD_DEBUG") != nullptr;
        checked = true;
    }
    return enabled;
}

static int backward_debug_report_interval() {
    static bool checked = false;
    static int interval = 100;
    if (!checked) {
        if (const char *env = std::getenv("MSPLAT_BACKWARD_DEBUG_INTERVAL")) {
            interval = std::max(1, std::atoi(env));
        }
        checked = true;
    }
    return interval;
}

struct BackwardDebugSample {
    double pre_avg = 0.0;
    double post_avg = 0.0;
    double pre_median = 0.0;
    double post_median = 0.0;
    int pre_max = 0;
    int post_max = 0;
    uint64_t pixel_warp_atomic_groups = 0;
    uint64_t tile_splat_atomic_groups = 0;
    uint64_t replay_active_pairs = 0;
    uint64_t replay_diagonal_steps = 0;
    uint64_t tightened_pairs_skipped = 0;
    uint64_t saturated_pixels = 0;
};

struct BackwardDebugAccum {
    uint64_t samples = 0;
    double pre_avg_sum = 0.0;
    double post_avg_sum = 0.0;
    double pre_median_sum = 0.0;
    double post_median_sum = 0.0;
    double pre_max_sum = 0.0;
    double post_max_sum = 0.0;
    uint64_t pixel_warp_atomic_groups = 0;
    uint64_t tile_splat_atomic_groups = 0;
    uint64_t replay_active_pairs = 0;
    uint64_t replay_diagonal_steps = 0;
    uint64_t tightened_pairs_skipped = 0;
    uint64_t saturated_pixels = 0;
};

static std::mutex g_backward_debug_mutex;
static BackwardDebugAccum g_backward_debug_accum;

static double median_from_sorted(const std::vector<int> &values) {
    if (values.empty()) return 0.0;
    size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) return (double)values[mid];
    return 0.5 * ((double)values[mid - 1] + (double)values[mid]);
}

static BackwardDebugSample make_backward_debug_sample(
    const int32_t *pre_bins,
    const int32_t *post_bins,
    const float *final_Ts,
    int tile_bounds_x,
    int tile_bounds_y,
    int img_width,
    int img_height
) {
    BackwardDebugSample sample;
    const int num_tiles = tile_bounds_x * tile_bounds_y;
    std::vector<int> pre_lengths;
    std::vector<int> post_lengths;
    pre_lengths.reserve(num_tiles);
    post_lengths.reserve(num_tiles);

    uint64_t pre_sum = 0;
    uint64_t post_sum = 0;

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        int start = pre_bins[2 * tile_id];
        int pre_end = std::max(start, pre_bins[2 * tile_id + 1]);
        int post_end = std::max(start, post_bins[2 * tile_id + 1]);
        int pre_len = std::max(0, pre_end - start);
        int post_len = std::max(0, post_end - start);
        int tile_x = tile_id % tile_bounds_x;
        int tile_y = tile_id / tile_bounds_x;
        int pixels_x = std::max(0, std::min(BLOCK_X, img_width - tile_x * BLOCK_X));
        int pixels_y = std::max(0, std::min(BLOCK_Y, img_height - tile_y * BLOCK_Y));
        uint64_t pixels = (uint64_t)pixels_x * (uint64_t)pixels_y;

        pre_lengths.push_back(pre_len);
        post_lengths.push_back(post_len);
        pre_sum += (uint64_t)pre_len;
        post_sum += (uint64_t)post_len;
        sample.pre_max = std::max(sample.pre_max, pre_len);
        sample.post_max = std::max(sample.post_max, post_len);
        uint64_t pixel_warps = (pixels + 31u) / 32u;
        sample.pixel_warp_atomic_groups += (uint64_t)post_len * pixel_warps;
        sample.tile_splat_atomic_groups += (uint64_t)post_len;
        sample.replay_active_pairs += (uint64_t)post_len * pixels;
        sample.tightened_pairs_skipped += (uint64_t)std::max(0, pre_len - post_len) * pixels;
        for (int offset = 0; offset < post_len; offset += BLOCK_SIZE) {
            int batch = std::min(BLOCK_SIZE, post_len - offset);
            sample.replay_diagonal_steps += (uint64_t)(batch + BLOCK_SIZE - 1);
        }
    }

    std::sort(pre_lengths.begin(), pre_lengths.end());
    std::sort(post_lengths.begin(), post_lengths.end());
    if (num_tiles > 0) {
        sample.pre_avg = (double)pre_sum / (double)num_tiles;
        sample.post_avg = (double)post_sum / (double)num_tiles;
    }
    sample.pre_median = median_from_sorted(pre_lengths);
    sample.post_median = median_from_sorted(post_lengths);

    const uint64_t num_pixels = (uint64_t)img_width * (uint64_t)img_height;
    for (uint64_t i = 0; i < num_pixels; ++i) {
        if (final_Ts[i] <= 1e-4f) {
            sample.saturated_pixels++;
        }
    }
    return sample;
}

static void record_backward_debug_sample(const BackwardDebugSample &sample, bool persplat_enabled) {
    std::lock_guard<std::mutex> lock(g_backward_debug_mutex);
    auto &acc = g_backward_debug_accum;
    acc.samples++;
    acc.pre_avg_sum += sample.pre_avg;
    acc.post_avg_sum += sample.post_avg;
    acc.pre_median_sum += sample.pre_median;
    acc.post_median_sum += sample.post_median;
    acc.pre_max_sum += sample.pre_max;
    acc.post_max_sum += sample.post_max;
    acc.pixel_warp_atomic_groups += sample.pixel_warp_atomic_groups;
    acc.tile_splat_atomic_groups += sample.tile_splat_atomic_groups;
    acc.replay_active_pairs += sample.replay_active_pairs;
    acc.replay_diagonal_steps += sample.replay_diagonal_steps;
    acc.tightened_pairs_skipped += sample.tightened_pairs_skipped;
    acc.saturated_pixels += sample.saturated_pixels;

    int interval = backward_debug_report_interval();
    if ((acc.samples % (uint64_t)interval) != 0u) return;

    double n = (double)acc.samples;
    double pixel_groups = (double)acc.pixel_warp_atomic_groups / n;
    double tile_groups = (double)acc.tile_splat_atomic_groups / n;
    double max_merge_pct = pixel_groups > 0.0 ? 100.0 * (1.0 - tile_groups / pixel_groups) : 0.0;
    fprintf(stderr,
            "\n  === Backward Raster Debug (n=%llu, mode=%s) ===\n"
            "  tile splats: avg %.1f -> %.1f, median %.1f -> %.1f, max %.1f -> %.1f\n"
            "  pixel atomic estimate: warp_groups %.1fM/sample, tile_merge_floor %.1fM/sample, max_reduction %.1f%%\n"
            "  persplat replay estimate: active_pairs %.1fM/sample, diagonal_steps %.1fM/sample, tightened_skip %.1fM/sample\n"
            "  saturated pixels: %.1f/sample\n",
            (unsigned long long)acc.samples,
            persplat_enabled ? "persplat" : "pixel",
            acc.pre_avg_sum / n, acc.post_avg_sum / n,
            acc.pre_median_sum / n, acc.post_median_sum / n,
            acc.pre_max_sum / n, acc.post_max_sum / n,
            pixel_groups / 1e6, tile_groups / 1e6, max_merge_pct,
            ((double)acc.replay_active_pairs / n) / 1e6,
            ((double)acc.replay_diagonal_steps / n) / 1e6,
            ((double)acc.tightened_pairs_skipped / n) / 1e6,
            (double)acc.saturated_pixels / n);
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

    // --- Overflow check: detect mismatched dynamic intersection counts ---
    // Only warn once; the dynamic path should size buffers to the exact GPU count.
    static bool overflow_warned = false;
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        record_forced_sync("overflow-check");
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            if (g_tcache.last_sort_path_dynamic) {
                g_tcache.invalidate_dynamic_capacity();
            } else {
                g_tcache.force_dynamic_intersections = true;
            }
            if (!overflow_warned) {
                fprintf(stderr, "WARNING: tile intersection overflow. "
                        "Switching to dynamic gaussian-tile intersections.\n");
                overflow_warned = true;
            }
        }
    }
    bool use_dynamic_intersections = should_use_dynamic_intersections(
        img_width, img_height, num_tiles, g_tcache.force_dynamic_intersections);
    bool use_dynamic_u32_keys = use_dynamic_intersections
        && should_use_32_bit_intersection_keys(num_tiles);
    g_tcache.last_sort_path_dynamic = use_dynamic_intersections;
    int64_t capacity = use_dynamic_intersections
        ? std::max<int64_t>(1, g_tcache.capacity)
        : (int64_t)num_points * kFixedTileCapacityMultiplier;
    uint32_t channels = 3;
    uint32_t use_mip_splatting_u32 = use_mip_splatting ? 1u : 0u;
    uint32_t use_half_sorted_buffers_u32 = should_use_half_sorted_buffers() ? 1u : 0u;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                            compute_loss, false, !use_dynamic_intersections,
                            use_dynamic_u32_keys, use_half_sorted_buffers_u32 != 0u,
                            ctx->device);
    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &opacity_comp = g_tcache.opacity_comp;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &cum_tiles_hit = g_tcache.cum_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &gaussian_ids_tmp = g_tcache.gaussian_ids_tmp;
    MTensor &isect_ids = g_tcache.isect_ids;
    MTensor &isect_ids_tmp = g_tcache.isect_ids_tmp;
    MTensor &isect_ids_u32 = g_tcache.isect_ids_u32;
    MTensor &isect_ids_u32_tmp = g_tcache.isect_ids_u32_tmp;
    MTensor &radix_counts = g_tcache.radix_counts;
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

    std::array<uint32_t, 2> loss_img_size = {img_width, img_height};
    uint32_t composite_gt_u32 = 0;

    // --- Constants copied into Metal encoders via setBytes ---
    std::array<float, 4> proj_intrins = {fx, fy, cx, cy};
    std::array<uint32_t, 2> proj_img_size = {img_width, img_height};
    std::array<uint32_t, 4> tile_bounds_arr = {
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    };
    std::array<float, 4> cam_pos_arr = {cam_pos[0], cam_pos[1], cam_pos[2], 0.0f};
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    std::array<uint32_t, 4> img_size_dim3 = {img_width, img_height, 1, 0xDEAD};
    std::array<int32_t, 2> block_size_dim2 = {RAST_BLOCK_X, RAST_BLOCK_Y};

    // Periodic diagnostic: print key dimensions for roofline analysis
    static int diag_count = 0;
    diag_count++;
    if (std::getenv("BENCHMARK") && (diag_count == 100 || diag_count == 500 || diag_count == 1500)) {
            fprintf(stderr, "\n=== Roofline Dimensions (iter %d) ===\n", diag_count);
            fprintf(stderr, "  num_points:     %d\n", num_points);
            fprintf(stderr, "  intersections:  %lld\n", (long long)capacity);
            fprintf(stderr, "  img:            %u x %u = %u pixels\n", img_width, img_height, img_width * img_height);
            fprintf(stderr, "  tiles:          %d x %d = %d\n", tile_bounds_x, tile_bounds_y, num_tiles);
            fprintf(stderr, "  SH degree:      %u (bases: %u)\n", degree, (degree + 1) * (degree + 1));
            fprintf(stderr, "  features_rest:  [%lld x %lld x %lld]\n",
                (long long)features_rest.size(0), (long long)features_rest.size(1), (long long)features_rest.size(2));
            fprintf(stderr, "  sort:           dynamic global radix\n");
            fprintf(stderr, "  sort keys:      %u-bit\n", use_dynamic_u32_keys ? 32u : 64u);
            fprintf(stderr, "  sort buffer:    %.1f MB (keys)\n",
                    (double)capacity * (use_dynamic_u32_keys ? 4.0 : 8.0) / 1e6);
            fprintf(stderr, "  opacities:      [%lld]\n", (long long)opacities.size(0));
            fprintf(stderr, "===========================\n\n");
    }

    // Helper lambdas to encode each stage onto a given encoder
    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        id<MTLComputePipelineState> pso = project_sh_forward_pipeline(
            ctx, (uint32_t)degrees_to_use, use_mip_splatting_u32 != 0u);
        NSUInteger tpg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:pso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins.data() length:sizeof(proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size.data() length:sizeof(proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr.data() length:sizeof(cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, opacity_comp, 23); ENC_SCALAR(enc, use_mip_splatting_u32, 24);
        ENC_BUF(enc, opacities, 25);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_count_prefix = [&](id<MTLComputeCommandEncoder> enc) {
        encode_int32_prefix_sum(ctx, enc, num_points_u32, num_tiles_hit, cum_tiles_hit, g_tcache.block_totals);
    };

    MTensor *sorted_isect_ids = &isect_ids;
    MTensor *sorted_isect_ids_u32 = &isect_ids_u32;
    MTensor *sorted_gaussian_ids = &gaussian_ids;

    auto encode_map_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        id<MTLComputePipelineState> pso = use_dynamic_u32_keys
            ? ctx->map_gaussian_to_intersects_u32_kernel_cpso
            : ctx->map_gaussian_to_intersects_kernel_cpso;
        NSUInteger tpg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:pso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, depths, 2);
        ENC_BUF(enc, radii_out, 3);
        ENC_BUF(enc, cum_tiles_hit, 4);
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:5];
        ENC_SCALAR(enc, capacity_u32, 6);
        if (use_dynamic_u32_keys) {
            ENC_BUF(enc, isect_ids_u32, 7);
        } else {
            ENC_BUF(enc, isect_ids, 7);
        }
        ENC_BUF(enc, gaussian_ids, 8);
        ENC_BUF(enc, aabb, 9);
        ENC_BUF(enc, g_tcache.overflow_flag, 10);
        ENC_BUF(enc, conics, 11);
        ENC_BUF(enc, opacities, 12);
        ENC_BUF(enc, opacity_comp, 13);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_radix_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_blocks = std::max<uint32_t>(1, (capacity_u32 + 255u) / 256u);
        uint32_t radix_passes = use_dynamic_u32_keys ? 4u : radix_pass_count_for_tiles(num_tiles);
        for (uint32_t pass = 0; pass < radix_passes; ++pass) {
            uint32_t shift = pass * 8u;
            bool even_pass = (pass % 2u) == 0u;
            MTensor &keys_in = use_dynamic_u32_keys
                ? (even_pass ? isect_ids_u32 : isect_ids_u32_tmp)
                : (even_pass ? isect_ids : isect_ids_tmp);
            MTensor &vals_in = even_pass ? gaussian_ids : gaussian_ids_tmp;
            MTensor &keys_out = use_dynamic_u32_keys
                ? (even_pass ? isect_ids_u32_tmp : isect_ids_u32)
                : (even_pass ? isect_ids_tmp : isect_ids);
            MTensor &vals_out = even_pass ? gaussian_ids_tmp : gaussian_ids;

            [enc setComputePipelineState:(use_dynamic_u32_keys
                ? ctx->radix_sort_histogram_u32_kernel_cpso
                : ctx->radix_sort_histogram_kernel_cpso)];
            ENC_SCALAR(enc, capacity_u32, 0);
            ENC_BUF(enc, keys_in, 1);
            ENC_BUF(enc, radix_counts, 2);
            ENC_SCALAR(enc, shift, 3);
            ENC_BUF(enc, cum_tiles_hit, 4);
            ENC_SCALAR(enc, num_points_u32, 5);
            [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->radix_sort_scan_kernel_cpso];
            ENC_BUF(enc, radix_counts, 0);
            ENC_SCALAR(enc, num_blocks, 1);
            ENC_BUF(enc, cum_tiles_hit, 2);
            ENC_SCALAR(enc, capacity_u32, 3);
            ENC_SCALAR(enc, num_points_u32, 4);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(use_dynamic_u32_keys
                ? ctx->radix_sort_scatter_u32_kernel_cpso
                : ctx->radix_sort_scatter_kernel_cpso)];
            ENC_SCALAR(enc, capacity_u32, 0);
            ENC_BUF(enc, keys_in, 1);
            ENC_BUF(enc, vals_in, 2);
            ENC_BUF(enc, keys_out, 3);
            ENC_BUF(enc, vals_out, 4);
            ENC_BUF(enc, radix_counts, 5);
            ENC_SCALAR(enc, shift, 6);
            ENC_BUF(enc, cum_tiles_hit, 7);
            ENC_SCALAR(enc, num_points_u32, 8);
            [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        // Publish the final ping-pong buffers for tile edge and pack consumers.
        if ((radix_passes & 1u) != 0u) {
            sorted_isect_ids = &isect_ids_tmp;
            sorted_isect_ids_u32 = &isect_ids_u32_tmp;
            sorted_gaussian_ids = &gaussian_ids_tmp;
        } else {
            sorted_isect_ids = &isect_ids;
            sorted_isect_ids_u32 = &isect_ids_u32;
            sorted_gaussian_ids = &gaussian_ids;
        }
    };

    auto encode_tile_edges_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:(use_dynamic_u32_keys
            ? ctx->get_tile_bin_edges_u32_kernel_cpso
            : ctx->get_tile_bin_edges_kernel_cpso)];
        ENC_SCALAR(enc, capacity_u32, 0);
        if (use_dynamic_u32_keys) {
            [enc setBuffer:(*sorted_isect_ids_u32).buffer() offset:0 atIndex:1];
        } else {
            [enc setBuffer:(*sorted_isect_ids).buffer() offset:0 atIndex:1];
        }
        ENC_BUF(enc, tile_bins, 2);
        ENC_BUF(enc, cum_tiles_hit, 3);
        ENC_SCALAR(enc, num_points_u32, 4);
        [enc dispatchThreads:MTLSizeMake(capacity_u32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto encode_pack_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        [enc setBuffer:(*sorted_gaussian_ids).buffer() offset:0 atIndex:0];
        ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3);
        ENC_BUF(enc, opacities, 4);
        ENC_BUF(enc, packed_xy_opac, 5);
        ENC_BUF(enc, packed_conic, 6);
        ENC_BUF(enc, packed_rgb, 7);
        ENC_BUF(enc, opacity_comp, 8);
        ENC_BUF(enc, packed_opacity_comp, 9);
        ENC_SCALAR(enc, capacity_u32, 10);
        ENC_BUF(enc, cum_tiles_hit, 11);
        ENC_SCALAR(enc, num_points_u32, 12);
        bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                 use_half_sorted_buffers_u32, 13, 14, 15, 16);
        [enc dispatchThreads:MTLSizeMake(capacity_u32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        encode_map_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_radix_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_tile_edges_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_pack_dynamic(enc);
    };

    auto encode_prefix_map_fixed = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        {
            NSUInteger tpg = MIN(ctx->scatter_to_prealloc_bins_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_prealloc_bins_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0);
            ENC_BUF(enc, xys, 1);
            ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3);
            ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 6);
            ENC_BUF(enc, g_tcache.prealloc_bins, 7);
            ENC_BUF(enc, g_tcache.overflow_flag, 8);
            ENC_BUF(enc, conics, 9);
            ENC_BUF(enc, opacities, 10);
            ENC_BUF(enc, opacity_comp, 11);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            encode_int32_prefix_sum(ctx, enc, num_tiles_u32, g_tcache.tile_scatter_counters,
                                    g_tcache.tile_offsets, g_tcache.block_totals);
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 1);
            ENC_BUF(enc, g_tcache.prealloc_bins, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5);
            ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7);
            ENC_BUF(enc, opacities, 8);
            ENC_BUF(enc, opacity_comp, 9);
            ENC_BUF(enc, packed_xy_opac, 10);
            ENC_BUF(enc, packed_conic, 11);
            ENC_BUF(enc, packed_rgb, 12);
            ENC_BUF(enc, packed_opacity_comp, 13);
            ENC_BUF(enc, tile_bins, 14);
            ENC_SCALAR(enc, capacity_u32, 15);
            ENC_BUF(enc, g_tcache.overflow_flag, 16);
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 17, 18, 19, 20);
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
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3.data() length:sizeof(img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, packed_opacity_comp, 7);
        ENC_BUF(enc, final_Ts, 8); ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, out_img, 10);
        ENC_BUF(enc, background, 11);
        [enc setBytes:block_size_dim2.data() length:sizeof(block_size_dim2) atIndex:12];
        bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                 use_half_sorted_buffers_u32, 13, 14, 15, 16);
        [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:tg_size];
    };

    auto encode_rast_fwd_chunked = [&](id<MTLComputeCommandEncoder> enc) {
        // Phase 1: dispatch forward chunked kernel — grid (tile_x, tile_y, K_max)
        uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
        uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
        uint32_t num_pix = img_width * img_height;
        std::array<uint32_t, 2> img_sz_2 = {img_width, img_height};
        MTLSize chunked_tg = MTLSizeMake(tile_x, tile_y, K_max);
        MTLSize tg_size = MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1);
        [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:0];
        [enc setBytes:img_size_dim3.data() length:sizeof(img_size_dim3) atIndex:1];
        ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
        ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
        ENC_BUF(enc, packed_opacity_comp, 7);
        ENC_BUF(enc, g_tcache.chunk_T, 8); ENC_BUF(enc, g_tcache.chunk_C, 9); ENC_BUF(enc, g_tcache.chunk_final_idx, 10);
        ENC_SCALAR(enc, CHUNK_SIZE, 11); ENC_SCALAR(enc, K_max, 12);
        [enc setBytes:block_size_dim2.data() length:sizeof(block_size_dim2) atIndex:13];
        bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                 use_half_sorted_buffers_u32, 14, 15, 16, 17);
        [enc dispatchThreadgroups:chunked_tg threadsPerThreadgroup:tg_size];

        // Phase 2: merge kernel — one thread per pixel
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        NSUInteger merge_tpg = 256;
        [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
        ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
        ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
        ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
        ENC_BUF(enc, background, 8);
        [enc setBytes:img_sz_2.data() length:sizeof(img_sz_2) atIndex:9];
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
        [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        ENC_BUF(enc, background, 4); ENC_SCALAR(enc, composite_gt_u32, 5);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];

        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        // Pass 2: vertical convolution + SSIM/L1 + loss reduction
        [enc setComputePipelineState:ctx->ssim_v_fwd_kernel_cpso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4);
        ENC_BUF(enc, loss_intermediates, 5); ENC_BUF(enc, loss_sum, 6);
        ENC_BUF(enc, background, 7); ENC_SCALAR(enc, composite_gt_u32, 8);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
    };

    bool did_dynamic_count_prepass = false;
    if (use_dynamic_intersections
        && (!compute_loss || !g_tcache.has_dynamic_capacity(num_points, img_height, img_width, num_tiles))) {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");
        dispatch_sync(ctx->d_queue, ^(){
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");
            encode_proj_sh(encoder);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_count_prefix(encoder);
            [encoder endEncoding];
        });
        ctx->syncCB();
        int64_t exact_intersections = read_dynamic_intersection_count(cum_tiles_hit, num_points);
        capacity = padded_dynamic_intersection_capacity(exact_intersections);
        capacity_u32 = (uint32_t)capacity;
        g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                                compute_loss, false, false, use_dynamic_u32_keys,
                                use_half_sorted_buffers_u32 != 0u, ctx->device);
        g_tcache.mark_dynamic_capacity(num_points, img_height, img_width, num_tiles);
        did_dynamic_count_prepass = true;
    } else {
        capacity = use_dynamic_intersections
            ? std::max<int64_t>(1, g_tcache.capacity)
            : (int64_t)num_points * kFixedTileCapacityMultiplier;
        capacity_u32 = (uint32_t)capacity;
    }

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
            [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
            [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
            [blit fillBuffer:tile_bins.buffer() range:NSMakeRange(0, tile_bins.nbytes()) value:0];
            if (!use_dynamic_intersections) {
                [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
            }
            [blit endEncoding];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            assert(encoder && "Failed to create compute command encoder");

            if (use_dynamic_intersections) {
                if (!did_dynamic_count_prepass) {
                    encode_proj_sh(encoder);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    encode_count_prefix(encoder);
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                encode_prefix_map(encoder);
            } else {
                encode_proj_sh(encoder);
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_prefix_map_fixed(encoder);
            }
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

    // --- Overflow check: detect mismatched dynamic intersection counts ---
    // Only warn once; the dynamic path should size buffers to the exact GPU count.
    static bool overflow_warned = false;
    static int iter_count_oc = 0;
    iter_count_oc++;
    bool num_points_changed = (num_points != g_tcache.fwd_num_points && g_tcache.fwd_num_points > 0);
    if (g_tcache.overflow_flag.defined() && g_tcache.fwd_num_points > 0
        && (num_points_changed || (iter_count_oc % 100) == 1)) {
        record_forced_sync("overflow-check");
        ctx->syncCB();
        int32_t flag_val = *g_tcache.overflow_flag.data<int32_t>();
        if (flag_val > 0) {
            if (g_tcache.last_sort_path_dynamic) {
                g_tcache.invalidate_dynamic_capacity();
            } else {
                g_tcache.force_dynamic_intersections = true;
            }
            if (!overflow_warned) {
                fprintf(stderr, "WARNING: tile intersection overflow. "
                        "Switching to dynamic gaussian-tile intersections.\n");
                overflow_warned = true;
            }
        }
    }
    bool use_dynamic_intersections = should_use_dynamic_intersections(
        img_width, img_height, num_tiles, g_tcache.force_dynamic_intersections);
    bool use_dynamic_u32_keys = use_dynamic_intersections
        && should_use_32_bit_intersection_keys(num_tiles);
    g_tcache.last_sort_path_dynamic = use_dynamic_intersections;
    int64_t capacity = use_dynamic_intersections
        ? std::max<int64_t>(1, g_tcache.capacity)
        : (int64_t)num_points * kFixedTileCapacityMultiplier;
    uint32_t channels = 3;
    uint32_t use_mip_splatting_u32 = use_mip_splatting ? 1u : 0u;
    uint32_t use_half_sorted_buffers_u32 = should_use_half_sorted_buffers() ? 1u : 0u;

    // --- Cached buffer pool ---
    g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                            ssim_weight > 0.0f, lpips_loss_weight > 0.0f,
                            !use_dynamic_intersections, use_dynamic_u32_keys,
                            use_half_sorted_buffers_u32 != 0u, ctx->device);
    g_tcache.ensure_backward(num_points, features_rest_bases, ctx->device);
    bool collect_backward_debug = should_collect_backward_debug();
    if (collect_backward_debug) {
        g_tcache.ensure_backward_debug(num_tiles, ctx->device);
    }

    MTensor &xys = g_tcache.xys;
    MTensor &depths = g_tcache.depths;
    MTensor &radii_out = g_tcache.radii_out;
    MTensor &conics = g_tcache.conics;
    MTensor &opacity_comp = g_tcache.opacity_comp;
    MTensor &num_tiles_hit = g_tcache.num_tiles_hit;
    MTensor &cum_tiles_hit = g_tcache.cum_tiles_hit;
    MTensor &colors = g_tcache.colors;
    MTensor &aabb = g_tcache.aabb;
    MTensor &gaussian_ids = g_tcache.gaussian_ids;
    MTensor &gaussian_ids_tmp = g_tcache.gaussian_ids_tmp;
    MTensor &isect_ids = g_tcache.isect_ids;
    MTensor &isect_ids_tmp = g_tcache.isect_ids_tmp;
    MTensor &isect_ids_u32 = g_tcache.isect_ids_u32;
    MTensor &isect_ids_u32_tmp = g_tcache.isect_ids_u32_tmp;
    MTensor &radix_counts = g_tcache.radix_counts;
    MTensor &tile_bins = g_tcache.tile_bins;
    MTensor &debug_tile_bins_before_raster = g_tcache.debug_tile_bins_before_raster;
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
    std::array<MTensor, 6> adam_grads = {
        v_mean3d, v_scale, v_quat, v_features_dc, v_features_rest, v_opacity
    };

    // --- Constants copied into Metal encoders via setBytes ---
    std::array<uint32_t, 2> loss_img_size = {img_width, img_height};
    uint32_t lpips_numel = img_width * img_height * 3;
    uint32_t use_loss_mask_u32 = use_loss_mask ? 1u : 0u;
    uint32_t use_alpha_loss_u32 = use_alpha_loss ? 1u : 0u;
    uint32_t composite_gt_u32 = composite_gt ? 1u : 0u;
    float alpha_loss_grad_scale = use_alpha_loss ? (alpha_loss_weight / (float)(img_height * img_width)) : 0.0f;
    std::array<float, 4> proj_intrins = {fx, fy, cx, cy};
    std::array<uint32_t, 2> proj_img_size = {img_width, img_height};
    std::array<uint32_t, 4> tile_bounds_arr = {
        (uint32_t)tile_bounds_x, (uint32_t)tile_bounds_y,
        (uint32_t)std::get<2>(tile_bounds), 0xDEAD
    };
    std::array<float, 4> cam_pos_arr = {cam_pos[0], cam_pos[1], cam_pos[2], 0.0f};
    uint32_t num_points_u32 = (uint32_t)num_points;
    uint32_t capacity_u32 = (uint32_t)capacity;
    std::array<uint32_t, 4> img_size_dim3 = {img_width, img_height, 1, 0xDEAD};
    std::array<int32_t, 2> block_size_dim2 = {RAST_BLOCK_X, RAST_BLOCK_Y};
    // tile_bounds for rasterize kernels must be 16x16 tile counts (tile_bins granularity)
    std::array<uint32_t, 4> rast_tb = {
        (img_width + 15u) / 16u,
        (img_height + 15u) / 16u, 1, 0xDEAD
    };
    std::array<uint32_t, 2> rast_isz = {img_width, img_height};
    std::array<float, 4> proj_bwd_intr = {fx, fy, cx, cy};
    std::array<uint32_t, 2> proj_bwd_isz = {img_width, img_height};

    // --- K_max for chunked rasterization ---
    uint32_t K_max = 1;
    constexpr uint32_t CHUNK_SIZE = 512;
    uint32_t bwd_K_max = K_max;
    constexpr uint32_t BWD_CHUNK_SIZE = 512;
    bool use_persplat_backward = should_use_persplat_backward(img_width, img_height, num_tiles);

    // ========================== FORWARD ENCODE LAMBDAS ==========================

    auto encode_copy_debug_tile_bins = [&](id<MTLComputeCommandEncoder> enc) {
        if (!collect_backward_debug) return;
        uint32_t count = (uint32_t)(num_tiles * 2);
        [enc setComputePipelineState:ctx->copy_int_buffer_kernel_cpso];
        ENC_SCALAR(enc, count, 0);
        ENC_BUF(enc, tile_bins, 1);
        ENC_BUF(enc, debug_tile_bins_before_raster, 2);
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto encode_proj_sh = [&](id<MTLComputeCommandEncoder> enc) {
        id<MTLComputePipelineState> pso = project_sh_forward_pipeline(
            ctx, (uint32_t)degrees_to_use, use_mip_splatting_u32 != 0u);
        NSUInteger tpg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:pso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_intrins.data() length:sizeof(proj_intrins) atIndex:7];
        [enc setBytes:proj_img_size.data() length:sizeof(proj_img_size) atIndex:8];
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:9];
        ENC_SCALAR(enc, clip_thresh, 10);
        ENC_BUF(enc, xys, 11); ENC_BUF(enc, depths, 12);
        ENC_BUF(enc, radii_out, 13); ENC_BUF(enc, conics, 14);
        ENC_BUF(enc, num_tiles_hit, 15);
        ENC_SCALAR(enc, degree, 16); ENC_SCALAR(enc, degrees_to_use, 17);
        [enc setBytes:cam_pos_arr.data() length:sizeof(cam_pos_arr) atIndex:18];
        ENC_BUF(enc, features_dc, 19); ENC_BUF(enc, features_rest, 20);
        ENC_BUF(enc, colors, 21); ENC_BUF(enc, aabb, 22);
        ENC_BUF(enc, opacity_comp, 23); ENC_SCALAR(enc, use_mip_splatting_u32, 24);
        ENC_BUF(enc, opacities, 25);

        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_count_prefix = [&](id<MTLComputeCommandEncoder> enc) {
        encode_int32_prefix_sum(ctx, enc, num_points_u32, num_tiles_hit, cum_tiles_hit, g_tcache.block_totals);
    };

    MTensor *sorted_isect_ids = &isect_ids;
    MTensor *sorted_isect_ids_u32 = &isect_ids_u32;
    MTensor *sorted_gaussian_ids = &gaussian_ids;

    auto encode_map_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        id<MTLComputePipelineState> pso = use_dynamic_u32_keys
            ? ctx->map_gaussian_to_intersects_u32_kernel_cpso
            : ctx->map_gaussian_to_intersects_kernel_cpso;
        NSUInteger tpg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:pso];
        ENC_SCALAR(enc, num_points_u32, 0);
        ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, depths, 2);
        ENC_BUF(enc, radii_out, 3);
        ENC_BUF(enc, cum_tiles_hit, 4);
        [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:5];
        ENC_SCALAR(enc, capacity_u32, 6);
        if (use_dynamic_u32_keys) {
            ENC_BUF(enc, isect_ids_u32, 7);
        } else {
            ENC_BUF(enc, isect_ids, 7);
        }
        ENC_BUF(enc, gaussian_ids, 8);
        ENC_BUF(enc, aabb, 9);
        ENC_BUF(enc, g_tcache.overflow_flag, 10);
        ENC_BUF(enc, conics, 11);
        ENC_BUF(enc, opacities, 12);
        ENC_BUF(enc, opacity_comp, 13);
        [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    };

    auto encode_radix_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_blocks = std::max<uint32_t>(1, (capacity_u32 + 255u) / 256u);
        uint32_t radix_passes = use_dynamic_u32_keys ? 4u : radix_pass_count_for_tiles(num_tiles);
        for (uint32_t pass = 0; pass < radix_passes; ++pass) {
            uint32_t shift = pass * 8u;
            bool even_pass = (pass % 2u) == 0u;
            MTensor &keys_in = use_dynamic_u32_keys
                ? (even_pass ? isect_ids_u32 : isect_ids_u32_tmp)
                : (even_pass ? isect_ids : isect_ids_tmp);
            MTensor &vals_in = even_pass ? gaussian_ids : gaussian_ids_tmp;
            MTensor &keys_out = use_dynamic_u32_keys
                ? (even_pass ? isect_ids_u32_tmp : isect_ids_u32)
                : (even_pass ? isect_ids_tmp : isect_ids);
            MTensor &vals_out = even_pass ? gaussian_ids_tmp : gaussian_ids;

            [enc setComputePipelineState:(use_dynamic_u32_keys
                ? ctx->radix_sort_histogram_u32_kernel_cpso
                : ctx->radix_sort_histogram_kernel_cpso)];
            ENC_SCALAR(enc, capacity_u32, 0);
            ENC_BUF(enc, keys_in, 1);
            ENC_BUF(enc, radix_counts, 2);
            ENC_SCALAR(enc, shift, 3);
            ENC_BUF(enc, cum_tiles_hit, 4);
            ENC_SCALAR(enc, num_points_u32, 5);
            [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:ctx->radix_sort_scan_kernel_cpso];
            ENC_BUF(enc, radix_counts, 0);
            ENC_SCALAR(enc, num_blocks, 1);
            ENC_BUF(enc, cum_tiles_hit, 2);
            ENC_SCALAR(enc, capacity_u32, 3);
            ENC_SCALAR(enc, num_points_u32, 4);
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            [enc setComputePipelineState:(use_dynamic_u32_keys
                ? ctx->radix_sort_scatter_u32_kernel_cpso
                : ctx->radix_sort_scatter_kernel_cpso)];
            ENC_SCALAR(enc, capacity_u32, 0);
            ENC_BUF(enc, keys_in, 1);
            ENC_BUF(enc, vals_in, 2);
            ENC_BUF(enc, keys_out, 3);
            ENC_BUF(enc, vals_out, 4);
            ENC_BUF(enc, radix_counts, 5);
            ENC_SCALAR(enc, shift, 6);
            ENC_BUF(enc, cum_tiles_hit, 7);
            ENC_SCALAR(enc, num_points_u32, 8);
            [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        // Publish the final ping-pong buffers for tile edge and pack consumers.
        if ((radix_passes & 1u) != 0u) {
            sorted_isect_ids = &isect_ids_tmp;
            sorted_isect_ids_u32 = &isect_ids_u32_tmp;
            sorted_gaussian_ids = &gaussian_ids_tmp;
        } else {
            sorted_isect_ids = &isect_ids;
            sorted_isect_ids_u32 = &isect_ids_u32;
            sorted_gaussian_ids = &gaussian_ids;
        }
    };

    auto encode_tile_edges_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:(use_dynamic_u32_keys
            ? ctx->get_tile_bin_edges_u32_kernel_cpso
            : ctx->get_tile_bin_edges_kernel_cpso)];
        ENC_SCALAR(enc, capacity_u32, 0);
        if (use_dynamic_u32_keys) {
            [enc setBuffer:(*sorted_isect_ids_u32).buffer() offset:0 atIndex:1];
        } else {
            [enc setBuffer:(*sorted_isect_ids).buffer() offset:0 atIndex:1];
        }
        ENC_BUF(enc, tile_bins, 2);
        ENC_BUF(enc, cum_tiles_hit, 3);
        ENC_SCALAR(enc, num_points_u32, 4);
        [enc dispatchThreads:MTLSizeMake(capacity_u32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto encode_pack_dynamic = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:ctx->pack_sorted_gaussians_kernel_cpso];
        [enc setBuffer:(*sorted_gaussian_ids).buffer() offset:0 atIndex:0];
        ENC_BUF(enc, xys, 1);
        ENC_BUF(enc, conics, 2);
        ENC_BUF(enc, colors, 3);
        ENC_BUF(enc, opacities, 4);
        ENC_BUF(enc, packed_xy_opac, 5);
        ENC_BUF(enc, packed_conic, 6);
        ENC_BUF(enc, packed_rgb, 7);
        ENC_BUF(enc, opacity_comp, 8);
        ENC_BUF(enc, packed_opacity_comp, 9);
        ENC_SCALAR(enc, capacity_u32, 10);
        ENC_BUF(enc, cum_tiles_hit, 11);
        ENC_SCALAR(enc, num_points_u32, 12);
        bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                 use_half_sorted_buffers_u32, 13, 14, 15, 16);
        [enc dispatchThreads:MTLSizeMake(capacity_u32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    auto encode_prefix_map = [&](id<MTLComputeCommandEncoder> enc) {
        encode_map_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_radix_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_tile_edges_dynamic(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        encode_pack_dynamic(enc);
    };

    auto encode_prefix_map_fixed = [&](id<MTLComputeCommandEncoder> enc) {
        uint32_t num_tiles_u32 = (uint32_t)num_tiles;
        {
            NSUInteger tpg = MIN(ctx->scatter_to_prealloc_bins_kernel_cpso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
            [enc setComputePipelineState:ctx->scatter_to_prealloc_bins_kernel_cpso];
            ENC_SCALAR(enc, num_points_u32, 0);
            ENC_BUF(enc, xys, 1);
            ENC_BUF(enc, depths, 2);
            ENC_BUF(enc, radii_out, 3);
            ENC_BUF(enc, aabb, 4);
            [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:5];
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 6);
            ENC_BUF(enc, g_tcache.prealloc_bins, 7);
            ENC_BUF(enc, g_tcache.overflow_flag, 8);
            ENC_BUF(enc, conics, 9);
            ENC_BUF(enc, opacities, 10);
            ENC_BUF(enc, opacity_comp, 11);
            [enc dispatchThreads:MTLSizeMake(num_points, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            encode_int32_prefix_sum(ctx, enc, num_tiles_u32, g_tcache.tile_scatter_counters,
                                    g_tcache.tile_offsets, g_tcache.block_totals);
        }
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        {
            [enc setComputePipelineState:ctx->bitonic_sort_per_tile_kernel_cpso];
            ENC_BUF(enc, g_tcache.tile_offsets, 0);
            ENC_BUF(enc, g_tcache.tile_scatter_counters, 1);
            ENC_BUF(enc, g_tcache.prealloc_bins, 2);
            ENC_BUF(enc, gaussian_ids, 3);
            ENC_SCALAR(enc, num_tiles_u32, 4);
            ENC_BUF(enc, xys, 5);
            ENC_BUF(enc, conics, 6);
            ENC_BUF(enc, colors, 7);
            ENC_BUF(enc, opacities, 8);
            ENC_BUF(enc, opacity_comp, 9);
            ENC_BUF(enc, packed_xy_opac, 10);
            ENC_BUF(enc, packed_conic, 11);
            ENC_BUF(enc, packed_rgb, 12);
            ENC_BUF(enc, packed_opacity_comp, 13);
            ENC_BUF(enc, tile_bins, 14);
            ENC_SCALAR(enc, capacity_u32, 15);
            ENC_BUF(enc, g_tcache.overflow_flag, 16);
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 17, 18, 19, 20);
            [enc dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    };

    auto encode_rast_fwd = [&](id<MTLComputeCommandEncoder> enc) {
        if (K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X, (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y, 1);
            [enc setComputePipelineState:ctx->nd_rasterize_forward_kernel_cpso];
            [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3.data() length:sizeof(img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, final_Ts, 8); ENC_BUF(enc, final_idx, 9); ENC_BUF(enc, out_img, 10);
            ENC_BUF(enc, background, 11);
            [enc setBytes:block_size_dim2.data() length:sizeof(block_size_dim2) atIndex:12];
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 13, 14, 15, 16);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            std::array<uint32_t, 2> img_sz_2 = {img_width, img_height};
            [enc setComputePipelineState:ctx->rasterize_forward_chunked_kernel_cpso];
            [enc setBytes:tile_bounds_arr.data() length:sizeof(tile_bounds_arr) atIndex:0];
            [enc setBytes:img_size_dim3.data() length:sizeof(img_size_dim3) atIndex:1];
            ENC_SCALAR(enc, channels, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5); ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, g_tcache.chunk_T, 8); ENC_BUF(enc, g_tcache.chunk_C, 9); ENC_BUF(enc, g_tcache.chunk_final_idx, 10);
            ENC_SCALAR(enc, CHUNK_SIZE, 11); ENC_SCALAR(enc, K_max, 12);
            [enc setBytes:block_size_dim2.data() length:sizeof(block_size_dim2) atIndex:13];
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 14, 15, 16, 17);
            [enc dispatchThreadgroups:MTLSizeMake(tile_x, tile_y, K_max) threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Merge
            [enc setComputePipelineState:ctx->rasterize_forward_merge_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3); ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, final_Ts, 5); ENC_BUF(enc, final_idx, 6); ENC_BUF(enc, out_img, 7);
            ENC_BUF(enc, background, 8);
            [enc setBytes:img_sz_2.data() length:sizeof(img_sz_2) atIndex:9];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        }
    };

    // Fused loss: ssim_h_fwd → fused_v_fwd_h_bwd → ssim_v_bwd
    // Eliminates loss_intermediates round-trip (130 MB/iter bandwidth saved).
    auto encode_loss_fwd_bwd = [&](id<MTLComputeCommandEncoder> enc) {
        MTLSize loss_tg_count = MTLSizeMake((img_width + 15) / 16, (img_height + 15) / 16, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        if (ssim_weight <= 0.0f) {
            id<MTLComputePipelineState> l1_pso = loss_pipeline(
                ctx, ctx->l1_loss_fwd_bwd_kernel_cpso, ctx->loss_l1_specializations,
                @"l1_loss_fwd_bwd_kernel", composite_gt_u32, use_loss_mask_u32,
                use_alpha_loss_u32, true, true);
            [enc setComputePipelineState:l1_pso];
            ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
            [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:2];
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
        id<MTLComputePipelineState> ssim_h_pso = loss_pipeline(
            ctx, ctx->ssim_h_fwd_kernel_cpso, ctx->loss_ssim_h_specializations,
            @"ssim_h_fwd_kernel", composite_gt_u32, use_loss_mask_u32,
            use_alpha_loss_u32, false, false);
        [enc setComputePipelineState:ssim_h_pso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:2];
        ENC_BUF(enc, g_tcache.ssim_h_buf, 3);
        ENC_BUF(enc, background, 4); ENC_SCALAR(enc, composite_gt_u32, 5);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 2: Fused V fwd + H bwd
        id<MTLComputePipelineState> ssim_fused_pso = loss_pipeline(
            ctx, ctx->ssim_fused_v_fwd_h_bwd_kernel_cpso,
            ctx->loss_ssim_fused_specializations, @"ssim_fused_v_fwd_h_bwd_kernel",
            composite_gt_u32, use_loss_mask_u32, use_alpha_loss_u32, true, true);
        [enc setComputePipelineState:ssim_fused_pso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        ENC_BUF(enc, g_tcache.ssim_h_buf, 2);
        [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:3];
        ENC_SCALAR(enc, ssim_weight, 4); ENC_SCALAR(enc, loss_inv_n, 5);
        ENC_BUF(enc, loss_intermediates, 6); ENC_BUF(enc, loss_sum, 7);
        ENC_BUF(enc, background, 8); ENC_SCALAR(enc, composite_gt_u32, 9);
        ENC_SCALAR(enc, use_loss_mask_u32, 10);
        ENC_BUF(enc, final_Ts, 11);
        ENC_SCALAR(enc, use_alpha_loss_u32, 12); ENC_SCALAR(enc, alpha_loss_weight, 13);
        [enc dispatchThreadgroups:loss_tg_count threadsPerThreadgroup:tg];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        // Pass 3: V bwd
        id<MTLComputePipelineState> ssim_v_bwd_pso = loss_pipeline(
            ctx, ctx->ssim_v_bwd_kernel_cpso, ctx->loss_ssim_v_bwd_specializations,
            @"ssim_v_bwd_kernel", composite_gt_u32, use_loss_mask_u32,
            use_alpha_loss_u32, true, false);
        [enc setComputePipelineState:ssim_v_bwd_pso];
        ENC_BUF(enc, out_img, 0); ENC_BUF(enc, gt_packed, 1);
        ENC_BUF(enc, loss_intermediates, 2);
        [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:3];
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
            [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:2];
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
            [enc setBytes:loss_img_size.data() length:sizeof(loss_img_size) atIndex:2];
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
        if (use_persplat_backward) {
            id<MTLComputePipelineState> pso = raster_backward_pipeline(
                ctx, ctx->rasterize_backward_persplat_kernel_cpso,
                ctx->raster_backward_persplat_specializations,
                @"rasterize_backward_persplat_kernel",
                use_alpha_loss_u32, use_half_sorted_buffers_u32, false);
            [enc setComputePipelineState:pso];
            [enc setBytes:rast_tb.data() length:sizeof(rast_tb) atIndex:0];
            [enc setBytes:rast_isz.data() length:sizeof(rast_isz) atIndex:1];
            ENC_BUF(enc, gaussian_ids, 2); ENC_BUF(enc, tile_bins, 3);
            ENC_BUF(enc, packed_xy_opac, 4); ENC_BUF(enc, packed_conic, 5);
            ENC_BUF(enc, packed_rgb, 6);
            ENC_BUF(enc, packed_opacity_comp, 7);
            ENC_BUF(enc, background, 8); ENC_BUF(enc, out_img, 9);
            ENC_BUF(enc, final_Ts, 10); ENC_BUF(enc, v_rendered, 11);
            ENC_BUF(enc, v_xy, 12); ENC_BUF(enc, v_conic, 13);
            ENC_BUF(enc, v_colors_rast, 14); ENC_BUF(enc, v_opacity, 15);
            ENC_BUF(enc, v_refine, 16);
            ENC_BUF(enc, gt_packed, 17); ENC_SCALAR(enc, use_alpha_loss_u32, 18);
            ENC_SCALAR(enc, alpha_loss_grad_scale, 19);
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 20, 21, 22, 23);
            [enc dispatchThreadgroups:MTLSizeMake(rast_tb[0], rast_tb[1], 1)
                 threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            return;
        }
        if (bwd_K_max <= 1) {
            // Monolithic
            MTLSize num_tg = MTLSizeMake((img_width+RAST_BLOCK_X-1)/RAST_BLOCK_X, (img_height+RAST_BLOCK_Y-1)/RAST_BLOCK_Y, 1);
            id<MTLComputePipelineState> pso = raster_backward_pipeline(
                ctx, ctx->rasterize_backward_kernel_cpso,
                ctx->raster_backward_specializations,
                @"rasterize_backward_kernel",
                use_alpha_loss_u32, use_half_sorted_buffers_u32,
                raster_backward_warp_merge_enabled());
            [enc setComputePipelineState:pso];
            [enc setBytes:rast_tb.data() length:sizeof(rast_tb) atIndex:0];
            [enc setBytes:rast_isz.data() length:sizeof(rast_isz) atIndex:1];
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
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 20, 21, 22, 23);
            [enc dispatchThreadgroups:num_tg threadsPerThreadgroup:MTLSizeMake(RAST_BLOCK_X, RAST_BLOCK_Y, 1)];
        } else {
            // Chunked backward
            uint32_t tile_x = (img_width + RAST_BLOCK_X - 1) / RAST_BLOCK_X;
            uint32_t tile_y = (img_height + RAST_BLOCK_Y - 1) / RAST_BLOCK_Y;
            uint32_t num_pix = img_width * img_height;
            std::array<uint32_t, 2> bwd_img_sz = {img_width, img_height};
            // Phase 1: prefix_T and after_C
            [enc setComputePipelineState:ctx->compute_chunk_prefix_suffix_kernel_cpso];
            ENC_SCALAR(enc, num_pix, 0); ENC_SCALAR(enc, bwd_K_max, 1);
            ENC_BUF(enc, g_tcache.chunk_T, 2); ENC_BUF(enc, g_tcache.chunk_C, 3);
            ENC_BUF(enc, g_tcache.chunk_final_idx, 4);
            ENC_BUF(enc, g_tcache.prefix_T, 5); ENC_BUF(enc, g_tcache.after_C, 6);
            [enc setBytes:bwd_img_sz.data() length:sizeof(bwd_img_sz) atIndex:7];
            [enc dispatchThreads:MTLSizeMake(img_width, img_height, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            // Phase 2: backward chunked
            id<MTLComputePipelineState> pso = raster_backward_pipeline(
                ctx, ctx->rasterize_backward_chunked_kernel_cpso,
                ctx->raster_backward_chunked_specializations,
                @"rasterize_backward_chunked_kernel",
                use_alpha_loss_u32, use_half_sorted_buffers_u32, false);
            [enc setComputePipelineState:pso];
            [enc setBytes:rast_tb.data() length:sizeof(rast_tb) atIndex:0];
            [enc setBytes:rast_isz.data() length:sizeof(rast_isz) atIndex:1];
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
            bind_sorted_half_buffers(enc, packed_conic, packed_rgb, packed_opacity_comp,
                                     use_half_sorted_buffers_u32, 25, 26, 27, 28);
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
    SHAdamParams sh_adam_hp{};
    if (num_adam_groups >= 5) {
        sh_adam_hp.dc_step_size = adam_step_sizes[3];
        sh_adam_hp.dc_bc2_sqrt = adam_bc2_sqrts[3];
        sh_adam_hp.rest_step_size = adam_step_sizes[4];
        sh_adam_hp.rest_bc2_sqrt = adam_bc2_sqrts[4];
        sh_adam_hp.beta1 = adam_beta1;
        sh_adam_hp.beta2 = adam_beta2;
        sh_adam_hp.eps = adam_eps;
        sh_adam_hp.reduce_second_moment = reduce_second_moment ? 1u : 0u;
    }

    auto encode_proj_sh_bwd_adam = [&](id<MTLComputeCommandEncoder> enc) {
        id<MTLComputePipelineState> pso = project_sh_backward_pipeline(
            ctx, (uint32_t)degrees_to_use, sh_adam_hp.reduce_second_moment != 0u);
        NSUInteger tpg = MIN(pso.maxTotalThreadsPerThreadgroup, (NSUInteger)num_points);
        [enc setComputePipelineState:pso];
        ENC_SCALAR(enc, num_points, 0); ENC_BUF(enc, means3d, 1); ENC_BUF(enc, scales, 2);
        ENC_SCALAR(enc, glob_scale, 3); ENC_BUF(enc, quats, 4);
        ENC_BUF(enc, viewmat, 5); ENC_BUF(enc, projmat, 6);
        [enc setBytes:proj_bwd_intr.data() length:sizeof(proj_bwd_intr) atIndex:7];
        [enc setBytes:proj_bwd_isz.data() length:sizeof(proj_bwd_isz) atIndex:8];
        ENC_BUF(enc, radii_out, 9); ENC_BUF(enc, conics, 10);
        ENC_BUF(enc, v_xy, 11); ENC_BUF(enc, v_depth, 12); ENC_BUF(enc, v_conic, 13);
        ENC_BUF(enc, v_mean3d, 14); ENC_BUF(enc, v_scale, 15); ENC_BUF(enc, v_quat, 16);
        ENC_SCALAR(enc, degree, 17); ENC_SCALAR(enc, degrees_to_use, 18);
        [enc setBytes:cam_pos_arr.data() length:sizeof(cam_pos_arr) atIndex:19];
        ENC_BUF(enc, v_colors_rast, 20);
        // Fused SH backward + Adam: pass params + optimizer state instead of gradient buffers
        [enc setBuffer:adam_params[3].buffer() offset:0 atIndex:21];  // features_dc params
        [enc setBuffer:adam_params[4].buffer() offset:0 atIndex:22];  // features_rest params
        [enc setBuffer:adam_exp_avg[3].buffer() offset:0 atIndex:23]; // dc exp_avg
        [enc setBuffer:adam_exp_avg_sq[3].buffer() offset:0 atIndex:24]; // dc exp_avg_sq
        [enc setBuffer:adam_exp_avg[4].buffer() offset:0 atIndex:25]; // rest exp_avg
        [enc setBuffer:adam_exp_avg_sq[4].buffer() offset:0 atIndex:26]; // rest exp_avg_sq
        [enc setBytes:&sh_adam_hp length:sizeof(sh_adam_hp) atIndex:27];
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
                [enc setBuffer:adam_grads[g].buffer() offset:0 atIndex:1];
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
        [blit fillBuffer:loss_sum.buffer() range:NSMakeRange(0, loss_sum.nbytes()) value:0];
        [blit fillBuffer:g_tcache.overflow_flag.buffer() range:NSMakeRange(0, g_tcache.overflow_flag.nbytes()) value:0];
        [blit fillBuffer:tile_bins.buffer() range:NSMakeRange(0, tile_bins.nbytes()) value:0];
        if (!use_dynamic_intersections) {
            [blit fillBuffer:g_tcache.tile_scatter_counters.buffer() range:NSMakeRange(0, g_tcache.tile_scatter_counters.nbytes()) value:0];
        }
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

    bool did_dynamic_count_prepass = false;
    if (use_dynamic_intersections
        && !g_tcache.has_dynamic_capacity(num_points, img_height, img_width, num_tiles)) {
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");
        dispatch_sync(ctx->d_queue, ^(){
            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            assert(enc && "Failed to create compute command encoder");
            encode_proj_sh(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_count_prefix(enc);
            [enc endEncoding];
        });
        record_forced_sync("dynamic-count-prepass");
        ctx->syncCB();
        int64_t exact_intersections = read_dynamic_intersection_count(cum_tiles_hit, num_points);
        capacity = padded_dynamic_intersection_capacity(exact_intersections);
        capacity_u32 = (uint32_t)capacity;
        g_tcache.ensure_forward(num_points, capacity, img_height, img_width, num_tiles,
                                ssim_weight > 0.0f, lpips_loss_weight > 0.0f,
                                false, use_dynamic_u32_keys,
                                use_half_sorted_buffers_u32 != 0u, ctx->device);
        g_tcache.mark_dynamic_capacity(num_points, img_height, img_width, num_tiles);
        did_dynamic_count_prepass = true;
    } else {
        capacity = use_dynamic_intersections
            ? std::max<int64_t>(1, g_tcache.capacity)
            : (int64_t)num_points * kFixedTileCapacityMultiplier;
        capacity_u32 = (uint32_t)capacity;
    }

    if (num_tiles >= 400) {
        K_max = 1;
    } else {
        uint32_t avg_per_tile = (uint32_t)(capacity / std::max(1, num_tiles));
        uint32_t conservative_max = avg_per_tile * 6;
        K_max = (conservative_max + CHUNK_SIZE - 1) / CHUNK_SIZE;
        if (K_max < 2) K_max = 2;
        uint32_t abs_max = (uint32_t)((capacity + CHUNK_SIZE - 1) / CHUNK_SIZE);
        if (K_max > abs_max) K_max = abs_max;
    }
    g_tcache.current_K_max = K_max;
    if (K_max > 1) {
        g_tcache.ensure_chunks(K_max, img_height, img_width, ctx->device);
    }
    bwd_K_max = K_max;

    id<MTLCommandBuffer> debug_command_buffer = nil;

    if (g_profile_stages) {
        auto record_stage_time = [&](int stage_idx, double ms) {
            if (stage_idx < 0 || stage_idx >= N_TRAIN_STAGES || ms < 0.0) return;
            std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
            g_stage_times[stage_idx].push_back(ms);
        };

        auto run_profiled_stage = [&](int stage_idx, const std::function<void(MPSCommandBuffer *)> &encode_stage) {
            __block double gpu_ms = 0.0;
            dispatch_sync(ctx->d_queue, ^(){
                MPSCommandBuffer *stage_cb = [MPSCommandBuffer commandBufferFromCommandQueue:ctx->queue];
                [stage_cb retain];
                encode_stage(stage_cb);
                [stage_cb commit];
                [stage_cb waitUntilCompleted];
                gpu_ms = (stage_cb.GPUEndTime - stage_cb.GPUStartTime) * 1000.0;
                [stage_cb release];
            });
            record_stage_time(stage_idx, gpu_ms);
        };

        auto run_profiled_compute_stage = [&](int stage_idx, const std::function<void(id<MTLComputeCommandEncoder>)> &encode_stage) {
            run_profiled_stage(stage_idx, [&](MPSCommandBuffer *stage_cb) {
                id<MTLComputeCommandEncoder> enc = [stage_cb computeCommandEncoder];
                assert(enc && "Failed to create compute command encoder");
                encode_stage(enc);
                [enc endEncoding];
            });
        };

        run_profiled_stage(0, [&](MPSCommandBuffer *stage_cb) {
            do_blit_zero(stage_cb);
        });

        run_profiled_compute_stage(1, [&](id<MTLComputeCommandEncoder> enc) {
            if (!use_dynamic_intersections || !did_dynamic_count_prepass) {
                encode_proj_sh(enc);
                if (use_dynamic_intersections) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    encode_count_prefix(enc);
                }
            }
        });

        run_profiled_compute_stage(2, [&](id<MTLComputeCommandEncoder> enc) {
            if (use_dynamic_intersections) {
                encode_map_dynamic(enc);
            } else {
                encode_prefix_map_fixed(enc);
            }
        });

        run_profiled_compute_stage(3, [&](id<MTLComputeCommandEncoder> enc) {
            if (use_dynamic_intersections) encode_radix_dynamic(enc);
        });

        run_profiled_compute_stage(4, [&](id<MTLComputeCommandEncoder> enc) {
            if (use_dynamic_intersections) encode_tile_edges_dynamic(enc);
        });

        run_profiled_compute_stage(5, [&](id<MTLComputeCommandEncoder> enc) {
            if (use_dynamic_intersections) encode_pack_dynamic(enc);
        });

        run_profiled_compute_stage(6, [&](id<MTLComputeCommandEncoder> enc) {
            encode_copy_debug_tile_bins(enc);
            if (collect_backward_debug) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            encode_rast_fwd(enc);
        });

        run_profiled_stage(7, [&](MPSCommandBuffer *stage_cb) {
            id<MTLComputeCommandEncoder> enc = [stage_cb computeCommandEncoder];
            assert(enc && "Failed to create compute command encoder");
            encode_loss_fwd_bwd(enc);
            [enc endEncoding];
            encode_lpips(stage_cb);
        });

        run_profiled_compute_stage(8, [&](id<MTLComputeCommandEncoder> enc) {
            encode_rast_bwd(enc);
        });

        run_profiled_compute_stage(9, [&](id<MTLComputeCommandEncoder> enc) {
            encode_proj_sh_bwd_adam(enc);
        });

        run_profiled_compute_stage(10, [&](id<MTLComputeCommandEncoder> enc) {
            encode_grad_stats(enc);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_pup_hessian(enc);
        });

        if (collect_backward_debug) {
            id<MTLBuffer> pre_bins_buffer = debug_tile_bins_before_raster.buffer();
            id<MTLBuffer> post_bins_buffer = tile_bins.buffer();
            id<MTLBuffer> final_Ts_buffer = final_Ts.buffer();
            const int32_t *pre_bins = static_cast<const int32_t *>([pre_bins_buffer contents]);
            const int32_t *post_bins = static_cast<const int32_t *>([post_bins_buffer contents]);
            const float *final_Ts_ptr = static_cast<const float *>([final_Ts_buffer contents]);
            BackwardDebugSample sample = make_backward_debug_sample(
                pre_bins, post_bins, final_Ts_ptr,
                (int)rast_tb[0], (int)rast_tb[1], (int)img_width, (int)img_height);
            record_backward_debug_sample(sample, use_persplat_backward);
        }

        {
            std::lock_guard<std::mutex> lock(g_stage_timing_mutex);
            g_stage_report_count++;
            int report_every = stage_profile_report_interval();
            if (g_stage_report_count % report_every == 0) {
                fprintf(stderr, "\n  === GPU Stage Profile (n=%d) ===\n", g_stage_report_count);
                double total_median = 0;
                for (int i = 0; i < N_TRAIN_STAGES; i++) {
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
    } else {
        // Production: single encoder for everything
        id<MTLCommandBuffer> command_buffer = ctx->getCommandBuffer();
        assert(command_buffer && "Failed to retrieve command buffer reference");
        debug_command_buffer = command_buffer;

        dispatch_sync(ctx->d_queue, ^(){
            do_blit_zero(command_buffer);

            id<MTLComputeCommandEncoder> enc = [command_buffer computeCommandEncoder];
            assert(enc && "Failed to create compute command encoder");

            // --- Forward: intersections → rasterize → loss ---
            if (use_dynamic_intersections) {
                if (!did_dynamic_count_prepass) {
                    encode_proj_sh(enc);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    encode_count_prefix(enc);
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                encode_prefix_map(enc);
            } else {
                encode_proj_sh(enc);
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode_prefix_map_fixed(enc);
            }
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_copy_debug_tile_bins(enc);
            if (collect_backward_debug) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
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

    if (collect_backward_debug && debug_command_buffer) {
        id<MTLBuffer> pre_bins_buffer = debug_tile_bins_before_raster.buffer();
        id<MTLBuffer> post_bins_buffer = tile_bins.buffer();
        id<MTLBuffer> final_Ts_buffer = final_Ts.buffer();
        [pre_bins_buffer retain];
        [post_bins_buffer retain];
        [final_Ts_buffer retain];
        int debug_tile_bounds_x = (int)rast_tb[0];
        int debug_tile_bounds_y = (int)rast_tb[1];
        int debug_img_width = (int)img_width;
        int debug_img_height = (int)img_height;
        bool debug_persplat_enabled = use_persplat_backward;
        [debug_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            @autoreleasepool {
                if (cb.status == MTLCommandBufferStatusCompleted) {
                    const int32_t *pre_bins = static_cast<const int32_t *>([pre_bins_buffer contents]);
                    const int32_t *post_bins = static_cast<const int32_t *>([post_bins_buffer contents]);
                    const float *final_Ts_ptr = static_cast<const float *>([final_Ts_buffer contents]);
                    BackwardDebugSample sample = make_backward_debug_sample(
                        pre_bins, post_bins, final_Ts_ptr,
                        debug_tile_bounds_x, debug_tile_bounds_y,
                        debug_img_width, debug_img_height);
                    record_backward_debug_sample(sample, debug_persplat_enabled);
                }
                [pre_bins_buffer release];
                [post_bins_buffer release];
                [final_Ts_buffer release];
            }
        }];
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
