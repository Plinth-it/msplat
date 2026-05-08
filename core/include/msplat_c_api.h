// C API for Swift interop. Thin wrapper around msplat C++ types.
// Opaque handles + free functions — works with any Swift version.

#ifndef MSPLAT_C_API_H
#define MSPLAT_C_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Config ──────────────────────────────────────────────────────────────────

typedef struct {
    int iterations;
    int shDegree;
    int shDegreeInterval;
    float ssimWeight;
    int numDownscales;
    int resolutionSchedule;
    int refineEvery;
    int warmupLength;
    int resetAlphaEvery;
    float densifyGradThresh;
    float densifySizeThresh;
    int stopScreenSizeAt;
    float splitScreenSize;
    bool keepCrs;
    bool renderMip;
    float downscaleFactor;
    float bgColor[3];
    float matchAlphaWeight;
    float backgroundNoiseStrength;
    float opacityDecay;
    float scaleDecay;
    float meanNoiseWeight;
    int growthStopIter;
    int maxSplats;
    float growthSelectFraction;
    float lrMean;
    float lrMeanEnd;
    float lrScale;
    float lrScaleEnd;
    float lrRotation;
    float lrCoeffsDc;
    float lrCoeffsShScale;
    float lrOpacity;
    float lpipsLossWeight;
    float randomInitSceneScale;
    bool reduceSecondMoment;
    int imagePrefetchWorkers;
} MsplatConfig;

static inline MsplatConfig msplat_default_config(void) {
    MsplatConfig c;
    c.iterations = 30000;
    c.shDegree = 3;
    c.shDegreeInterval = 0;
    c.ssimWeight = 0.2f;
    c.numDownscales = 0;
    c.resolutionSchedule = 3000;
    c.refineEvery = 200;
    c.warmupLength = 0;
    c.resetAlphaEvery = 0;
    c.densifyGradThresh = 0.0025f;
    c.densifySizeThresh = 0.01f;
    c.stopScreenSizeAt = 15000;
    c.splitScreenSize = 0.25f;
    c.keepCrs = false;
    c.renderMip = false;
    c.downscaleFactor = 1.0f;
    c.bgColor[0] = 0.0f; c.bgColor[1] = 0.0f; c.bgColor[2] = 0.0f;
    c.matchAlphaWeight = 0.1f;
    c.backgroundNoiseStrength = 0.1f;
    c.opacityDecay = 0.004f;
    c.scaleDecay = 0.002f;
    c.meanNoiseWeight = 50.0f;
    c.growthStopIter = 15000;
    c.maxSplats = 10000000;
    c.growthSelectFraction = 0.25f;
    c.lrMean = 2e-5f;
    c.lrMeanEnd = 2e-7f;
    c.lrScale = 7e-3f;
    c.lrScaleEnd = 5e-3f;
    c.lrRotation = 0.002f;
    c.lrCoeffsDc = 2e-3f;
    c.lrCoeffsShScale = 10.0f;
    c.lrOpacity = 0.012f;
    c.lpipsLossWeight = 0.0f;
    c.randomInitSceneScale = 0.0f;
    c.reduceSecondMoment = false;
    c.imagePrefetchWorkers = 2;
    return c;
}

// ── Stats ───────────────────────────────────────────────────────────────────

typedef struct {
    int iteration;
    int splatCount;
    float msPerStep;
} MsplatStats;

typedef struct {
    float psnr;
    float ssim;
    float l1;
    int numTest;
    int numGaussians;
} MsplatEvalMetrics;

// ── Pixel buffer ────────────────────────────────────────────────────────────

typedef struct {
    float* data;   // RGB float32, HWC layout. Caller must free() this.
    int width;
    int height;
} MsplatPixelBuffer;

// ── Dataset ─────────────────────────────────────────────────────────────────

typedef void* MsplatDataset;

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery);
void msplat_dataset_destroy(MsplatDataset ds);
int msplat_dataset_num_train(MsplatDataset ds);
int msplat_dataset_num_test(MsplatDataset ds);
int msplat_dataset_initial_point_count(MsplatDataset ds);
bool msplat_dataset_camera_has_alpha(MsplatDataset ds, int cameraIndex);
bool msplat_dataset_camera_has_mask(MsplatDataset ds, int cameraIndex);

// ── Trainer ─────────────────────────────────────────────────────────────────

typedef void* MsplatTrainer;

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config);
void msplat_trainer_destroy(MsplatTrainer t);

MsplatStats msplat_trainer_step(MsplatTrainer t);
void msplat_trainer_train(MsplatTrainer t);
MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t);
MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest);
MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex);

/// Render into a caller-provided RGBA uint8 buffer (no allocation, no float copy).
/// outRGBA must be at least width*height*4 bytes. Returns dimensions via outWidth/outHeight.
/// Call once with outRGBA=NULL to get dimensions, then allocate and call again.
void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight);
void msplat_trainer_export_ply(MsplatTrainer t, const char* path);
void msplat_trainer_export_lod_ply(MsplatTrainer t, const char* path, int targetCount);
void msplat_trainer_decimate_to_lod(MsplatTrainer t, int targetCount);
void msplat_trainer_export_splat(MsplatTrainer t, const char* path);
int msplat_trainer_load_ply(MsplatTrainer t, const char* path);
void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path);
int msplat_trainer_splat_count(MsplatTrainer t);
int msplat_trainer_iteration(MsplatTrainer t);
void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]);

// ── Lifecycle ───────────────────────────────────────────────────────────────

void msplat_set_metallib_path(const char* path);
void msplat_set_lpips_weights_path(const char* path);
const char* msplat_last_error(void);
void msplat_sync(void);
void msplat_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // MSPLAT_C_API_H
