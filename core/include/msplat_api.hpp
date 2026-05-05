#pragma once

// Swift-compatible C++ API for msplat.
// Designed for Swift 5.9+ C++ interop — no std::tuple,
// no std::unordered_map, no templates in the public interface.
// Internal types hidden via PIMPL.

#include <cstdint>
#include <memory>
#include <string>

namespace msplat {

// ── Config ──────────────────────────────────────────────────────────────────

struct Config {
    int iterations = 30000;
    int shDegree = 3;
    int shDegreeInterval = 1;
    float ssimWeight = 0.2f;
    int numDownscales = 2;
    int resolutionSchedule = 3000;
    int refineEvery = 100;
    int warmupLength = 500;
    int resetAlphaEvery = 30;
    float densifyGradThresh = 0.008f;
    float densifySizeThresh = 0.01f;
    int stopScreenSizeAt = 4000;
    float splitScreenSize = 0.05f;
    bool keepCrs = false;
    bool renderMip = false;
    float downscaleFactor = 1.0f;
    float bgColor[3] = {0.0f, 0.0f, 0.0f};  // Brush-compatible black background
    float matchAlphaWeight = 0.1f;
    float backgroundNoiseStrength = 0.1f;
    float opacityDecay = 0.004f;
    float scaleDecay = 0.002f;
    float meanNoiseWeight = 50.0f;
    int growthStopIter = 15000;
    int maxSplats = 10000000;
    float growthSelectFraction = 0.25f;
    float lrMean = 0.00256f;
    float lrMeanEnd = 0.0000256f;
    float lrScale = 0.022f;
    float lrScaleEnd = 0.022f;
    float lrRotation = 0.002f;
    float lrCoeffsDc = 0.012f;
    float lrCoeffsShScale = 10.0f;
    float lrOpacity = 0.035f;
    float lpipsLossWeight = 0.0f;
    float randomInitSceneScale = 0.0f;  // 0 estimates from cameras when no point cloud exists
    bool reduceSecondMoment = false;
};

// ── Stats ───────────────────────────────────────────────────────────────────

struct Stats {
    int iteration = 0;
    int splatCount = 0;
    float msPerStep = 0.0f;
};

struct EvalMetrics {
    float psnr = 0.0f;
    float ssim = 0.0f;
    float l1 = 0.0f;
    int numTest = 0;
    int numGaussians = 0;
};

// ── PixelBuffer ─────────────────────────────────────────────────────────────

/// Rendered image data. Owns its pixel buffer.
struct PixelBuffer {
    float* data = nullptr;   // RGB float32, HWC layout
    int width = 0;
    int height = 0;

    PixelBuffer() = default;
    PixelBuffer(float* d, int w, int h) : data(d), width(w), height(h) {}
    PixelBuffer(const PixelBuffer&) = delete;
    PixelBuffer& operator=(const PixelBuffer&) = delete;
    PixelBuffer(PixelBuffer&& o) : data(o.data), width(o.width), height(o.height) {
        o.data = nullptr;
    }
    PixelBuffer& operator=(PixelBuffer&& o) {
        if (this != &o) {
            free(data);
            data = o.data; width = o.width; height = o.height;
            o.data = nullptr;
        }
        return *this;
    }
    ~PixelBuffer() { free(data); }
};

// ── Dataset ─────────────────────────────────────────────────────────────────

class Dataset {
public:
    Dataset(const std::string& path, float downscaleFactor,
            bool evalMode, int testEvery);
    ~Dataset();

    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    Dataset(Dataset&&) noexcept;
    Dataset& operator=(Dataset&&) noexcept;

    int numTrain() const;
    int numTest() const;
    void cameraPose(int index, float camToWorld[16]) const;
    bool cameraHasAlpha(int index) const;
    bool cameraHasMask(int index) const;

    // Opaque handle for Trainer
    void* _handle() const;

    struct Impl;
private:
    std::unique_ptr<Impl> impl;
};

// ── Trainer ─────────────────────────────────────────────────────────────────

class Trainer {
public:
    Trainer(Dataset& dataset, const Config& config);
    ~Trainer();

    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    /// Run one training step. Returns stats.
    Stats step();

    /// Run N steps (from current iteration to config.iterations).
    /// Calls callback every callbackEvery steps with current Stats.
    /// Use callbackEvery=0 to disable callbacks.
    void train(int callbackEvery);

    /// Evaluate on held-out test cameras.
    EvalMetrics evaluate();

    /// Render a camera view. Caller owns the returned PixelBuffer.
    PixelBuffer render(int cameraIndex, bool useTest);

    /// Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).
    /// Uses intrinsics from the given reference camera.
    PixelBuffer renderFromPose(const float camToWorld[16], int refCameraIndex);

    /// Render from pose directly into a caller-provided RGBA uint8 buffer.
    /// outRGBA must hold width*height*4 bytes. Avoids intermediate float allocation.
    /// Call with outRGBA=nullptr to query dimensions only.
    void renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                            uint8_t* outRGBA, int* outWidth, int* outHeight);

    /// Export scene to PLY format.
    void exportPly(const std::string& path);

    /// Export a deterministic importance-ranked LOD PLY with at most targetCount gaussians.
    void exportLodPly(const std::string& path, int targetCount);

    /// Decimate the active model in memory to at most targetCount gaussians.
    void decimateToLod(int targetCount);

    /// Export scene to .splat format.
    void exportSplat(const std::string& path);

    /// Save full training state (params + optimizer) for resume.
    void saveCheckpoint(const std::string& path);

    /// Load checkpoint and resume training. Returns the saved iteration.
    int loadCheckpoint(const std::string& path);

    int splatCount() const;
    int iteration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync();
void cleanup();

} // namespace msplat
