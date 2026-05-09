// ObjC++ implementation of the Swift-facing C++ API.
// This is the ONLY file that touches internal C++ types (Model, Camera, MTensor).

#include "msplat_api.hpp"

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <array>
#include <stdexcept>
#include <string>

namespace msplat {

// ── Dataset::Impl ───────────────────────────────────────────────────────────

struct Dataset::Impl {
    InputData data;
    std::vector<Camera> trainCams;
    std::vector<Camera> testCams;
};

Dataset::Dataset(const std::string& path, float downscaleFactor,
                 bool evalMode, int testEvery)
    : impl(std::make_unique<Impl>())
{
    impl->data = inputDataFromX(path);

    for (auto& cam : impl->data.cameras)
        cam.configureLazyImageLoad(downscaleFactor);
    for (auto& cam : impl->data.evalCameras)
        cam.configureLazyImageLoad(downscaleFactor);

    if (evalMode) {
        auto split = impl->data.splitTrainTest(testEvery);
        impl->trainCams = std::get<0>(split);
        impl->testCams = std::get<1>(split);
    } else {
        auto t = impl->data.getCameras(false);
        impl->trainCams = std::get<0>(t);
    }
}

Dataset::~Dataset() = default;
Dataset::Dataset(Dataset&&) noexcept = default;
Dataset& Dataset::operator=(Dataset&&) noexcept = default;

int Dataset::numTrain() const { return (int)impl->trainCams.size(); }
int Dataset::numTest() const { return (int)impl->testCams.size(); }
int Dataset::initialPointCount() const { return (int)impl->data.points.count; }
void Dataset::cameraPose(int index, float camToWorld[16]) const {
    if (index >= 0 && index < (int)impl->trainCams.size())
        memcpy(camToWorld, impl->trainCams[index].camToWorld, 16 * sizeof(float));
}
bool Dataset::cameraHasAlpha(int index) const {
    return index >= 0 && index < (int)impl->trainCams.size()
        && impl->trainCams[index].imageHasAlpha();
}
bool Dataset::cameraHasMask(int index) const {
    return index >= 0 && index < (int)impl->trainCams.size()
        && impl->trainCams[index].hasExplicitMask();
}
void* Dataset::_handle() const { return impl.get(); }

// ── Trainer::Impl ───────────────────────────────────────────────────────────

struct Trainer::Impl {
    std::unique_ptr<Model> model;
    Config config;
    Dataset::Impl* ds = nullptr;
    int currentStep = 0;

    std::mt19937 bgRng{1337};
    std::unique_ptr<CameraPrefetcher> cameraPrefetcher;

    void resetCameraPrefetcher() {
        size_t workerCount = static_cast<size_t>(std::clamp(config.imagePrefetchWorkers, 1, 8));
        cameraPrefetcher = std::make_unique<CameraPrefetcher>(ds->trainCams, 42, workerCount);
    }

    size_t nextCamera() {
        if (!cameraPrefetcher) resetCameraPrefetcher();
        return cameraPrefetcher->next();
    }

    std::array<float, 3> sampleBackground() {
        std::array<float, 3> bg = {config.bgColor[0], config.bgColor[1], config.bgColor[2]};
        if (config.backgroundNoiseStrength <= 0.0f) {
            return bg;
        }
        std::uniform_real_distribution<float> dist(-config.backgroundNoiseStrength,
                                                    config.backgroundNoiseStrength);
        for (float& channel : bg) {
            channel = std::clamp(channel + dist(bgRng), 0.0f, 1.0f);
        }
        return bg;
    }
};

static Camera cameraWithPose(const Camera& reference, const float camToWorld[16]) {
    Camera cam;
    cam.width = reference.width;
    cam.height = reference.height;
    cam.fx = reference.fx;
    cam.fy = reference.fy;
    cam.cx = reference.cx;
    cam.cy = reference.cy;
    cam.k1 = reference.k1;
    cam.k2 = reference.k2;
    cam.k3 = reference.k3;
    cam.p1 = reference.p1;
    cam.p2 = reference.p2;
    memcpy(cam.camToWorld, camToWorld, 16 * sizeof(float));
    return cam;
}

Trainer::Trainer(Dataset& dataset, const Config& config)
    : impl(std::make_unique<Impl>())
{
    impl->config = config;
    impl->ds = static_cast<Dataset::Impl*>(dataset._handle());
    impl->resetCameraPrefetcher();

    impl->model = std::make_unique<Model>(
        impl->ds->data,
        (int)impl->ds->trainCams.size(),
        config.numDownscales, config.resolutionSchedule,
        config.shDegree, config.shDegreeInterval,
        config.refineEvery, config.warmupLength, config.resetAlphaEvery,
        config.densifyGradThresh, config.densifySizeThresh,
        config.stopScreenSizeAt, config.splitScreenSize,
        config.iterations, config.keepCrs, config.growthStopIter,
        config.maxSplats, config.growthSelectFraction,
        config.opacityDecay, config.scaleDecay,
        config.meanNoiseWeight,
        config.lrMean, config.lrMeanEnd, config.lrScale, config.lrScaleEnd,
        config.lrRotation, config.lrCoeffsDc, config.lrCoeffsShScale, config.lrOpacity,
        config.randomInitSceneScale, config.reduceSecondMoment,
        42,
        config.bgColor, config.renderMip
    );
}

Trainer::~Trainer() = default;

Stats Trainer::step() {
    impl->currentStep++;
    size_t camIdx = impl->nextCamera();
    Camera& cam = impl->ds->trainCams[camIdx];

    int ds = impl->model->getDownscaleFactor(impl->currentStep);
    std::array<float, 3> stepBg = impl->sampleBackground();
    MTensor& gtPacked = cam.getGPUPackedImage(ds);
    bool useLossMask = cam.hasLossMask();
    float lossMaskMean = 1.0f;
    if (useLossMask) {
        lossMaskMean = cam.getLossMaskMean(ds);
    }
    bool useAlphaLoss = !useLossMask && cam.imageHasAlpha();
    bool compositeGt = cam.hasCompositeAlpha()
        && (stepBg[0] != 0.0f || stepBg[1] != 0.0f || stepBg[2] != 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();

    impl->model->fullIteration(cam, impl->currentStep, gtPacked, useLossMask, lossMaskMean,
                               useAlphaLoss, impl->config.matchAlphaWeight,
                               stepBg.data(), compositeGt, impl->config.ssimWeight,
                               impl->config.lpipsLossWeight);
    impl->model->schedulersStep(impl->currentStep);
    impl->model->afterTrain(impl->currentStep);
    msplat_commit();

    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

    Stats s;
    s.iteration = impl->currentStep;
    s.splatCount = (int)impl->model->means.size(0);
    s.msPerStep = ms;
    return s;
}

void Trainer::train(int callbackEvery) {
    while (impl->currentStep < impl->config.iterations) {
        step();
        // Note: callbacks handled at the Swift level via polling iteration()
        // to keep the C++ API free of function pointer complexity
    }
}

EvalMetrics Trainer::evaluate() {
    auto& testCams = impl->ds->testCams;
    if (testCams.empty())
        return {};

    double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
    int n = (int)testCams.size();
    const float evalBg[3] = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < n; i++) {
        Camera& cam = testCams[i];
        cam.ensureImageLoaded();
        MTensor rgb = impl->model->render(cam, impl->config.iterations, evalBg);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        int dsf = impl->model->getDownscaleFactor(impl->config.iterations);
        MTensor gtCpu = cam.getGPUImage(dsf, evalBg).cpu();
        quantizeRenderedForEval(rgbCpu);

        sumPsnr += psnr(rgbCpu, gtCpu);
        sumSsim += ssim_eval(rgbCpu, gtCpu);
        sumL1 += l1_loss(rgbCpu, gtCpu);
    }

    EvalMetrics m;
    m.psnr = (float)(sumPsnr / n);
    m.ssim = (float)(sumSsim / n);
    m.l1 = (float)(sumL1 / n);
    m.numTest = n;
    m.numGaussians = (int)impl->model->means.size(0);
    return m;
}

PixelBuffer Trainer::render(int cameraIndex, bool useTest) {
    auto& cams = useTest ? impl->ds->testCams : impl->ds->trainCams;
    if (cameraIndex < 0 || cameraIndex >= (int)cams.size())
        return {};

    Camera& cam = cams[cameraIndex];
    cam.ensureImageLoaded();
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    // Use malloc so callers can free() — PixelBuffer destructor handles both
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));

    return PixelBuffer(buf, w, h);
}

PixelBuffer Trainer::renderFromPose(const float camToWorld[16], int refCameraIndex) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size())
        return {};

    Camera& reference = cams[refCameraIndex];
    reference.ensureImageLoaded();
    Camera cam = cameraWithPose(reference, camToWorld);
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();
    MTensor rgbCpu = rgb.cpu();

    int h = (int)rgbCpu.size(0);
    int w = (int)rgbCpu.size(1);
    float* buf = (float*)malloc(h * w * 3 * sizeof(float));
    memcpy(buf, rgbCpu.data_ptr(), h * w * 3 * sizeof(float));
    return PixelBuffer(buf, w, h);
}

void Trainer::renderFromPoseToBuffer(const float camToWorld[16], int refCameraIndex,
                                  uint8_t* outRGBA, int* outWidth, int* outHeight) {
    auto& cams = impl->ds->trainCams;
    if (refCameraIndex < 0 || refCameraIndex >= (int)cams.size()) {
        if (outWidth) *outWidth = 0;
        if (outHeight) *outHeight = 0;
        return;
    }

    Camera& reference = cams[refCameraIndex];
    reference.ensureImageLoaded();
    int downscale = impl->model->getDownscaleFactor(impl->currentStep);
    int queryWidth = static_cast<int>(reference.width / static_cast<float>(downscale));
    int queryHeight = static_cast<int>(reference.height / static_cast<float>(downscale));
    if (outWidth) *outWidth = queryWidth;
    if (outHeight) *outHeight = queryHeight;
    if (!outRGBA) return;

    Camera cam = cameraWithPose(reference, camToWorld);
    MTensor rgb = impl->model->render(cam, impl->currentStep);
    msplat_gpu_sync();

    int h = (int)rgb.size(0), w = (int)rgb.size(1);
    if (outWidth) *outWidth = w;
    if (outHeight) *outHeight = h;

    // Read directly from GPU tensor (unified memory on Apple Silicon)
    const float* src = (const float*)rgb.data_ptr();
    int n = w * h;
    for (int i = 0; i < n; i++) {
        outRGBA[i * 4]     = (uint8_t)(fminf(fmaxf(src[i*3],   0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 1] = (uint8_t)(fminf(fmaxf(src[i*3+1], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 2] = (uint8_t)(fminf(fmaxf(src[i*3+2], 0.f), 1.f) * 255.f);
        outRGBA[i * 4 + 3] = 255;
    }
}

void Trainer::exportPly(const std::string& path) {
    impl->model->savePly(path, impl->currentStep);
}

void Trainer::exportLodPly(const std::string& path, int targetCount) {
    impl->model->saveLodPly(path, impl->currentStep, targetCount);
}

void Trainer::decimateToLod(int targetCount) {
    impl->model->decimateToLod(targetCount);
}

void Trainer::exportSplat(const std::string& path) {
    impl->model->saveSplat(path);
}

int Trainer::loadPly(const std::string& path) {
    impl->currentStep = impl->model->loadPly(path);
    impl->resetCameraPrefetcher();
    return impl->currentStep;
}

void Trainer::saveCheckpoint(const std::string& path) {
    impl->model->saveCheckpoint(path, impl->currentStep);
}

int Trainer::loadCheckpoint(const std::string& path) {
    impl->currentStep = impl->model->loadCheckpoint(path);
    impl->resetCameraPrefetcher();
    return impl->currentStep;
}

int Trainer::splatCount() const {
    return (int)impl->model->means.size(0);
}

int Trainer::iteration() const {
    return impl->currentStep;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

void sync() { msplat_gpu_sync(); }
void cleanup() { cleanup_msplat_metal(); }

} // namespace msplat

// ── C API (for Swift interop) ───────────────────────────────────────────────

#include "msplat_c_api.h"

static thread_local std::string g_last_error;

static void clearLastError() {
    g_last_error.clear();
}

static void setLastError(const std::exception& error) {
    g_last_error = error.what();
}

static void setLastError(const char* error) {
    g_last_error = error;
}

template <typename T, typename F>
static T callCatching(T fallback, const char* unknownError, F&& f) {
    try {
        clearLastError();
        return f();
    } catch (const std::exception& error) {
        setLastError(error);
    } catch (...) {
        setLastError(unknownError);
    }
    return fallback;
}

template <typename F>
static void callCatchingVoid(const char* unknownError, F&& f) {
    try {
        clearLastError();
        f();
    } catch (const std::exception& error) {
        setLastError(error);
    } catch (...) {
        setLastError(unknownError);
    }
}

static msplat::Config configFromC(MsplatConfig c) {
    msplat::Config cfg;
    cfg.iterations = c.iterations;
    cfg.shDegree = c.shDegree;
    cfg.shDegreeInterval = c.shDegreeInterval;
    cfg.ssimWeight = c.ssimWeight;
    cfg.numDownscales = c.numDownscales;
    cfg.resolutionSchedule = c.resolutionSchedule;
    cfg.refineEvery = c.refineEvery;
    cfg.warmupLength = c.warmupLength;
    cfg.resetAlphaEvery = c.resetAlphaEvery;
    cfg.densifyGradThresh = c.densifyGradThresh;
    cfg.densifySizeThresh = c.densifySizeThresh;
    cfg.stopScreenSizeAt = c.stopScreenSizeAt;
    cfg.splitScreenSize = c.splitScreenSize;
    cfg.growthStopIter = c.growthStopIter;
    cfg.maxSplats = c.maxSplats;
    cfg.growthSelectFraction = c.growthSelectFraction;
    cfg.lrMean = c.lrMean;
    cfg.lrMeanEnd = c.lrMeanEnd;
    cfg.lrScale = c.lrScale;
    cfg.lrScaleEnd = c.lrScaleEnd;
    cfg.lrRotation = c.lrRotation;
    cfg.lrCoeffsDc = c.lrCoeffsDc;
    cfg.lrCoeffsShScale = c.lrCoeffsShScale;
    cfg.lrOpacity = c.lrOpacity;
    cfg.lpipsLossWeight = c.lpipsLossWeight;
    cfg.randomInitSceneScale = c.randomInitSceneScale;
    cfg.reduceSecondMoment = c.reduceSecondMoment;
    cfg.imagePrefetchWorkers = c.imagePrefetchWorkers;
    cfg.matchAlphaWeight = c.matchAlphaWeight;
    cfg.backgroundNoiseStrength = c.backgroundNoiseStrength;
    cfg.opacityDecay = c.opacityDecay;
    cfg.scaleDecay = c.scaleDecay;
    cfg.meanNoiseWeight = c.meanNoiseWeight;
    cfg.keepCrs = c.keepCrs;
    cfg.renderMip = c.renderMip;
    cfg.downscaleFactor = c.downscaleFactor;
    memcpy(cfg.bgColor, c.bgColor, sizeof(cfg.bgColor));
    return cfg;
}

MsplatDataset msplat_dataset_create(const char* path, float downscaleFactor,
                                     bool evalMode, int testEvery) {
    try {
        clearLastError();
        if (!path) {
            throw std::runtime_error("msplat dataset path is null");
        }
        auto* ds = new msplat::Dataset(std::string(path), downscaleFactor, evalMode, testEvery);
        return static_cast<MsplatDataset>(ds);
    } catch (const std::exception& error) {
        setLastError(error);
        return nullptr;
    } catch (...) {
        setLastError("unknown msplat dataset creation error");
        return nullptr;
    }
}

void msplat_dataset_destroy(MsplatDataset ds) {
    delete static_cast<msplat::Dataset*>(ds);
}

int msplat_dataset_num_train(MsplatDataset ds) {
    return callCatching<int>(0, "unknown msplat dataset count error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        return static_cast<msplat::Dataset*>(ds)->numTrain();
    });
}

int msplat_dataset_num_test(MsplatDataset ds) {
    return callCatching<int>(0, "unknown msplat dataset count error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        return static_cast<msplat::Dataset*>(ds)->numTest();
    });
}

int msplat_dataset_initial_point_count(MsplatDataset ds) {
    return callCatching<int>(0, "unknown msplat dataset point-count error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        return static_cast<msplat::Dataset*>(ds)->initialPointCount();
    });
}

bool msplat_dataset_camera_has_alpha(MsplatDataset ds, int cameraIndex) {
    return callCatching<bool>(false, "unknown msplat dataset alpha query error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        return static_cast<msplat::Dataset*>(ds)->cameraHasAlpha(cameraIndex);
    });
}

bool msplat_dataset_camera_has_mask(MsplatDataset ds, int cameraIndex) {
    return callCatching<bool>(false, "unknown msplat dataset mask query error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        return static_cast<msplat::Dataset*>(ds)->cameraHasMask(cameraIndex);
    });
}

void msplat_dataset_camera_pose(MsplatDataset ds, int cameraIndex, float camToWorld[16]) {
    callCatchingVoid("unknown msplat dataset pose query error", [&]() {
        if (!ds) throw std::runtime_error("msplat dataset handle is null");
        if (!camToWorld) throw std::runtime_error("msplat camera pose output is null");
        static_cast<msplat::Dataset*>(ds)->cameraPose(cameraIndex, camToWorld);
    });
}

MsplatTrainer msplat_trainer_create(MsplatDataset ds, MsplatConfig config) {
    try {
        clearLastError();
        if (!ds) {
            throw std::runtime_error("msplat trainer creation requires a valid dataset");
        }
        auto* dataset = static_cast<msplat::Dataset*>(ds);
        auto cfg = configFromC(config);
        auto* trainer = new msplat::Trainer(*dataset, cfg);
        return static_cast<MsplatTrainer>(trainer);
    } catch (const std::exception& error) {
        setLastError(error);
        return nullptr;
    } catch (...) {
        setLastError("unknown msplat trainer creation error");
        return nullptr;
    }
}

void msplat_trainer_destroy(MsplatTrainer t) {
    delete static_cast<msplat::Trainer*>(t);
}

MsplatStats msplat_trainer_step(MsplatTrainer t) {
    return callCatching<MsplatStats>({}, "unknown msplat trainer step error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        auto stats = static_cast<msplat::Trainer*>(t)->step();
        return MsplatStats{stats.iteration, stats.splatCount, stats.msPerStep};
    });
}

void msplat_trainer_train(MsplatTrainer t) {
    callCatchingVoid("unknown msplat trainer train error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        static_cast<msplat::Trainer*>(t)->train(0);
    });
}

MsplatEvalMetrics msplat_trainer_evaluate(MsplatTrainer t) {
    return callCatching<MsplatEvalMetrics>({}, "unknown msplat trainer evaluate error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        auto m = static_cast<msplat::Trainer*>(t)->evaluate();
        return MsplatEvalMetrics{m.psnr, m.ssim, m.l1, m.numTest, m.numGaussians};
    });
}

MsplatPixelBuffer msplat_trainer_render(MsplatTrainer t, int cameraIndex, bool useTest) {
    return callCatching<MsplatPixelBuffer>({}, "unknown msplat trainer render error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        auto buf = static_cast<msplat::Trainer*>(t)->render(cameraIndex, useTest);
        MsplatPixelBuffer result{buf.data, buf.width, buf.height};
        buf.data = nullptr; // Transfer ownership to caller
        return result;
    });
}

MsplatPixelBuffer msplat_trainer_render_pose(MsplatTrainer t, const float camToWorld[16], int refCameraIndex) {
    return callCatching<MsplatPixelBuffer>({}, "unknown msplat trainer pose render error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!camToWorld) throw std::runtime_error("msplat render pose input is null");
        auto buf = static_cast<msplat::Trainer*>(t)->renderFromPose(camToWorld, refCameraIndex);
        MsplatPixelBuffer result{buf.data, buf.width, buf.height};
        buf.data = nullptr;
        return result;
    });
}

void msplat_trainer_render_pose_to_buffer(MsplatTrainer t, const float camToWorld[16],
                                      int refCameraIndex, uint8_t* outRGBA,
                                      int* outWidth, int* outHeight) {
    callCatchingVoid("unknown msplat trainer pose buffer render error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!camToWorld) throw std::runtime_error("msplat render pose input is null");
        static_cast<msplat::Trainer*>(t)->renderFromPoseToBuffer(
            camToWorld, refCameraIndex, outRGBA, outWidth, outHeight);
    });
}

void msplat_trainer_export_ply(MsplatTrainer t, const char* path) {
    callCatchingVoid("unknown msplat trainer PLY export error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat PLY export path is null");
        static_cast<msplat::Trainer*>(t)->exportPly(std::string(path));
    });
}

void msplat_trainer_export_lod_ply(MsplatTrainer t, const char* path, int targetCount) {
    callCatchingVoid("unknown msplat trainer LOD PLY export error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat LOD PLY export path is null");
        static_cast<msplat::Trainer*>(t)->exportLodPly(std::string(path), targetCount);
    });
}

void msplat_trainer_decimate_to_lod(MsplatTrainer t, int targetCount) {
    callCatchingVoid("unknown msplat trainer LOD decimation error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        static_cast<msplat::Trainer*>(t)->decimateToLod(targetCount);
    });
}

void msplat_trainer_export_splat(MsplatTrainer t, const char* path) {
    callCatchingVoid("unknown msplat trainer splat export error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat splat export path is null");
        static_cast<msplat::Trainer*>(t)->exportSplat(std::string(path));
    });
}

int msplat_trainer_load_ply(MsplatTrainer t, const char* path) {
    return callCatching<int>(-1, "unknown msplat trainer PLY load error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat PLY load path is null");
        return static_cast<msplat::Trainer*>(t)->loadPly(std::string(path));
    });
}

void msplat_trainer_save_checkpoint(MsplatTrainer t, const char* path) {
    callCatchingVoid("unknown msplat trainer checkpoint save error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat checkpoint save path is null");
        static_cast<msplat::Trainer*>(t)->saveCheckpoint(std::string(path));
    });
}

int msplat_trainer_load_checkpoint(MsplatTrainer t, const char* path) {
    return callCatching<int>(-1, "unknown msplat trainer checkpoint load error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        if (!path) throw std::runtime_error("msplat checkpoint load path is null");
        return static_cast<msplat::Trainer*>(t)->loadCheckpoint(std::string(path));
    });
}

int msplat_trainer_splat_count(MsplatTrainer t) {
    return callCatching<int>(0, "unknown msplat trainer splat-count error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        return static_cast<msplat::Trainer*>(t)->splatCount();
    });
}

int msplat_trainer_iteration(MsplatTrainer t) {
    return callCatching<int>(0, "unknown msplat trainer iteration error", [&]() {
        if (!t) throw std::runtime_error("msplat trainer handle is null");
        return static_cast<msplat::Trainer*>(t)->iteration();
    });
}

const char* msplat_last_error(void) { return g_last_error.c_str(); }
void msplat_sync(void) {
    callCatchingVoid("unknown msplat sync error", []() {
        msplat::sync();
    });
}

void msplat_cleanup(void) {
    callCatchingVoid("unknown msplat cleanup error", []() {
        msplat::cleanup();
    });
}
