#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include "model.hpp"
#include "kdtree_tensor.hpp"
#include "msplat.hpp"
#include "loaders.hpp"

namespace fs = std::filesystem;

static const double C0 = 0.28209479177387814;
static constexpr float MIN_OPACITY = 1.0f / 255.0f;
static constexpr float MIN_QUAT_NORM_SQR = 1e-6f;
static constexpr int RANDOM_INIT_SPLAT_COUNT = 10000;
static constexpr float BOUND_PERCENTILE = 0.8f;

struct InitialSplats {
    std::vector<float> xyz;
    std::vector<float> featuresDc;
    std::vector<float> scales;
    std::vector<float> quats;
    std::vector<float> opacities;
    int64_t count = 0;
    bool random = false;
};

int numShBases(int degree){
    switch(degree){
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 25;
    }
}

// Metrics on CPU MTensor data
float psnr(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double mse = 0;
    for (int64_t i = 0; i < n; i++) { double d = r[i] - g[i]; mse += d * d; }
    mse /= n;
    return 10.0f * std::log10(1.0 / mse);
}

float l1_loss(const MTensor& rendered, const MTensor& gt) {
    int64_t n = rendered.numel();
    const float *r = rendered.data<float>(), *g = gt.data<float>();
    double sum = 0;
    for (int64_t i = 0; i < n; i++) sum += std::abs(r[i] - g[i]);
    return (float)(sum / n);
}

void quantizeRenderedForEval(MTensor& rendered) {
    float *r = rendered.data<float>();
    for (int64_t i = 0; i < rendered.numel(); i++) {
        r[i] = std::round(std::clamp(r[i], 0.0f, 1.0f) * 255.0f) / 255.0f;
    }
}

static float inverseSigmoid(float x) {
    x = std::clamp(x, 1e-6f, 1.0f - 1e-6f);
    return std::log(x / (1.0f - x));
}

static float estimateRandomInitSceneScale(const std::vector<Camera>& cameras) {
    if (cameras.size() < 2) return 1.0f;

    float totalNearest = 0.0f;
    for (size_t i = 0; i < cameras.size(); ++i) {
        float nearest = std::numeric_limits<float>::infinity();
        const float xi = cameras[i].camToWorld[3];
        const float yi = cameras[i].camToWorld[7];
        const float zi = cameras[i].camToWorld[11];
        for (size_t j = 0; j < cameras.size(); ++j) {
            if (i == j) continue;
            const float dx = xi - cameras[j].camToWorld[3];
            const float dy = yi - cameras[j].camToWorld[7];
            const float dz = zi - cameras[j].camToWorld[11];
            nearest = std::min(nearest, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        totalNearest += nearest;
    }

    return std::max(3.0f * totalNearest / static_cast<float>(cameras.size()), 1.0f);
}

static float estimateMedianExtent(const float *xyz, int64_t count) {
    return std::max(PointsTensor::percentileMedianSize(xyz, count, BOUND_PERCENTILE), 0.01f);
}

static float scheduledLr(float start, float end, int step, int maxSteps) {
    float t = maxSteps > 0 ? std::clamp((float)step / (float)maxSteps, 0.0f, 1.0f) : 1.0f;
    start = std::max(start, 1e-12f);
    end = std::max(end, 1e-12f);
    return std::exp(std::log(start) * (1.0f - t) + std::log(end) * t);
}

static float logDet6x6(const float *m) {
    float l[36] = {};
    for (int j = 0; j < 6; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < j; ++k) {
            sum += l[j * 6 + k] * l[j * 6 + k];
        }
        const float diag = m[j * 6 + j] - sum;
        if (diag <= 0.0f || !std::isfinite(diag)) {
            return -std::numeric_limits<float>::infinity();
        }
        l[j * 6 + j] = std::sqrt(diag);
        for (int i = j + 1; i < 6; ++i) {
            sum = 0.0f;
            for (int k = 0; k < j; ++k) {
                sum += l[i * 6 + k] * l[j * 6 + k];
            }
            l[i * 6 + j] = (m[i * 6 + j] - sum) / l[j * 6 + j];
        }
    }

    float logDet = 0.0f;
    for (int i = 0; i < 6; ++i) {
        logDet += std::log(l[i * 6 + i]);
    }
    return 2.0f * logDet;
}

static InitialSplats createRandomInitialSplats(const std::vector<Camera>& cameras,
                                               float sceneScaleOverride,
                                               uint32_t randomSeed) {
    if (cameras.empty()) {
        throw std::runtime_error("Cannot create random splats without cameras");
    }

    const int64_t numPoints = RANDOM_INIT_SPLAT_COUNT;
    const float sceneScale = sceneScaleOverride > 0.0f
        ? sceneScaleOverride
        : estimateRandomInitSceneScale(cameras);
    const float nearDepth = std::max(sceneScale * 0.05f, 1e-6f);
    const float farDepth = std::max(sceneScale, nearDepth * 1.001f);
    const float logNear = std::log(nearDepth);
    const float logFar = std::log(farDepth);
    const float defaultLogScale = std::log(sceneScale / std::cbrt(static_cast<float>(numPoints)));

    InitialSplats init;
    init.count = numPoints;
    init.random = true;
    init.xyz.resize(numPoints * 3);
    init.featuresDc.resize(numPoints * 3);
    init.scales.resize(numPoints * 3, defaultLogScale);
    init.quats.resize(numPoints * 4);
    init.opacities.resize(numPoints);

    std::mt19937 rng(randomSeed);
    std::uniform_int_distribution<size_t> camDist(0, cameras.size() - 1);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> quatDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> opacityDist(inverseSigmoid(0.1f), inverseSigmoid(0.25f));

    for (int64_t i = 0; i < numPoints; ++i) {
        const Camera& cam = cameras[camDist(rng)];
        const float fovX = 2.0f * std::atan(cam.width / (2.0f * cam.fx));
        const float fovY = 2.0f * std::atan(cam.height / (2.0f * cam.fy));
        std::uniform_real_distribution<float> xAngle(-0.5f * fovX, 0.5f * fovX);
        std::uniform_real_distribution<float> yAngle(-0.5f * fovY, 0.5f * fovY);
        std::uniform_real_distribution<float> logDepth(logNear, logFar);

        const float depth = std::exp(logDepth(rng));
        const float localX = std::tan(xAngle(rng)) * depth;
        const float localY = std::tan(yAngle(rng)) * depth;
        const float localZ = -depth;
        const float* m = cam.camToWorld;
        init.xyz[i*3+0] = m[0] * localX + m[1] * localY + m[2]  * localZ + m[3];
        init.xyz[i*3+1] = m[4] * localX + m[5] * localY + m[6]  * localZ + m[7];
        init.xyz[i*3+2] = m[8] * localX + m[9] * localY + m[10] * localZ + m[11];

        for (int c = 0; c < 3; ++c) {
            init.featuresDc[i*3+c] = unit(rng);
        }

        float qx = quatDist(rng);
        float qy = quatDist(rng);
        float qz = quatDist(rng);
        float qw = quatDist(rng);
        float qLen = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        qLen = std::max(qLen, 1e-6f);
        init.quats[i*4+0] = qw / qLen;
        init.quats[i*4+1] = qx / qLen;
        init.quats[i*4+2] = qy / qLen;
        init.quats[i*4+3] = qz / qLen;
        init.opacities[i] = opacityDist(rng);
    }

    return init;
}

// Model constructor
Model::Model(const InputData &inputData, int numCameras,
    int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
    int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
    int maxSteps, bool keepCrs, int growthStopIter,
    int maxSplats, float growthSelectFraction,
    float opacityDecay, float scaleDecay, float meanNoiseWeight,
    float lrMean, float lrMeanEnd, float lrScale, float lrScaleEnd,
    float lrRotation, float lrCoeffsDc, float lrCoeffsShScale, float lrOpacity,
    float randomInitSceneScale, bool reduceSecondMoment,
    uint32_t randomSeed,
    const float* bgColor,
    bool renderMip)
    : numCameras(numCameras), numDownscales(numDownscales), resolutionSchedule(resolutionSchedule),
      shDegree(shDegree), shDegreeInterval(shDegreeInterval),
      refineEvery(refineEvery), warmupLength(warmupLength), resetAlphaEvery(resetAlphaEvery),
      stopSplitAt(std::max(growthStopIter, 0)),
      maxSplats(std::max(maxSplats, 1)),
      growthSelectFraction(std::clamp(growthSelectFraction, 0.0f, 1.0f)),
      densifyGradThresh(densifyGradThresh), densifySizeThresh(densifySizeThresh),
      stopScreenSizeAt(stopScreenSizeAt), splitScreenSize(splitScreenSize),
      maxSteps(maxSteps), opacityDecay(opacityDecay), scaleDecay(scaleDecay),
      meanNoiseWeight(meanNoiseWeight),
      keepCrs(keepCrs), renderMip(renderMip) {
    baseMeansLrInit = lrMean;
    baseMeansLrFinal = lrMeanEnd;
    scales_lr_init = lrScale;
    scales_lr_final = lrScaleEnd;
    rotation_lr = lrRotation;
    coeffs_dc_lr = lrCoeffsDc;
    coeffs_rest_lr = lrCoeffsDc / std::max(lrCoeffsShScale, 1e-6f);
    opacity_lr = lrOpacity;
    this->reduceSecondMoment = reduceSecondMoment;

    InitialSplats randomInit;
    const bool useRandomInit = inputData.points.count == 0;
    if (useRandomInit) {
        randomInit = createRandomInitialSplats(inputData.cameras, randomInitSceneScale, randomSeed);
    }

    int64_t numPoints = useRandomInit ? randomInit.count : inputData.points.count;
    const std::vector<float> &sourceXyz = useRandomInit ? randomInit.xyz : inputData.points.xyz;
    updateMeanLrSceneScale(estimateMedianExtent(sourceXyz.data(), numPoints));

    scale = inputData.scale;
    memcpy(translation, inputData.translation, sizeof(translation));

    // Means: copy xyz directly to GPU
    means = gpu_empty({numPoints, 3}, DType::Float32);
    memcpy(means.data_ptr(), sourceXyz.data(), numPoints * 3 * sizeof(float));

    // Scales: KNN for point-cloud init, Brush-style scene-scale default for random init.
    {
        scales = gpu_empty({numPoints, 3}, DType::Float32);
        float *sp = scales.data<float>();
        if (useRandomInit) {
            memcpy(sp, randomInit.scales.data(), randomInit.scales.size() * sizeof(float));
        } else {
            PointsTensor pt(inputData.points.xyz.data(), numPoints);
            auto sc = pt.scales();  // vector<float> of length numPoints
            for (int64_t i = 0; i < numPoints; i++) {
                float v = std::log(sc[i]);
                sp[i*3] = sp[i*3+1] = sp[i*3+2] = v;
            }
        }
    }

    // Point clouds use identity rotations; random init mirrors Brush's random rotations.
    {
        quats = gpu_empty({numPoints, 4}, DType::Float32);
        float *qp = quats.data<float>();
        if (useRandomInit) {
            memcpy(qp, randomInit.quats.data(), randomInit.quats.size() * sizeof(float));
        } else {
            for (int64_t i = 0; i < numPoints; i++) {
                qp[i*4+0] = 1.0f;
                qp[i*4+1] = 0.0f;
                qp[i*4+2] = 0.0f;
                qp[i*4+3] = 0.0f;
            }
        }
    }

    // SH features: f_dc = rgb2sh(rgb), f_rest = zeros.
    int dimSh = numShBases(shDegree);
    {
        featuresDc = gpu_empty({numPoints, 3}, DType::Float32);
        float *dp = featuresDc.data<float>();
        if (useRandomInit) {
            memcpy(dp, randomInit.featuresDc.data(), randomInit.featuresDc.size() * sizeof(float));
        } else {
            const uint8_t *rgb = inputData.points.rgb.data();
            for (int64_t i = 0; i < numPoints; i++) {
                for (int c = 0; c < 3; c++)
                    dp[i*3+c] = (float)((rgb[i*3+c] / 255.0 - 0.5) / C0);
            }
        }
        featuresRest = gpu_zeros({numPoints, (int64_t)(dimSh - 1), 3}, DType::Float32);
    }

    // Point-cloud opacity matches Brush point init; random init uses Brush's opacity range.
    {
        opacities = gpu_empty({numPoints, 1}, DType::Float32);
        float *op = opacities.data<float>();
        if (useRandomInit) {
            memcpy(op, randomInit.opacities.data(), randomInit.opacities.size() * sizeof(float));
        } else {
            for (int64_t i = 0; i < numPoints; i++) op[i] = 0.0f;
        }
    }

    if (!inputData.initialGaussianPlyPath.empty()) {
        auto g = loadGaussianPly(inputData.initialGaussianPlyPath, scale, translation, keepCrs,
                                 inputData.initialGaussianSubsampleStep);
        means = g.means;
        scales = g.scales;
        quats = g.quats;
        featuresDc = g.featuresDc;
        featuresRest = g.featuresRest;
        opacities = g.opacities;
        if (g.hasRenderMip) this->renderMip = g.renderMip;
        ensureLoadedShCapacity();
        updateMeanLrSceneScaleFromActive();
    }

    // Brush-compatible default background for transparent image compositing.
    backgroundColor = gpu_empty({3}, DType::Float32);
    trainingBackgroundColor = gpu_empty({3}, DType::Float32);
    static const float defaultBg[3] = {0.0f, 0.0f, 0.0f};
    memcpy(backgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    memcpy(trainingBackgroundColor.data_ptr(), bgColor ? bgColor : defaultBg, 3 * sizeof(float));
    setupOptimizers();
}

void Model::setupOptimizers(){
    releaseOptimizers();


    num_active = means.size(0);
    buf_capacity = num_active * 4;
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    const float lr_init[] = {
        means_lr_init, scales_lr_init, rotation_lr,
        coeffs_dc_lr, coeffs_rest_lr, opacity_lr
    };
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = params[g]->shape();
        shape[0] = buf_capacity;
        adam_exp_avg_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_exp_avg_sq_buf[g] = gpu_zeros(shape, DType::Float32);
        adam_lr[g] = lr_init[g];
    }
    adam_step_count = 0;
    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();
}

void Model::releaseOptimizers(){
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g].reset(); adam_exp_avg_sq[g].reset();
        adam_exp_avg_buf[g].reset(); adam_exp_avg_sq_buf[g].reset();
    }
    means_buf.reset(); scales_buf.reset(); quats_buf.reset();
    featuresDc_buf.reset(); featuresRest_buf.reset(); opacities_buf.reset();
    densify_split_flag.reset(); densify_dup_flag.reset();
    densify_split_prefix.reset(); densify_dup_prefix.reset();
    densify_keep_flag.reset(); densify_keep_prefix.reset();
    densify_block_totals.reset(); densify_compact_scratch.reset(); densify_random_samples.reset();
}

void Model::ensureLoadedShCapacity() {
    if (featuresRest.size(1) > 0 || shDegree <= 0) return;
    featuresRest = gpu_zeros({means.size(0), (int64_t)(numShBases(shDegree) - 1), 3}, DType::Float32);
}

void Model::updateMeanLrSceneScale(float sceneScale, int scheduleStep) {
    currentMeanLrSceneScale = std::max(sceneScale, 0.01f);
    meanNoiseMax = currentMeanLrSceneScale;
    means_lr_init = baseMeansLrInit * currentMeanLrSceneScale;
    means_lr_final = baseMeansLrFinal * currentMeanLrSceneScale;
    if (scheduleStep >= 0) {
        adam_lr[0] = scheduledLr(means_lr_init, means_lr_final, scheduleStep, maxSteps);
    }
}

void Model::updateMeanLrSceneScaleFromActive(int scheduleStep) {
    if (!means.defined() || means.size(0) <= 0) return;
    msplat_gpu_sync();
    updateMeanLrSceneScale(estimateMedianExtent(means.data<float>(), means.size(0)), scheduleStep);
}

void Model::schedulersStep(int step){
    adam_lr[0] = scheduledLr(means_lr_init, means_lr_final, step, maxSteps);
    adam_lr[1] = scheduledLr(scales_lr_init, scales_lr_final, step, maxSteps);
}

void Model::refreshViews(){
    means = means_buf.view(num_active);
    scales = scales_buf.view(num_active);
    quats = quats_buf.view(num_active);
    featuresDc = featuresDc_buf.view(num_active);
    featuresRest = featuresRest_buf.view(num_active);
    opacities = opacities_buf.view(num_active);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        adam_exp_avg[g] = adam_exp_avg_buf[g].view(num_active);
        adam_exp_avg_sq[g] = adam_exp_avg_sq_buf[g].view(num_active);
    }
}

void Model::ensureCapacity(int needed){
    if (needed <= buf_capacity) return;
    int new_cap = std::max(needed, buf_capacity * 2);

    auto grow = [&](MTensor &buf) {
        auto shape = buf.shape();
        shape[0] = new_cap;
        MTensor new_buf = gpu_zeros(shape, DType::Float32);
        size_t copy_bytes = num_active * buf.stride0() * sizeof(float);
        memcpy(new_buf.data_ptr(), buf.data_ptr(), copy_bytes);
        buf = new_buf;
    };
    grow(means_buf); grow(scales_buf); grow(quats_buf);
    grow(featuresDc_buf); grow(featuresRest_buf); grow(opacities_buf);
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        grow(adam_exp_avg_buf[g]);
        grow(adam_exp_avg_sq_buf[g]);
    }
    densify_split_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_split_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_dup_prefix = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_flag = gpu_zeros({new_cap}, DType::Int32);
    densify_keep_prefix = gpu_zeros({new_cap}, DType::Int32);
    int max_blocks = (new_cap + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest_buf.stride0();
    densify_compact_scratch = gpu_zeros({(int64_t)new_cap * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({new_cap, 3}, DType::Float32);

    buf_capacity = new_cap;
    refreshViews();
}

int Model::getDownscaleFactor(int step) {
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::max(remaining, 0);
}

namespace {

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float percentileInPlace(std::vector<float>& values, float q) {
    if (values.empty()) return 0.0f;
    q = std::clamp(q, 0.0f, 1.0f);
    size_t idx = std::min(
        values.size() - 1,
        static_cast<size_t>(q * static_cast<float>(values.size())));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

void weightedSampleWithoutReplacement(
    const std::vector<float>& weights,
    int count,
    std::vector<uint8_t>& selected,
    std::mt19937& rng
) {
    if (count <= 0) return;

    std::vector<std::pair<float, int>> keys;
    keys.reserve(weights.size());
    std::uniform_real_distribution<float> uniform(1e-12f, 1.0f);
    for (int i = 0; i < (int)weights.size(); ++i) {
        float weight = weights[i];
        if (selected[i] || !std::isfinite(weight) || weight <= 0.0f) continue;
        keys.emplace_back(std::log(uniform(rng)) / weight, i);
    }

    int take = std::min(count, (int)keys.size());
    if (take <= 0) return;
    auto byDescendingKey = [](const auto& a, const auto& b) {
        return a.first > b.first;
    };
    if (take < (int)keys.size()) {
        std::nth_element(keys.begin(), keys.begin() + take, keys.end(), byDescendingKey);
    }
    for (int i = 0; i < take; ++i) {
        selected[keys[i].second] = 1;
    }
}

}

float Model::prepareBrushRefineFlags(int step, int checkScreen, bool allowGrowth, float cullCenter[3]) {
    msplat_gpu_sync();

    int N = num_active;
    int32_t *split = densify_split_flag.data<int32_t>();
    int32_t *dup = densify_dup_flag.data<int32_t>();
    std::fill(split, split + N, 0);
    std::fill(dup, dup + N, 0);

    const float *meansPtr = means.data<float>();
    const float *scalesPtr = scales.data<float>();
    const float *quatsPtr = quats.data<float>();
    const float *featuresDcPtr = featuresDc.data<float>();
    const float *opacPtr = opacities.data<float>();
    const float *visPtr = visCounts.data<float>();
    const float *gradPtr = xysGradNorm.data<float>();
    const float *screenPtr = max2DSize.data<float>();

    std::vector<float> xs, ys, zs;
    xs.reserve(N); ys.reserve(N); zs.reserve(N);
    for (int i = 0; i < N; ++i) {
        float x = meansPtr[i * 3 + 0];
        float y = meansPtr[i * 3 + 1];
        float z = meansPtr[i * 3 + 2];
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            xs.push_back(x); ys.push_back(y); zs.push_back(z);
        }
    }

    float maxAllowedBounds = std::numeric_limits<float>::max();
    if (!xs.empty()) {
        float loQ = (1.0f - BOUND_PERCENTILE) * 0.5f;
        float hiQ = 1.0f - loQ;
        float loX = percentileInPlace(xs, loQ), hiX = percentileInPlace(xs, hiQ);
        float loY = percentileInPlace(ys, loQ), hiY = percentileInPlace(ys, hiQ);
        float loZ = percentileInPlace(zs, loQ), hiZ = percentileInPlace(zs, hiQ);
        cullCenter[0] = 0.5f * (loX + hiX);
        cullCenter[1] = 0.5f * (loY + hiY);
        cullCenter[2] = 0.5f * (loZ + hiZ);
        float extent = 0.5f * std::max({hiX - loX, hiY - loY, hiZ - loZ});
        maxAllowedBounds = extent * 100.0f;
    } else {
        cullCenter[0] = cullCenter[1] = cullCenter[2] = 0.0f;
    }

    std::vector<uint8_t> pruned(N, 0);
    std::vector<uint8_t> selected(N, 0);
    int prunedCount = 0;
    int thresholdCount = 0;
    float halfMaxDim = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));

    for (int i = 0; i < N; ++i) {
        bool bad = false;
        float opacityRaw = opacPtr[i];
        bad = bad || !std::isfinite(opacityRaw) || sigmoidf(opacityRaw) < MIN_OPACITY;

        for (int c = 0; c < 3; ++c) {
            float logScale = scalesPtr[i * 3 + c];
            float scaleValue = std::exp(logScale);
            bad = bad || !std::isfinite(logScale) || !std::isfinite(scaleValue)
                || scaleValue < 1e-10f || scaleValue > maxAllowedBounds;
        }

        for (int c = 0; c < 3; ++c) {
            float coord = meansPtr[i * 3 + c];
            bad = bad || !std::isfinite(coord)
                || std::fabs(coord - cullCenter[c]) > maxAllowedBounds;
        }
        float quatNormSqr = 0.0f;
        for (int c = 0; c < 4; ++c) {
            float q = quatsPtr[i * 4 + c];
            bad = bad || !std::isfinite(q);
            quatNormSqr += q * q;
        }
        bad = bad || !std::isfinite(quatNormSqr) || quatNormSqr < MIN_QUAT_NORM_SQR;
        for (int c = 0; c < 3; ++c) {
            bad = bad || !std::isfinite(featuresDcPtr[i * 3 + c]);
        }

        pruned[i] = bad ? 1 : 0;
        prunedCount += bad ? 1 : 0;

        float refineWeight = gradPtr[i] * halfMaxDim;
        if (!bad && visPtr[i] > 0.0f && std::isfinite(refineWeight) && refineWeight > densifyGradThresh) {
            thresholdCount++;
        }
    }

    std::mt19937 rng((uint32_t)step);

    std::vector<float> weights(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        if (pruned[i] || visPtr[i] <= 0.0f) continue;
        float opacity = sigmoidf(opacPtr[i]);
        weights[i] = std::isfinite(opacity) ? opacity * visPtr[i] : 0.0f;
    }
    weightedSampleWithoutReplacement(weights, prunedCount, selected, rng);

    int selectedCount = 0;
    for (uint8_t flag : selected) selectedCount += flag ? 1 : 0;
    int currentAfterPrune = N - prunedCount + selectedCount;
    int headroom = std::max(0, maxSplats - currentAfterPrune);

    if (allowGrowth && checkScreen && splitScreenSize > 0.0f && headroom > 0) {
        for (int i = 0; i < N && headroom > 0; ++i) {
            if (pruned[i] || selected[i] || visPtr[i] <= 0.0f) continue;
            if (screenPtr[i] > splitScreenSize) {
                selected[i] = 1;
                headroom--;
            }
        }
    }

    int growCount = 0;
    if (allowGrowth) {
        growCount = (int)std::round((float)thresholdCount * growthSelectFraction);
        growCount = std::max(0, growCount - prunedCount);
        growCount = std::min(growCount, headroom);
    }
    if (growCount > 0) {
        std::fill(weights.begin(), weights.end(), 0.0f);
        for (int i = 0; i < N; ++i) {
            if (pruned[i] || selected[i] || visPtr[i] <= 0.0f) continue;
            float refineWeight = gradPtr[i] * halfMaxDim;
            if (std::isfinite(refineWeight) && refineWeight > densifyGradThresh) {
                weights[i] = refineWeight;
            }
        }
        weightedSampleWithoutReplacement(weights, growCount, selected, rng);
    }

    for (int i = 0; i < N; ++i) {
        split[i] = selected[i] ? 1 : 0;
        dup[i] = 0;
    }

    return maxAllowedBounds;
}

void Model::afterTrain(int step, int phaseStep, int phaseTotal){
    if (!radii.defined()) return;

    int refineStep = phaseStep > 0 ? phaseStep : step;
    int totalForPhase = phaseTotal > 0 ? phaseTotal : maxSteps;
    float phaseProgress = totalForPhase > 0
        ? std::clamp(static_cast<float>(refineStep) / static_cast<float>(totalForPhase), 0.0f, 1.0f)
        : 0.0f;
    if (refineStep % refineEvery == 0 && refineStep > warmupLength && phaseProgress <= 0.95f){
        bool resetEnabled = resetAlphaEvery > 0;
        int resetInterval = resetEnabled ? resetAlphaEvery * refineEvery : 0;
        bool allowGrowth = step < stopSplitAt && num_active < maxSplats;
        if (allowGrowth && resetEnabled) {
            allowGrowth = refineStep % resetInterval > numCameras + refineEvery;
        }

        {
            int numPointsBefore = num_active;
            ensureCapacity(3 * num_active);  // worst case: every gaussian splits

            // Fill random samples for splits (CPU randn, shared memory)
            {
                std::mt19937 rng(step);
                static constexpr float splitOffsetStd = 0.7071067811865476f;
                std::normal_distribution<float> dist(0.0f, splitOffsetStd);
                float *p = densify_random_samples.data<float>();
                for (int64_t i = 0; i < num_active * 3; i++) p[i] = dist(rng);
            }

            float half_max_dim = 0.5f * static_cast<float>((std::max)(lastWidth, lastHeight));
            int check_screen = (allowGrowth && step < stopScreenSizeAt) ? 1 : 0;
            bool checkHuge = resetEnabled && step > refineEvery * resetAlphaEvery;
            int fr_stride = (int)featuresRest_buf.stride0();
            float cullCenter[3] = {};
            float maxAllowedBounds = prepareBrushRefineFlags(step, check_screen, allowGrowth, cullCenter);
            int densifyMaxCount = std::max(maxSplats, 2 * num_active);

            int new_count = msplat_densify(
                num_active, buf_capacity,
                densifyGradThresh, densifySizeThresh, splitScreenSize, check_screen,
                growthSelectFraction, (uint32_t)step, densifyMaxCount,
                MIN_OPACITY, maxAllowedBounds, 0.15f, checkHuge ? 1 : 0,
                cullCenter, maxAllowedBounds, 1,
                xysGradNorm, visCounts, max2DSize, half_max_dim,
                means_buf, scales_buf, quats_buf,
                featuresDc_buf, featuresRest_buf, opacities_buf, fr_stride,
                adam_exp_avg_buf, adam_exp_avg_sq_buf,
                densify_split_flag, densify_dup_flag,
                densify_split_prefix, densify_dup_prefix,
                densify_keep_flag, densify_keep_prefix,
                densify_block_totals, densify_compact_scratch,
                densify_random_samples
            );

            if (new_count <= 0 && numPointsBefore > 0) {
                throw std::runtime_error("Densification would cull all active Gaussians; aborting before the model becomes empty.");
            }
            num_active = new_count;
            refreshViews();
            updateMeanLrSceneScaleFromActive(step);
            std::cout << "Densified: " << numPointsBefore << " -> " << num_active << " gaussians" << std::endl;
        }

        if (resetEnabled && step < stopSplitAt && refineStep % resetInterval == refineEvery){
            msplat_gpu_sync();
            constexpr float resetLogit = -1.3862943611198906f;
            float *op = opacities.data<float>();
            for (int64_t i = 0; i < opacities.numel(); i++)
                if (op[i] > resetLogit) op[i] = resetLogit;

            adam_exp_avg[5].zero();
            adam_exp_avg_sq[5].zero();
            fprintf(stderr, "Opacity reset at step %d\n", step);
        }

        applyRefineDecay(step);

        xysGradNorm.reset();
        visCounts.reset();
        max2DSize.reset();
    }
}

void Model::applyRefineDecay(int step) {
    float trainT = maxSteps > 0 ? (float)step / (float)maxSteps : 1.0f;
    trainT = std::clamp(trainT, 0.0f, 1.0f);
    float shrinkStrength = 1.0f - trainT;
    float minusOpacity = std::max(opacityDecay, 0.0f) * shrinkStrength;
    float scaleFactor = 1.0f - std::max(scaleDecay, 0.0f) * shrinkStrength;
    if (minusOpacity <= 0.0f && scaleFactor >= 1.0f) return;

    msplat_gpu_sync();

    if (minusOpacity > 0.0f) {
        float *op = opacities.data<float>();
        for (int64_t i = 0; i < opacities.numel(); ++i) {
            float alpha = 1.0f / (1.0f + std::exp(-op[i]));
            alpha = std::clamp(alpha - minusOpacity, 1e-12f, 1.0f - 1e-12f);
            op[i] = std::log(alpha / (1.0f - alpha));
        }
    }

    if (scaleFactor < 1.0f) {
        float logScaleDelta = std::log(std::max(scaleFactor, 1e-12f));
        float *sc = scales.data<float>();
        for (int64_t i = 0; i < scales.numel(); ++i) {
            sc[i] += logScaleDelta;
        }
    }
}

void Model::save(const std::string &filename, int step) {
    std::string ext = fs::path(filename).extension().string();
    if (ext == ".splat")
        saveSplat(filename);
    else
        savePly(filename, step);
    fprintf(stderr, "Saved %s\n", filename.c_str());
}

void Model::savePly(const std::string &filename, int step){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs, renderMip};
    saveGaussianPly(filename, p, step);
}

void Model::saveLodPly(const std::string &filename, int step, int64_t targetCount){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs, renderMip};
    GaussianLodStats stats;
    GaussianLodStats *statsPtr = nullptr;
    if (visCounts.defined() && xysGradNorm.defined() && max2DSize.defined()
        && visCounts.size(0) == means.size(0)
        && xysGradNorm.size(0) == means.size(0)
        && max2DSize.size(0) == means.size(0)) {
        msplat_gpu_sync();
        stats.visCounts = visCounts.data<float>();
        stats.xysGradNorm = xysGradNorm.data<float>();
        stats.max2DSize = max2DSize.data<float>();
        statsPtr = &stats;
    }
    saveGaussianLodPly(filename, p, step, targetCount, statsPtr);
}

std::vector<float> Model::computePupLodScores(std::vector<Camera> &cams) {
    const int numPoints = static_cast<int>(means.size(0));
    if (numPoints <= 0) return {};
    if (cams.empty()) {
        throw std::runtime_error("Cannot compute PUP LOD scores without training cameras");
    }

    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    MTensor pupHessian = gpu_zeros({numPoints, 36}, DType::Float32);
    MTensor tmpVis = gpu_zeros({numPoints}, DType::Float32);
    MTensor tmpGrad = gpu_zeros({numPoints}, DType::Float32);
    MTensor tmpScreen = gpu_zeros({numPoints}, DType::Float32);

    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    MTensor adamP[N_ADAM_GROUPS];
    MTensor tempAvg[N_ADAM_GROUPS];
    MTensor tempSq[N_ADAM_GROUPS];
    float zeroSteps[N_ADAM_GROUPS] = {};
    float adamBc2[N_ADAM_GROUPS] = {};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adamP[i] = *params[i];
        tempAvg[i] = gpu_zeros(params[i]->shape(), DType::Float32);
        tempSq[i] = gpu_zeros(params[i]->shape(), DType::Float32);
        adamBc2[i] = 1.0f;
    }

    const float bg[3] = {0.0f, 0.0f, 0.0f};
    memcpy(trainingBackgroundColor.data_ptr(), bg, 3 * sizeof(float));

    for (size_t viewIndex = 0; viewIndex < cams.size(); ++viewIndex) {
        Camera &cam = cams[viewIndex];
        std::cout << "PUP scoring: view " << (viewIndex + 1) << "/" << cams.size() << std::endl;
        cam.ensureImageLoaded();
        auto s = prepareCam(cam, maxSteps, 1);
        lastHeight = s.height;
        lastWidth = s.width;

        MTensor gt = cam.getGPUImage(1, bg);
        MTensor alpha;
        MTensor *alphaTarget = nullptr;
        float alphaLossWeight = 0.0f;
        int pupChannels = 3;
        if (cam.hasLossMask() || cam.imageHasAlpha()) {
            alpha = cam.getGPULossMask(1);
            alphaTarget = &alpha;
            alphaLossWeight = 0.25f;
            pupChannels = 4;
        }
        MTensor &unusedMask = gt;
        MTensor &alphaTargetTensor = alphaTarget ? *alphaTarget : gt;
        const float lossInvN = 1.0f / static_cast<float>(s.height * s.width * pupChannels);
        const float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
        const float invWidth = 1.0f / static_cast<float>((std::max)(lastWidth, 1));
        const float invHeight = 1.0f / static_cast<float>((std::max)(lastHeight, 1));

        msplat_train_step(
            numPoints, means, scales, 1.0f,
            quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
            s.height, s.width, s.tileBounds, 0.01f,
            s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
            opacities, trainingBackgroundColor, renderMip ? 1 : 0,
            gt, unusedMask, 0,
            alphaTargetTensor, alphaTarget ? 1 : 0, alphaLossWeight,
            window2d, 0.0f, 0.0f,
            lossInvN, (int)featuresRest.size(-2),
            N_ADAM_GROUPS,
            adamP, tempAvg, tempSq,
            zeroSteps, adamBc2,
            adam_beta1, adam_beta2, adam_eps,
            reduceSecondMoment ? 1 : 0,
            tmpVis, tmpGrad, tmpScreen, invMaxDim, invWidth, invHeight,
            &pupHessian);
        msplat_commit();
    }

    msplat_gpu_sync();
    MTensor hessianCpu = pupHessian.cpu();
    const float *hessian = hessianCpu.data<float>();
    std::vector<float> scores(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        scores[i] = logDet6x6(hessian + i * 36);
    }

    pupHessian.reset();
    tmpVis.reset();
    tmpGrad.reset();
    tmpScreen.reset();
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        tempAvg[i].reset();
        tempSq[i].reset();
    }

    return scores;
}

void Model::decimateToLod(int64_t targetCount){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs, renderMip};
    GaussianLodStats stats;
    GaussianLodStats *statsPtr = nullptr;
    if (visCounts.defined() && xysGradNorm.defined() && max2DSize.defined()
        && visCounts.size(0) == means.size(0)
        && xysGradNorm.size(0) == means.size(0)
        && max2DSize.size(0) == means.size(0)) {
        msplat_gpu_sync();
        stats.visCounts = visCounts.data<float>();
        stats.xysGradNorm = xysGradNorm.data<float>();
        stats.max2DSize = max2DSize.data<float>();
        statsPtr = &stats;
    }

    auto g = decimateGaussians(p, targetCount, statsPtr);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    xysGradNorm.reset();
    visCounts.reset();
    max2DSize.reset();
    updateMeanLrSceneScaleFromActive();
    setupOptimizers();
}

void Model::decimateToLod(int64_t targetCount, const std::vector<float> &scores){
    if ((int64_t)scores.size() != means.size(0)) {
        throw std::runtime_error("PUP LOD score count does not match active Gaussian count");
    }

    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs, renderMip};
    GaussianLodStats stats;
    stats.pupScores = scores.data();

    auto g = decimateGaussians(p, targetCount, &stats);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    xysGradNorm.reset();
    visCounts.reset();
    max2DSize.reset();
    updateMeanLrSceneScaleFromActive();
    setupOptimizers();
}

void Model::saveSplat(const std::string &filename){
    GaussianParams p{means, scales, quats, featuresDc, featuresRest, opacities,
                     scale, {translation[0], translation[1], translation[2]}, keepCrs, renderMip};
    saveGaussianSplat(filename, p);
}

int Model::loadPly(const std::string &filename){
    auto g = loadGaussianPly(filename, scale, translation, keepCrs);
    means = g.means;
    scales = g.scales;
    quats = g.quats;
    featuresDc = g.featuresDc;
    featuresRest = g.featuresRest;
    opacities = g.opacities;
    if (g.hasRenderMip) renderMip = g.renderMip;
    ensureLoadedShCapacity();
    updateMeanLrSceneScaleFromActive();
    setupOptimizers();
    return g.step;
}

// ── Checkpoint save/load ────────────────────────────────────────────────────

static constexpr uint32_t CKPT_MAGIC = 0x4C50534D; // "MSPL"
static constexpr uint32_t CKPT_VERSION = 2;
static constexpr uint32_t CKPT_MIN_VERSION = 1;

static void writeTensor(std::ofstream &f, MTensor &t) {
    uint32_t ndim = t.ndim();
    f.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    for (int i = 0; i < (int)ndim; i++) {
        int64_t s = t.size(i);
        f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    }
    uint64_t bytes = t.nbytes();
    f.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    f.write(reinterpret_cast<const char*>(t.data_ptr()), bytes);
}

static MTensor readTensor(std::ifstream &f) {
    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++)
        f.read(reinterpret_cast<char*>(&shape[i]), sizeof(int64_t));
    uint64_t bytes;
    f.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
    MTensor t = gpu_empty(shape, DType::Float32);
    f.read(reinterpret_cast<char*>(t.data_ptr()), bytes);
    return t;
}

void Model::saveCheckpoint(const std::string &filename, int step) {
    msplat_gpu_sync();

    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file for writing: " + filename);

    // Header
    f.write(reinterpret_cast<const char*>(&CKPT_MAGIC), sizeof(CKPT_MAGIC));
    f.write(reinterpret_cast<const char*>(&CKPT_VERSION), sizeof(CKPT_VERSION));

    // Scalar state
    uint32_t u;
    u = (uint32_t)step;            f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)num_active;      f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)shDegree;        f.write(reinterpret_cast<const char*>(&u), sizeof(u));
    u = (uint32_t)adam_step_count;  f.write(reinterpret_cast<const char*>(&u), sizeof(u));

    // Adam learning rates
    f.write(reinterpret_cast<const char*>(adam_lr), sizeof(adam_lr));
    f.write(reinterpret_cast<const char*>(&means_lr_init), sizeof(means_lr_init));
    f.write(reinterpret_cast<const char*>(&means_lr_final), sizeof(means_lr_final));
    f.write(reinterpret_cast<const char*>(&scales_lr_init), sizeof(scales_lr_init));
    f.write(reinterpret_cast<const char*>(&scales_lr_final), sizeof(scales_lr_final));

    // Gaussian parameters (views — only num_active elements)
    writeTensor(f, means);
    writeTensor(f, scales);
    writeTensor(f, quats);
    writeTensor(f, featuresDc);
    writeTensor(f, featuresRest);
    writeTensor(f, opacities);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg[g]);
    for (int g = 0; g < N_ADAM_GROUPS; g++) writeTensor(f, adam_exp_avg_sq[g]);

    f.close();
    std::cout << "Checkpoint saved: " << filename << " (step " << step
              << ", " << num_active << " gaussians, "
              << fs::file_size(filename) / (1024*1024) << " MB)" << std::endl;
}

int Model::loadCheckpoint(const std::string &filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open checkpoint file: " + filename);

    // Header
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != CKPT_MAGIC) throw std::runtime_error("Not a valid msplat checkpoint file");
    if (version < CKPT_MIN_VERSION || version > CKPT_VERSION) {
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));
    }

    // Scalar state
    uint32_t step, numPts, shDeg, adamSteps;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    f.read(reinterpret_cast<char*>(&numPts), sizeof(numPts));
    f.read(reinterpret_cast<char*>(&shDeg), sizeof(shDeg));
    f.read(reinterpret_cast<char*>(&adamSteps), sizeof(adamSteps));

    f.read(reinterpret_cast<char*>(adam_lr), sizeof(adam_lr));
    f.read(reinterpret_cast<char*>(&means_lr_init), sizeof(means_lr_init));
    f.read(reinterpret_cast<char*>(&means_lr_final), sizeof(means_lr_final));
    if (version >= 2) {
        f.read(reinterpret_cast<char*>(&scales_lr_init), sizeof(scales_lr_init));
        f.read(reinterpret_cast<char*>(&scales_lr_final), sizeof(scales_lr_final));
    }
    adam_step_count = (int)adamSteps;

    // Gaussian parameters — read into fresh tensors
    means = readTensor(f);
    scales = readTensor(f);
    quats = readTensor(f);
    featuresDc = readTensor(f);
    featuresRest = readTensor(f);
    opacities = readTensor(f);

    // Optimizer state
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg[g] = readTensor(f);
    for (int g = 0; g < N_ADAM_GROUPS; g++) adam_exp_avg_sq[g] = readTensor(f);

    f.close();

    // Rebuild backing buffers with loaded data (don't call setupOptimizers —
    // it would zero the optimizer state we just loaded)
    num_active = (int)numPts;
    buf_capacity = num_active * 4;

    // Copy gaussian params into oversized backing buffers
    auto allocBuf = [&](MTensor &buf, const MTensor &param) {
        auto shape = param.shape();
        shape[0] = buf_capacity;
        buf = gpu_zeros(shape, DType::Float32);
        memcpy(buf.data_ptr(), param.data_ptr(), param.nbytes());
    };
    allocBuf(means_buf, means);
    allocBuf(scales_buf, scales);
    allocBuf(quats_buf, quats);
    allocBuf(featuresDc_buf, featuresDc);
    allocBuf(featuresRest_buf, featuresRest);
    allocBuf(opacities_buf, opacities);

    // Copy optimizer state into oversized backing buffers
    for (int g = 0; g < N_ADAM_GROUPS; g++) {
        auto shape = adam_exp_avg[g].shape();
        shape[0] = buf_capacity;
        MTensor avg_buf = gpu_zeros(shape, DType::Float32);
        MTensor sq_buf = gpu_zeros(shape, DType::Float32);
        memcpy(avg_buf.data_ptr(), adam_exp_avg[g].data_ptr(), adam_exp_avg[g].nbytes());
        memcpy(sq_buf.data_ptr(), adam_exp_avg_sq[g].data_ptr(), adam_exp_avg_sq[g].nbytes());
        adam_exp_avg_buf[g] = avg_buf;
        adam_exp_avg_sq_buf[g] = sq_buf;
    }

    // Allocate densification scratch buffers
    densify_split_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_split_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_dup_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_flag = gpu_zeros({buf_capacity}, DType::Int32);
    densify_keep_prefix = gpu_zeros({buf_capacity}, DType::Int32);
    int max_blocks = (buf_capacity + 1023) / 1024;
    densify_block_totals = gpu_zeros({max_blocks}, DType::Int32);
    int64_t fr_stride = featuresRest.numel() / featuresRest.size(0);
    densify_compact_scratch = gpu_zeros({(int64_t)buf_capacity * fr_stride}, DType::Float32);
    densify_random_samples = gpu_zeros({buf_capacity, 3}, DType::Float32);

    refreshViews();
    {
        float loadedMeansLrInit = means_lr_init;
        float loadedMeansLrFinal = means_lr_final;
        msplat_gpu_sync();
        currentMeanLrSceneScale = estimateMedianExtent(means.data<float>(), means.size(0));
        meanNoiseMax = currentMeanLrSceneScale;
        if (currentMeanLrSceneScale > 0.0f) {
            baseMeansLrInit = loadedMeansLrInit / currentMeanLrSceneScale;
            baseMeansLrFinal = loadedMeansLrFinal / currentMeanLrSceneScale;
        }
        means_lr_init = loadedMeansLrInit;
        means_lr_final = loadedMeansLrFinal;
    }

    std::cout << "Checkpoint loaded: " << filename << " (step " << step
              << ", " << num_active << " gaussians)" << std::endl;

    return (int)step;
}

Model::CamSetup Model::prepareCam(Camera& cam, int step, int forcedDownscale) {
    const float sf = (float)(forcedDownscale > 0 ? forcedDownscale : getDownscaleFactor(step));
    CamSetup s;
    s.fx = cam.fx / sf; s.fy = cam.fy / sf;
    s.cx = cam.cx / sf; s.cy = cam.cy / sf;
    s.height = static_cast<int>(cam.height / sf);
    s.width = static_cast<int>(cam.width / sf);

    float fovX = 2.0f * std::atan(s.width / (2.0f * s.fx));
    float fovY = 2.0f * std::atan(s.height / (2.0f * s.fy));

    if (!cam.cachedViewMat.defined() || cam.cachedFovX != fovX || cam.cachedFovY != fovY) {
        const float *d = cam.camToWorld;
        float R[3][3], Rinv[3][3], T[3], Tinv[3];
        for (int i = 0; i < 3; i++) {
            R[i][0] = d[i*4+0]; R[i][1] = -d[i*4+1]; R[i][2] = -d[i*4+2]; T[i] = d[i*4+3];
        }
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rinv[i][j] = R[j][i];
        for (int i = 0; i < 3; i++) Tinv[i] = -(Rinv[i][0]*T[0] + Rinv[i][1]*T[1] + Rinv[i][2]*T[2]);
        float vm[16] = { Rinv[0][0],Rinv[0][1],Rinv[0][2],Tinv[0], Rinv[1][0],Rinv[1][1],Rinv[1][2],Tinv[1], Rinv[2][0],Rinv[2][1],Rinv[2][2],Tinv[2], 0,0,0,1 };
        float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        float pm[16] = { 0.001f/r_p,0,0,0, 0,0.001f/t_p,0,0, 0,0,(1000.0f+0.001f)/(1000.0f-0.001f),-1000.0f*0.001f/(1000.0f-0.001f), 0,0,1,0 };
        float pvm[16] = {};
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) pvm[i*4+j] += pm[i*4+k] * vm[k*4+j];

        cam.cachedViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedViewMat.data_ptr(), vm, sizeof(vm));
        cam.cachedProjViewMat = gpu_empty({4, 4}, DType::Float32);
        memcpy(cam.cachedProjViewMat.data_ptr(), pvm, sizeof(pvm));
        cam.cachedCamPos[0] = T[0]; cam.cachedCamPos[1] = T[1]; cam.cachedCamPos[2] = T[2];
        cam.cachedFovX = fovX; cam.cachedFovY = fovY;
    }

    s.degreesToUse = shDegreeInterval > 0
        ? (std::min<int>)(step / shDegreeInterval, shDegree)
        : shDegree;
    int b = featuresRest.size(-2) + 1;
    s.degree = (b <= 1) ? 0 : (b <= 4) ? 1 : (b <= 9) ? 2 : (b <= 16) ? 3 : 4;
    s.tileBounds = std::make_tuple(
        (s.width + BLOCK_X - 1) / BLOCK_X,
        (s.height + BLOCK_Y - 1) / BLOCK_Y, 1);
    s.cam_pos[0] = cam.cachedCamPos[0];
    s.cam_pos[1] = cam.cachedCamPos[1];
    s.cam_pos[2] = cam.cachedCamPos[2];

    return s;
}

MTensor Model::render(Camera& cam, int step, const float *bgColorOverride){
    auto s = prepareCam(cam, step);
    MTensor overrideBackground;
    MTensor &renderBackground = bgColorOverride ? overrideBackground : backgroundColor;
    if (bgColorOverride) {
        overrideBackground = gpu_empty({3}, DType::Float32);
        memcpy(overrideBackground.data_ptr(), bgColorOverride, 3 * sizeof(float));
    }
    return msplat_render(
        means.size(0), means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, renderBackground, renderMip ? 1 : 0);
}

void Model::fullIteration(Camera& cam, int step, MTensor &gt, MTensor *lossMask, float lossMaskMean,
                          MTensor *alphaTarget, float matchAlphaWeight,
                          const float *stepBgColor, float ssimWeight, float lpipsLossWeight,
                          int forcedDownscale){
    auto s = prepareCam(cam, step, forcedDownscale);
    lastHeight = s.height; lastWidth = s.width;
    int numPoints = means.size(0);

    // Initialize SSIM window (once)
    if (!window2d.defined()) {
        auto w = createSSIMWindow(11, 1.5f);
        window2d = gpu_empty({11, 11}, DType::Float32);
        memcpy(window2d.data_ptr(), w.data(), w.size() * sizeof(float));
    }

    adam_step_count++;
    float bc1 = 1.0f - std::pow(adam_beta1, adam_step_count);
    float bc2 = 1.0f - std::pow(adam_beta2, adam_step_count);
    MTensor adam_p[N_ADAM_GROUPS];
    MTensor adam_ea[N_ADAM_GROUPS], adam_eas[N_ADAM_GROUPS];
    float adam_ss[N_ADAM_GROUPS], adam_bc2s[N_ADAM_GROUPS];
    MTensor *params[] = {&means, &scales, &quats, &featuresDc, &featuresRest, &opacities};
    for (int i = 0; i < N_ADAM_GROUPS; ++i) {
        adam_p[i] = *params[i];
        adam_ea[i] = adam_exp_avg[i];
        adam_eas[i] = adam_exp_avg_sq[i];
        adam_ss[i] = adam_lr[i] / bc1;
        adam_bc2s[i] = std::sqrt(bc2);
    }

    if (!xysGradNorm.defined()) {
    
        xysGradNorm = gpu_zeros({numPoints}, DType::Float32);
        visCounts = gpu_zeros({numPoints}, DType::Float32);
        max2DSize = gpu_zeros({numPoints}, DType::Float32);
    }

    float invMaxDim = 1.0f / static_cast<float>((std::max)(lastHeight, lastWidth));
    float invWidth = 1.0f / static_cast<float>((std::max)(lastWidth, 1));
    float invHeight = 1.0f / static_cast<float>((std::max)(lastHeight, 1));
    (void)lossMaskMean;
    float lossInvN = 1.0f / (float)(s.height * s.width * 3);
    MTensor &lossMaskTensor = lossMask ? *lossMask : gt;
    bool useAlphaLoss = alphaTarget && matchAlphaWeight > 0.0f;
    MTensor &alphaTargetTensor = alphaTarget ? *alphaTarget : gt;
    if (stepBgColor) {
        memcpy(trainingBackgroundColor.data_ptr(), stepBgColor, 3 * sizeof(float));
    }

    auto [r, loss] = msplat_train_step(
        numPoints, means, scales, 1.0f,
        quats, cam.cachedViewMat, cam.cachedProjViewMat, s.fx, s.fy, s.cx, s.cy,
        s.height, s.width, s.tileBounds, 0.01f,
        s.degree, s.degreesToUse, s.cam_pos, featuresDc, featuresRest,
        opacities, trainingBackgroundColor, renderMip ? 1 : 0,
        gt, lossMaskTensor, lossMask ? 1 : 0,
        alphaTargetTensor, useAlphaLoss ? 1 : 0, matchAlphaWeight,
        window2d, ssimWeight, lpipsLossWeight,
        lossInvN, (int)featuresRest.size(-2),
        N_ADAM_GROUPS,
        adam_p, adam_ea, adam_eas,
        adam_ss, adam_bc2s,
        adam_beta1, adam_beta2, adam_eps,
        reduceSecondMoment ? 1 : 0,
        visCounts, xysGradNorm, max2DSize, invMaxDim, invWidth, invHeight);

    if (meanNoiseWeight > 0.0f) {
        msplat_apply_mean_noise(numPoints, means, opacities, r,
                                adam_lr[0] * meanNoiseWeight,
                                meanNoiseMax, (uint32_t)step);
    }

    radii = r;
}
