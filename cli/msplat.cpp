#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <array>
#include <stdexcept>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "random_iter.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "bindings.h"

namespace fs = std::filesystem;

static std::string replaceAll(std::string text, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

static std::string datasetNameFromPath(const std::string &projectRoot) {
    fs::path path(projectRoot);
    if (!path.filename().empty()) return path.filename().string();
    if (path.has_parent_path()) return path.parent_path().filename().string();
    return "dataset";
}

static fs::path resolveBrushExportPath(const std::string &projectRoot, const std::string &exportPath) {
    std::string pathText = replaceAll(exportPath, "{dataset}", datasetNameFromPath(projectRoot));
    fs::path path(pathText);
    if (path.is_absolute()) return path;

    fs::path base = fs::path(projectRoot).parent_path();
    if (base.empty()) base = ".";
    return (base / path).lexically_normal();
}

static fs::path exportPathForStep(const std::string &projectRoot, const std::string &exportPath,
                                  const std::string &exportName, const std::string &outputScene,
                                  int step) {
    if (exportPath.empty()) {
        fs::path p(outputScene);
        return p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string()));
    }

    fs::path dir = resolveBrushExportPath(projectRoot, exportPath);
    std::string name = replaceAll(exportName, "{iter}", std::to_string(step));
    if (name.find(".ply") == std::string::npos) name += ".ply";
    return dir / name;
}

static void filterCameras(InputData &inputData, int maxFrames, int subsampleFrames) {
    if (maxFrames <= 0 && subsampleFrames <= 1) return;

    auto filter = [&](std::vector<Camera> &cameras) {
        if (cameras.empty()) return;

        std::vector<Camera> filtered;
        size_t step = static_cast<size_t>(std::max(subsampleFrames, 1));
        size_t limit = maxFrames > 0 ? static_cast<size_t>(maxFrames) : cameras.size();
        filtered.reserve(std::min(cameras.size(), limit));
        for (size_t i = 0; i < cameras.size() && filtered.size() < limit; i += step) {
            filtered.push_back(std::move(cameras[i]));
        }
        if (filtered.empty()) throw std::runtime_error("Camera filtering removed every frame");
        cameras = std::move(filtered);
    };

    filter(inputData.cameras);
    filter(inputData.evalCameras);
}

static void subsamplePoints(InputData &inputData, int subsampleStep) {
    if (subsampleStep <= 1 || inputData.points.count <= 0) return;
    inputData.initialGaussianSubsampleStep = subsampleStep;

    Points filtered;
    filtered.xyz.reserve((inputData.points.count / subsampleStep + 1) * 3);
    filtered.rgb.reserve((inputData.points.count / subsampleStep + 1) * 3);
    for (int64_t i = 0; i < inputData.points.count; i += subsampleStep) {
        filtered.xyz.insert(filtered.xyz.end(),
                            inputData.points.xyz.begin() + i * 3,
                            inputData.points.xyz.begin() + i * 3 + 3);
        if (!inputData.points.rgb.empty()) {
            filtered.rgb.insert(filtered.rgb.end(),
                                inputData.points.rgb.begin() + i * 3,
                                inputData.points.rgb.begin() + i * 3 + 3);
        }
    }
    filtered.count = static_cast<int64_t>(filtered.xyz.size() / 3);
    inputData.points = std::move(filtered);
}

static float cameraDownscaleFactor(const Camera &camera, float downScaleFactor, int maxResolution) {
    float factor = std::max(downScaleFactor, 1.0f);
    if (maxResolution > 0 && camera.width > 0 && camera.height > 0) {
        int maxDim = std::max(camera.width, camera.height);
        if (maxDim > maxResolution) {
            factor = std::max(factor, static_cast<float>(maxDim) / static_cast<float>(maxResolution));
        }
    }
    return factor;
}

int main(int argc, char *argv[]) {
    CLI::App app{"msplat — 3D Gaussian Splatting for Apple Silicon"};
    app.set_version_flag("--version", APP_VERSION);

    // Required
    std::string projectRoot;
    app.add_option("input", projectRoot, "Path to dataset (COLMAP, Nerfstudio, Polycam)")
        ->required()
        ->check(CLI::ExistingDirectory);

    // Output
    std::string outputScene = "splat.ply";
    app.add_option("-o,--output", outputScene, "Output scene path");
    uint32_t seed = 42;
    app.add_option("--seed", seed, "Brush-style random seed");
    int saveEvery = 5000;
    app.add_option("-s,--save-every,--export-every", saveEvery, "Save/export every N steps (-1 to disable)");
    std::string exportPath = "./{dataset}_exports/";
    app.add_option("--export-path", exportPath, "Brush-style export directory, supports {dataset}");
    std::string exportName = "export_{iter}.ply";
    app.add_option("--export-name", exportName, "Brush-style export filename, supports {iter}");
    int lodLevels = 0;
    app.add_option("--lod-levels", lodLevels, "Export N importance-ranked LOD PLY files after training")
        ->check(CLI::Range(0, 16));
    float lodKeepRatio = 0.5f;
    app.add_option("--lod-keep-ratio", lodKeepRatio, "Fraction of splats to keep per LOD level")
        ->check(CLI::Range(0.01f, 1.0f));
    int lodDecimationKeep = 0;
    app.add_option("--lod-decimation-keep", lodDecimationKeep, "Brush-style LOD keep percentage")
        ->check(CLI::Range(1, 100));
    int lodRefineSteps = 5000;
    app.add_option("--lod-refine-steps", lodRefineSteps, "Optimize each decimated LOD for N extra steps")
        ->check(CLI::Range(0, 1000000));
    int lodImageScale = 50;
    app.add_option("--lod-image-scale", lodImageScale, "Image scale percent used during LOD refinement")
        ->check(CLI::Range(1, 100));

    // Resume
    std::string resume;
    app.add_option("--resume", resume, "Resume training from PLY file")
        ->check(CLI::ExistingFile);
    int startIter = 0;
    app.add_option("--start-iter", startIter, "Brush-style iteration to resume from")
        ->check(CLI::NonNegativeNumber);

    // Validation
    bool validate = false;
    app.add_flag("--val", validate, "Withhold a camera for validation");
    std::string valImage = "random";
    app.add_option("--val-image", valImage, "Validation image filename");
    std::string valRender;
    app.add_option("--val-render", valRender, "Directory to render validation images");

    // Evaluation
    bool evalMode = false;
    app.add_flag("--eval", evalMode, "Evaluate on held-out test views");
    int testEvery = 8;
    app.add_option("--test-every", testEvery, "Hold out every Nth image for eval")
        ->check(CLI::Range(2, 100));
    int evalSplitEvery = 0;
    app.add_option("--eval-split-every", evalSplitEvery, "Brush-style eval split period")
        ->check(CLI::Range(2, 100));
    int evalEvery = 1000;
    app.add_option("--eval-every", evalEvery, "Evaluate every N steps (0 to disable periodic eval)")
        ->check(CLI::Range(0, 1000000));
    bool evalSaveToDisk = false;
    app.add_flag("--eval-save-to-disk", evalSaveToDisk, "Save periodic eval renders under export path");

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters,--total-train-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    int maxResolution = 1920;
    app.add_option("--max-resolution", maxResolution, "Brush-style max loaded image resolution (0 disables)")
        ->check(CLI::NonNegativeNumber);
    int maxFrames = 0;
    app.add_option("--max-frames", maxFrames, "Brush-style maximum frames to load (0 disables)")
        ->check(CLI::NonNegativeNumber);
    int subsampleFrames = 1;
    app.add_option("--subsample-frames", subsampleFrames, "Brush-style frame subsampling step")
        ->check(CLI::PositiveNumber);
    int subsamplePointStep = 1;
    app.add_option("--subsample-points", subsamplePointStep, "Brush-style initial point subsampling step")
        ->check(CLI::PositiveNumber);
    std::string alphaModeText;
    app.add_option("--alpha-mode", alphaModeText, "Brush alpha mode override: masked or transparent");
    int numDownscales = 0;
    app.add_option("--num-downscales", numDownscales, "Progressive downscale levels");
    int resolutionSchedule = 3000;
    app.add_option("--resolution-schedule", resolutionSchedule, "Double resolution every N steps");
    int shDegree = 3;
    app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4));
    int shDegreeInterval = 0;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps (0 = full degree immediately)")
        ->check(CLI::NonNegativeNumber);
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    int refineEvery = 200;
    app.add_option("--refine-every", refineEvery, "Densify/prune every N steps");
    int warmupLength = 0;
    app.add_option("--warmup-length", warmupLength, "Steps before first densification");
    int resetAlphaEvery = 0;
    app.add_option("--reset-alpha-every", resetAlphaEvery, "Reset opacity every N refinements, or 0 to disable");
    float densifyGradThresh = 0.0025f;
    app.add_option("--densify-grad-thresh,--growth-grad-threshold", densifyGradThresh, "Gradient threshold for split/dup");
    float densifySizeThresh = 0.01f;
    app.add_option("--densify-size-thresh", densifySizeThresh, "Size threshold (dup vs split)");
    int stopScreenSizeAt = 15000;
    auto *stopScreenSizeAtOption = app.add_option("--stop-screen-size-at", stopScreenSizeAt,
                                                  "Stop splitting large gaussians after N steps (defaults to growth-stop-iter)");
    int growthStopIter = 15000;
    app.add_option("--growth-stop-iter", growthStopIter, "Stop splat growth after this iteration")
        ->check(CLI::Range(0, 1000000));
    int maxSplats = 10000000;
    app.add_option("--max-splats", maxSplats, "Maximum splat count allowed during growth")
        ->check(CLI::Range(1, 100000000));
    float growthSelectFraction = 0.25f;
    app.add_option("--growth-select-fraction", growthSelectFraction, "Fraction of high-gradient splats selected for growth")
        ->check(CLI::Range(0.0f, 1.0f));
    float splitScreenSize = 0.25f;
    app.add_option("--split-screen-size,--split-at-screen-size", splitScreenSize, "Screen-space split threshold");
    float matchAlphaWeight = 0.1f;
    app.add_option("--match-alpha-weight", matchAlphaWeight, "Alpha L1 loss weight for transparent targets")
        ->check(CLI::Range(0.0f, 100.0f));
    float lpipsLossWeight = 0.0f;
    app.add_option("--lpips-loss-weight", lpipsLossWeight, "LPIPS perceptual loss weight")
        ->check(CLI::Range(0.0f, 100.0f));
    float opacityDecay = 0.004f;
    app.add_option("--opac-decay", opacityDecay, "Opacity shrink applied at refinement steps")
        ->check(CLI::Range(0.0f, 1.0f));
    float scaleDecay = 0.002f;
    app.add_option("--scale-decay", scaleDecay, "Scale shrink applied at refinement steps")
        ->check(CLI::Range(0.0f, 1.0f));
    float meanNoiseWeight = 50.0f;
    app.add_option("--mean-noise-weight", meanNoiseWeight, "Low-opacity mean noise weight during growth")
        ->check(CLI::Range(0.0f, 100000.0f));
    float lrMean = 2e-5f;
    app.add_option("--lr-mean", lrMean, "Initial learning rate for mean parameters")
        ->check(CLI::PositiveNumber);
    float lrMeanEnd = 2e-7f;
    app.add_option("--lr-mean-end", lrMeanEnd, "Final learning rate for mean parameters")
        ->check(CLI::PositiveNumber);
    float lrScale = 7e-3f;
    app.add_option("--lr-scale", lrScale, "Initial learning rate for scale parameters")
        ->check(CLI::PositiveNumber);
    float lrScaleEnd = 5e-3f;
    app.add_option("--lr-scale-end", lrScaleEnd, "Final learning rate for scale parameters")
        ->check(CLI::PositiveNumber);
    float lrRotation = 0.002f;
    app.add_option("--lr-rotation", lrRotation, "Learning rate for rotation parameters")
        ->check(CLI::PositiveNumber);
    float lrCoeffsDc = 2e-3f;
    app.add_option("--lr-coeffs-dc", lrCoeffsDc, "Learning rate for base SH coefficients")
        ->check(CLI::PositiveNumber);
    float lrCoeffsShScale = 10.0f;
    app.add_option("--lr-coeffs-sh-scale", lrCoeffsShScale, "Divisor for higher-order SH coefficient learning rate")
        ->check(CLI::PositiveNumber);
    float lrOpacity = 0.012f;
    app.add_option("--lr-opac", lrOpacity, "Learning rate for opacity parameters")
        ->check(CLI::PositiveNumber);
    float randomInitSceneScale = 0.0f;
    app.add_option("--random-init-scene-scale", randomInitSceneScale,
                   "Scene scale for random init when no point cloud exists (0 = estimate from cameras)")
        ->check(CLI::NonNegativeNumber);
    bool reduceSecondMoment = false;
    app.add_flag("--reduce-second-moment", reduceSecondMoment,
                 "Use Brush-style scalar second moment for SH Adam updates");
    float backgroundNoiseStrength = 0.1f;
    app.add_option("--background-noise-strength", backgroundNoiseStrength, "Uniform background jitter strength per training step")
        ->check(CLI::Range(0.0f, 1.0f));
    bool keepCrs = true;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    bool normalizeCrs = false;
    app.add_flag("--normalize-crs", normalizeCrs, "Export msplat's normalized internal coordinate frame");
    bool renderMip = false;
    app.add_flag("--render-mip", renderMip, "Use MIP splatting opacity compensation during training and rendering");
    std::string renderMode;
    app.add_option("--render-mode", renderMode, "Brush render mode: default or mip");
    std::vector<float> bgColor = {0.0f, 0.0f, 0.0f};
    app.add_option("--bg-color,--background-color", bgColor, "Background RGB (0-1), default black")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");

    CLI11_PARSE(app, argc, argv);

    if (normalizeCrs) keepCrs = false;
    if (stopScreenSizeAtOption->count() == 0) stopScreenSizeAt = growthStopIter;
    if (lodDecimationKeep > 0) lodKeepRatio = static_cast<float>(lodDecimationKeep) / 100.0f;
    if (evalSplitEvery > 0) {
        testEvery = evalSplitEvery;
        evalMode = true;
    }
    if (!renderMode.empty()) {
        if (renderMode == "mip") {
            renderMip = true;
        } else if (renderMode == "default") {
            renderMip = false;
        } else {
            std::cerr << "--render-mode must be 'default' or 'mip'" << std::endl;
            return 1;
        }
    }
    AlphaModeOverride alphaMode = AlphaModeOverride::Auto;
    if (!alphaModeText.empty()) {
        if (alphaModeText == "masked" || alphaModeText == "mask") {
            alphaMode = AlphaModeOverride::Masked;
        } else if (alphaModeText == "transparent") {
            alphaMode = AlphaModeOverride::Transparent;
        } else {
            std::cerr << "--alpha-mode must be 'masked' or 'transparent'" << std::endl;
            return 1;
        }
    }
    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);
        filterCameras(inputData, maxFrames, subsampleFrames);
        subsamplePoints(inputData, subsamplePointStep);

        for (auto &cam : inputData.cameras)
            cam.loadImage(cameraDownscaleFactor(cam, downScaleFactor, maxResolution), alphaMode);
        for (auto &cam : inputData.evalCameras)
            cam.loadImage(cameraDownscaleFactor(cam, downScaleFactor, maxResolution), alphaMode);

        std::vector<Camera> cams;
        std::vector<Camera> testCams;
        Camera *valCam = nullptr;

        if (evalMode) {
            auto [train, test] = inputData.splitTrainTest(testEvery);
            cams = train; testCams = test;
            std::cout << "Eval mode: " << cams.size() << " train, " << testCams.size() << " test" << std::endl;
        } else {
            auto [train, val] = inputData.getCameras(validate, valImage);
            cams = train; valCam = val;
        }

        Model model(inputData, cams.size(),
                     numDownscales, resolutionSchedule, shDegree, shDegreeInterval,
                     refineEvery, warmupLength, resetAlphaEvery, densifyGradThresh,
                     densifySizeThresh, stopScreenSizeAt, splitScreenSize,
                     numIters, keepCrs, growthStopIter,
                     maxSplats, growthSelectFraction,
                     opacityDecay, scaleDecay, meanNoiseWeight,
                     lrMean, lrMeanEnd, lrScale, lrScaleEnd,
                     lrRotation, lrCoeffsDc, lrCoeffsShScale, lrOpacity,
                     randomInitSceneScale, reduceSecondMoment,
                     seed,
                     bgColor.data(), renderMip);

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices, seed);
        std::mt19937 bgRng(seed);
        auto sampleBackground = [&]() {
            std::array<float, 3> bg = {bgColor[0], bgColor[1], bgColor[2]};
            if (backgroundNoiseStrength > 0.0f) {
                std::uniform_real_distribution<float> dist(-backgroundNoiseStrength, backgroundNoiseStrength);
                for (float &channel : bg) channel = std::clamp(channel + dist(bgRng), 0.0f, 1.0f);
            }
            return bg;
        };
        auto evaluationImageDir = [&](int evalStep) {
            fs::path base;
            if (exportPath.empty()) {
                base = fs::path(outputScene).parent_path();
                if (base.empty()) base = ".";
            } else {
                base = resolveBrushExportPath(projectRoot, exportPath);
            }
            return base / ("eval_" + std::to_string(evalStep));
        };
        auto runEvaluation = [&](int evalStep, bool saveImages) {
            if (!evalMode || testCams.empty()) return;

            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();
            const float evalBg[3] = {0.0f, 0.0f, 0.0f};
            fs::path imageDir;
            if (saveImages) {
                imageDir = evaluationImageDir(evalStep);
                fs::create_directories(imageDir);
            }

            std::cout << "\n=== Evaluation (" << nTest << " test views";
            if (evalStep != numIters) std::cout << ", step " << evalStep;
            std::cout << ") ===" << std::endl;

            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], evalStep, evalBg);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = testCams[i].getGPUImage(model.getDownscaleFactor(evalStep), evalBg).cpu();
                quantizeRenderedForEval(rgb_cpu);

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                if (saveImages) {
                    Image evalImg;
                    evalImg.width = (int)rgb_cpu.size(1);
                    evalImg.height = (int)rgb_cpu.size(0);
                    evalImg.data.resize(evalImg.width * evalImg.height * 3);
                    memcpy(evalImg.ptr(), rgb_cpu.data_ptr(), evalImg.data.size() * sizeof(float));
                    imwriteRGB((imageDir / (fs::path(testCams[i].filePath).stem().string() + ".png")).string(), evalImg);
                }

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        };

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;
        if (startIter > 0) step = static_cast<size_t>(startIter) + 1;

        bool benchmarking = std::getenv("BENCHMARK") != nullptr;
        int bench_warmup = 50;
        std::vector<double> bench_iter_ms, bench_cpu_ms, bench_drain_ms;
        if (benchmarking) {
            bench_iter_ms.reserve(numIters);
            bench_cpu_ms.reserve(numIters);
            bench_drain_ms.reserve(numIters);
        }
        auto cpu_now = []() { return std::chrono::high_resolution_clock::now(); };

        auto bench_start = cpu_now();
        for (; step <= (size_t)numIters; step++) {
            Camera &cam = cams[camsIter.next()];

            auto iter_start = cpu_now();
            int downscale = model.getDownscaleFactor(step);
            std::array<float, 3> stepBg = sampleBackground();
            MTensor gt = cam.getGPUImage(downscale, stepBg.data());
            MTensor *lossMask = nullptr;
            MTensor mask;
            float lossMaskMean = 1.0f;
            MTensor *alphaTarget = nullptr;
            MTensor alpha;
            if (cam.hasLossMask()) {
                mask = cam.getGPULossMask(downscale);
                lossMask = &mask;
                lossMaskMean = cam.getLossMaskMean(downscale);
            } else if (cam.imageHasAlpha()) {
                alpha = cam.getGPULossMask(downscale);
                alphaTarget = &alpha;
            }
            model.fullIteration(cam, step, gt, lossMask, lossMaskMean,
                                alphaTarget, matchAlphaWeight, stepBg.data(), ssimWeight,
                                lpipsLossWeight);
            model.schedulersStep(step);
            model.afterTrain(step);
            msplat_commit();

            if (benchmarking && step > (size_t)bench_warmup) {
                auto pre_sync = cpu_now();
                msplat_gpu_sync();
                auto iter_end = cpu_now();
                double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(pre_sync - iter_start).count() / 1000.0;
                double drain_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - pre_sync).count() / 1000.0;
                bench_iter_ms.push_back(iter_ms);
                bench_cpu_ms.push_back(cpu_ms);
                bench_drain_ms.push_back(drain_ms);
            }

            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path p = exportPathForStep(projectRoot, exportPath, exportName, outputScene, (int)step);
                if (p.has_parent_path()) fs::create_directories(p.parent_path());
                model.save(p.string(), step);
            }

            if (evalEvery > 0 && step % (size_t)evalEvery == 0) {
                runEvaluation((int)step, evalSaveToDisk);
            }

            if (!valRender.empty() && step % 10 == 0) {
                MTensor rgb = model.render(*valCam, step);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                Image valImg;
                valImg.width = (int)rgb_cpu.size(1);
                valImg.height = (int)rgb_cpu.size(0);
                valImg.data.resize(valImg.width * valImg.height * 3);
                memcpy(valImg.ptr(), rgb_cpu.data_ptr(), valImg.data.size() * sizeof(float));
                imwriteRGB((fs::path(valRender) / (std::to_string(step) + ".png")).string(), valImg);
            }
        }

        if (benchmarking && !bench_iter_ms.empty()) {
            auto bench_end = cpu_now();
            double total_s = std::chrono::duration_cast<std::chrono::milliseconds>(bench_end - bench_start).count() / 1000.0;
            size_t n = bench_iter_ms.size();
            std::vector<double> sorted = bench_iter_ms;
            std::sort(sorted.begin(), sorted.end());
            double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
            double mean = sum / n;
            double median = (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2];
            double sq_sum = 0;
            for (double v : sorted) sq_sum += (v - mean) * (v - mean);
            double stddev = std::sqrt(sq_sum / n);

            std::cout << "\n=== Benchmark (" << n << " iters, " << bench_warmup << " warmup, " << total_s << "s total) ===\n";
            std::cout << "  mean:   " << mean   << " ms/iter\n";
            std::cout << "  median: " << median  << " ms/iter\n";
            std::cout << "  stddev: " << stddev  << " ms/iter\n";
            std::cout << "  p5:     " << sorted[(size_t)(n * 0.05)] << " ms/iter\n";
            std::cout << "  p95:    " << sorted[(size_t)(n * 0.95)] << " ms/iter\n";
            std::cout << "  min:    " << sorted.front() << " ms/iter\n";
            std::cout << "  max:    " << sorted.back()  << " ms/iter\n";
            std::cout << "  wall:   " << total_s << "s for " << numIters << " iters\n";

            auto stats = [](std::vector<double> &v) {
                std::vector<double> s = v;
                std::sort(s.begin(), s.end());
                size_t n = s.size();
                double sum = std::accumulate(s.begin(), s.end(), 0.0);
                double med = (n % 2 == 0) ? (s[n/2-1] + s[n/2]) / 2.0 : s[n/2];
                return std::make_pair(sum / n, med);
            };
            auto [cpu_mean, cpu_med] = stats(bench_cpu_ms);
            auto [drain_mean, drain_med] = stats(bench_drain_ms);
            std::cout << "\n  --- CPU dispatch vs GPU drain ---\n";
            std::cout << "  cpu dispatch:  mean=" << cpu_mean << "  median=" << cpu_med << " ms\n";
            std::cout << "  gpu drain:     mean=" << drain_mean << "  median=" << drain_med << " ms\n";
            std::cout << "  gpu fraction:  " << (drain_med / median * 100) << "%\n";

            // GPU timing from completion handlers (PROFILE_GPU=1)
            std::vector<double> gpu_times;
            msplat_drain_gpu_times(gpu_times);
            if (!gpu_times.empty()) {
                auto [gpu_mean, gpu_med] = stats(gpu_times);
                std::vector<double> gs = gpu_times;
                std::sort(gs.begin(), gs.end());
                std::cout << "\n  --- GPU kernel time (from CB completion handlers) ---\n";
                std::cout << "  gpu exec:   mean=" << gpu_mean << "  median=" << gpu_med << " ms\n";
                std::cout << "  gpu p5:     " << gs[(size_t)(gs.size() * 0.05)] << " ms\n";
                std::cout << "  gpu p95:    " << gs[(size_t)(gs.size() * 0.95)] << " ms\n";
                std::cout << "  gpu min:    " << gs.front() << " ms\n";
                std::cout << "  gpu max:    " << gs.back() << " ms\n";
                std::cout << "  n_cbs:      " << gs.size() << "\n";
            }

            // Per-stage GPU timing (PROFILE_STAGES=1)
            constexpr int MAX_STAGES = 16;
            std::vector<double> stage_times[MAX_STAGES];
            const char* stage_names[MAX_STAGES] = {};
            int n_stages = 0;
            msplat_drain_stage_times(stage_times, MAX_STAGES, n_stages, stage_names);
            bool has_stage_data = false;
            for (int i = 0; i < n_stages; i++) if (!stage_times[i].empty()) { has_stage_data = true; break; }
            if (has_stage_data) {
                std::cout << "\n  --- Per-stage GPU time (Metal timestamp counters) ---\n";
                double total_med = 0;
                for (int i = 0; i < n_stages; i++) {
                    if (stage_times[i].empty()) continue;
                    auto [s_mean, s_med] = stats(stage_times[i]);
                    total_med += s_med;
                    std::cout << "  " << std::left << std::setw(22) << stage_names[i]
                              << "median=" << std::fixed << std::setprecision(3) << s_med
                              << "ms  mean=" << s_mean << "ms  (" << stage_times[i].size() << " samples)\n";
                }
                std::cout << "  " << std::left << std::setw(22) << "TOTAL (sum medians)"
                          << std::fixed << std::setprecision(3) << total_med << "ms\n";
            }
            std::cout << "\n";
        }

        inputData.saveCameras((fs::path(outputScene).parent_path() / "cameras.json").string(), keepCrs);
        model.save(outputScene, numIters);
        if (lodLevels > 0) {
            fs::path outputPath(outputScene);
            fs::path dir = outputPath.parent_path();
            if (dir.empty()) dir = ".";
            std::string stem = outputPath.stem().string();
            for (int level = 1; level <= lodLevels; level++) {
                int64_t sourceCount = model.means.size(0);
                int64_t targetCount = std::max<int64_t>(1, (int64_t)(sourceCount * lodKeepRatio));
                fs::path lodPath = dir / (stem + "_lod" + std::to_string(level) + ".ply");
                if (lodRefineSteps > 0) {
                    model.decimateToLod(targetCount);
                    float cumulativeScale = std::pow((float)lodImageScale / 100.0f, (float)level);
                    std::vector<Camera> lodCams;
                    std::vector<Camera> *lodTrainCams = &cams;
                    if (cumulativeScale < 1.0f) {
                        lodCams = cams;
                        for (Camera &cam : lodCams) cam.applyImageScale(cumulativeScale);
                        lodTrainCams = &lodCams;
                    }
                    std::cout << "LOD " << level << "/" << lodLevels << ": " << sourceCount
                              << " -> " << model.means.size(0) << " gaussians, refining "
                              << lodRefineSteps << " steps at image scale "
                              << (cumulativeScale * 100.0f) << "%" << std::endl;

                    for (int refineStep = 1; refineStep <= lodRefineSteps; refineStep++) {
                        Camera &cam = (*lodTrainCams)[camsIter.next()];
                        std::array<float, 3> stepBg = sampleBackground();
                        MTensor gt = cam.getGPUImage(1, stepBg.data());
                        MTensor *lossMask = nullptr;
                        MTensor mask;
                        float lossMaskMean = 1.0f;
                        MTensor *alphaTarget = nullptr;
                        MTensor alpha;
                        if (cam.hasLossMask()) {
                            mask = cam.getGPULossMask(1);
                            lossMask = &mask;
                            lossMaskMean = cam.getLossMaskMean(1);
                        } else if (cam.imageHasAlpha()) {
                            alpha = cam.getGPULossMask(1);
                            alphaTarget = &alpha;
                        }
                        int globalStep = numIters + (level - 1) * lodRefineSteps + refineStep;
                        model.fullIteration(cam, globalStep, gt, lossMask, lossMaskMean,
                                            alphaTarget, matchAlphaWeight, stepBg.data(), ssimWeight,
                                            lpipsLossWeight, 1);
                        model.schedulersStep(refineStep);
                        model.afterTrain(globalStep, refineStep, lodRefineSteps);
                        msplat_commit();
                    }
                    model.save(lodPath.string(), numIters + level * lodRefineSteps);
                } else {
                    model.saveLodPly(lodPath.string(), numIters, targetCount);
                }
            }
        }

        // Evaluation
        bool finalEvalAlreadyRun = evalEvery > 0 && numIters % evalEvery == 0;
        if (!finalEvalAlreadyRun) runEvaluation(numIters, false);

        // Validation
        if (valCam) {
            const float evalBg[3] = {0.0f, 0.0f, 0.0f};
            MTensor rgb = model.render(*valCam, numIters, evalBg);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters), evalBg).cpu();
            quantizeRenderedForEval(rgb_cpu);

            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
