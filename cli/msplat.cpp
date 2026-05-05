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
    int saveEvery = -1;
    app.add_option("-s,--save-every", saveEvery, "Save every N steps (-1 to disable)");
    int lodLevels = 0;
    app.add_option("--lod-levels", lodLevels, "Export N importance-ranked LOD PLY files after training")
        ->check(CLI::Range(0, 16));
    float lodKeepRatio = 0.5f;
    app.add_option("--lod-keep-ratio", lodKeepRatio, "Fraction of splats to keep per LOD level")
        ->check(CLI::Range(0.01f, 1.0f));
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

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    int numDownscales = 0;
    app.add_option("--num-downscales", numDownscales, "Progressive downscale levels");
    int resolutionSchedule = 3000;
    app.add_option("--resolution-schedule", resolutionSchedule, "Double resolution every N steps");
    int shDegree = 3;
    app.add_option("--sh-degree", shDegree, "Max spherical harmonics degree")
        ->check(CLI::Range(0, 4));
    int shDegreeInterval = 1;
    app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps");
    float ssimWeight = 0.2f;
    app.add_option("--ssim-weight", ssimWeight, "SSIM loss weight (0 = L1 only)")
        ->check(CLI::Range(0.0f, 1.0f));
    int refineEvery = 200;
    app.add_option("--refine-every", refineEvery, "Densify/prune every N steps");
    int warmupLength = 0;
    app.add_option("--warmup-length", warmupLength, "Steps before first densification");
    int resetAlphaEvery = 0;
    app.add_option("--reset-alpha-every", resetAlphaEvery, "Reset opacity every N refinements, or 0 to disable");
    float densifyGradThresh = 0.0020f;
    app.add_option("--densify-grad-thresh", densifyGradThresh, "Gradient threshold for split/dup");
    float densifySizeThresh = 0.01f;
    app.add_option("--densify-size-thresh", densifySizeThresh, "Size threshold (dup vs split)");
    int stopScreenSizeAt = 15000;
    app.add_option("--stop-screen-size-at", stopScreenSizeAt, "Stop splitting large gaussians after N steps");
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
    app.add_option("--split-screen-size", splitScreenSize, "Screen-space split threshold");
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
    bool keepCrs = false;
    app.add_flag("--keep-crs", keepCrs, "Retain input coordinate reference system");
    bool renderMip = false;
    app.add_flag("--render-mip", renderMip, "Use MIP splatting opacity compensation during training and rendering");
    std::vector<float> bgColor = {0.0f, 0.0f, 0.0f};
    app.add_option("--bg-color", bgColor, "Background RGB (0-1), default black")
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");

    CLI11_PARSE(app, argc, argv);

    if (validate || !valRender.empty()) validate = true;
    if (!valRender.empty() && !fs::exists(valRender)) fs::create_directories(valRender);
    downScaleFactor = std::max(downScaleFactor, 1.0f);

    try {
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);

        for (auto &cam : inputData.cameras)
            cam.loadImage(downScaleFactor);

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
                     bgColor.data(), renderMip);

        std::vector<size_t> camIndices(cams.size());
        std::iota(camIndices.begin(), camIndices.end(), 0);
        InfiniteRandomIterator<size_t> camsIter(camIndices);
        std::mt19937 bgRng(1337);
        auto sampleBackground = [&]() {
            std::array<float, 3> bg = {bgColor[0], bgColor[1], bgColor[2]};
            if (backgroundNoiseStrength > 0.0f) {
                std::uniform_real_distribution<float> dist(-backgroundNoiseStrength, backgroundNoiseStrength);
                for (float &channel : bg) channel = std::clamp(channel + dist(bgRng), 0.0f, 1.0f);
            }
            return bg;
        };

        size_t step = 1;
        if (!resume.empty()) step = model.loadPly(resume) + 1;

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
                fs::path p(outputScene);
                model.save(p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string())).string(), step);
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
                int64_t targetCount = std::max<int64_t>(1, (int64_t)std::llround(sourceCount * lodKeepRatio));
                fs::path lodPath = dir / (stem + "_lod" + std::to_string(level) + ".ply");
                if (lodRefineSteps > 0) {
                    model.decimateToLod(targetCount);
                    float cumulativeScale = std::pow((float)lodImageScale / 100.0f, (float)level);
                    int lodDownscale = std::max(1, (int)std::lround(1.0f / std::max(cumulativeScale, 0.01f)));
                    std::cout << "LOD " << level << "/" << lodLevels << ": " << sourceCount
                              << " -> " << model.means.size(0) << " gaussians, refining "
                              << lodRefineSteps << " steps at downscale " << lodDownscale << std::endl;

                    for (int refineStep = 1; refineStep <= lodRefineSteps; refineStep++) {
                        Camera &cam = cams[camsIter.next()];
                        std::array<float, 3> stepBg = sampleBackground();
                        MTensor gt = cam.getGPUImage(lodDownscale, stepBg.data());
                        MTensor *lossMask = nullptr;
                        MTensor mask;
                        float lossMaskMean = 1.0f;
                        MTensor *alphaTarget = nullptr;
                        MTensor alpha;
                        if (cam.hasLossMask()) {
                            mask = cam.getGPULossMask(lodDownscale);
                            lossMask = &mask;
                            lossMaskMean = cam.getLossMaskMean(lodDownscale);
                        } else if (cam.imageHasAlpha()) {
                            alpha = cam.getGPULossMask(lodDownscale);
                            alphaTarget = &alpha;
                        }
                        int globalStep = numIters + (level - 1) * lodRefineSteps + refineStep;
                        model.fullIteration(cam, globalStep, gt, lossMask, lossMaskMean,
                                            alphaTarget, matchAlphaWeight, stepBg.data(), ssimWeight,
                                            lpipsLossWeight, lodDownscale);
                        model.schedulersStep(refineStep);
                        msplat_commit();
                    }
                    model.save(lodPath.string(), numIters + level * lodRefineSteps);
                } else {
                    model.saveLodPly(lodPath.string(), numIters, targetCount);
                }
            }
        }

        // Evaluation
        if (evalMode && !testCams.empty()) {
            double sumPsnr = 0, sumSsim = 0, sumL1 = 0;
            int nTest = testCams.size();

            std::cout << "\n=== Evaluation (" << nTest << " test views) ===" << std::endl;
            for (int i = 0; i < nTest; i++) {
                MTensor rgb = model.render(testCams[i], numIters);
                msplat_gpu_sync();
                MTensor rgb_cpu = rgb.cpu();
                MTensor gt_cpu = testCams[i].getGPUImage(model.getDownscaleFactor(numIters), bgColor.data()).cpu();
                quantizeRenderedForEval(rgb_cpu);

                float p = psnr(rgb_cpu, gt_cpu);
                float s = ssim_eval(rgb_cpu, gt_cpu);
                float l = l1_loss(rgb_cpu, gt_cpu);
                sumPsnr += p; sumSsim += s; sumL1 += l;

                std::cout << "  [" << (i+1) << "/" << nTest << "] "
                          << fs::path(testCams[i].filePath).filename().string()
                          << "  PSNR=" << p << "  SSIM=" << s << "  L1=" << l << std::endl;
            }
            std::cout << "\n  PSNR:  " << (sumPsnr / nTest)
                      << "  SSIM:  " << (sumSsim / nTest)
                      << "  L1:  " << (sumL1 / nTest)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        // Validation
        if (valCam) {
            MTensor rgb = model.render(*valCam, numIters);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters), bgColor.data()).cpu();
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
