#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <array>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <vector>
#include <memory>
#include <atomic>
#include <cstdlib>
#include <CLI/CLI.hpp>
#include "model.hpp"
#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "bindings.h"

namespace fs = std::filesystem;
using CliClock = std::chrono::steady_clock;

static double secondsBetween(CliClock::time_point start, CliClock::time_point end) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

static std::string formatDuration(double seconds) {
    seconds = std::max(seconds, 0.0);
    std::ostringstream out;
    if (seconds < 60.0) {
        out << std::fixed << std::setprecision(2) << seconds << " s";
        return out.str();
    }

    const int minutes = static_cast<int>(seconds / 60.0);
    const double remainingSeconds = seconds - static_cast<double>(minutes * 60);
    out << minutes << "m " << std::fixed << std::setprecision(1) << remainingSeconds << "s";
    return out.str();
}

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

static std::string formatBrushIteration(int step, int totalSteps) {
    int digits = totalSteps > 0
        ? static_cast<int>(std::floor(std::log10(static_cast<double>(totalSteps)))) + 1
        : 1;
    std::ostringstream out;
    out << std::setw(std::max(digits, 1)) << std::setfill('0') << step;
    return out.str();
}

static std::string formatProgressLine(size_t step, int totalSteps, size_t completedSteps,
                                      int64_t splatCount, double elapsedSeconds) {
    const double percent = totalSteps > 0
        ? 100.0 * static_cast<double>(std::min<size_t>(step, static_cast<size_t>(totalSteps))) /
              static_cast<double>(totalSteps)
        : 100.0;
    std::ostringstream out;
    out << "Progress: " << std::fixed << std::setprecision(1) << percent
        << "% (" << step << "/" << totalSteps << ")  "
        << splatCount << " gaussians";
    if (elapsedSeconds > 0.0 && completedSteps > 0) {
        const double stepsPerSecond = static_cast<double>(completedSteps) / elapsedSeconds;
        out << "  " << std::setprecision(2) << stepsPerSecond << " it/s";
    }
    return out.str();
}

struct FinalPsnrResult {
    double psnr = 0.0;
    double ssim = 0.0;
    double l1 = 0.0;
    size_t views = 0;
};

struct QualityPresetDefaults {
    int maxResolution;
    float growthGradThreshold;
    float growthSelectFraction;
};

static QualityPresetDefaults qualityPresetDefaults(const std::string &name) {
    if (name == "fast") {
        return {1920, 0.0025f, 0.25f};
    }
    if (name == "brush") {
        return {2560, 0.001f, 0.25f};
    }
    return {2560, 0.001f, 0.25f};
}

static FinalPsnrResult computeTrainPsnr(Model &model, std::vector<Camera> &cams,
                                        int step, const float *evalBg) {
    FinalPsnrResult result;
    if (cams.empty()) return result;

    const int downscale = model.getDownscaleFactor(step);
    for (Camera &cam : cams) {
        cam.ensureImageLoaded();
        MTensor rgb = model.render(cam, step, evalBg);
        msplat_gpu_sync();
        MTensor rgbCpu = rgb.cpu();
        MTensor gtCpu = cam.getGPUImage(downscale, evalBg).cpu();
        quantizeRenderedForEval(rgbCpu);
        result.psnr += psnr(rgbCpu, gtCpu);
        result.ssim += ssim_eval(rgbCpu, gtCpu);
        result.l1 += l1_loss(rgbCpu, gtCpu);
        result.views++;
    }
    if (result.views > 0) {
        result.psnr /= static_cast<double>(result.views);
        result.ssim /= static_cast<double>(result.views);
        result.l1 /= static_cast<double>(result.views);
    }
    return result;
}

static fs::path brushExportPathForName(const std::string &projectRoot, const std::string &exportPath,
                                       const std::string &exportName, int step, int totalSteps) {
    fs::path dir = resolveBrushExportPath(projectRoot, exportPath);
    std::string name = replaceAll(exportName, "{iter}", formatBrushIteration(step, totalSteps));
    return dir / name;
}

static fs::path exportPathForStep(const std::string &projectRoot, const std::string &exportPath,
                                  const std::string &exportName, const std::string &outputScene,
                                  int step, int totalSteps) {
    if (exportPath.empty()) {
        fs::path p(outputScene);
        return p.replace_filename(fs::path(p.stem().string() + "_" + std::to_string(step) + p.extension().string()));
    }

    return brushExportPathForName(projectRoot, exportPath, exportName, step, totalSteps);
}

static std::vector<std::string> splitArgsStr(const std::string &content) {
    std::istringstream in(content);
    std::vector<std::string> args;
    std::string arg;
    while (in >> arg) args.push_back(arg);
    return args;
}

static std::string optionName(std::string token) {
    size_t eq = token.find('=');
    if (eq != std::string::npos) token.resize(eq);
    return token;
}

static const std::unordered_map<std::string, std::string>& optionCanonicalNames() {
    static const std::unordered_map<std::string, std::string> names = {
        {"-o", "output"}, {"--output", "output"},
        {"--seed", "seed"},
        {"-s", "export-every"}, {"--save-every", "export-every"}, {"--export-every", "export-every"},
        {"--export-path", "export-path"}, {"--export-name", "export-name"},
        {"--lod-levels", "lod-levels"}, {"--lod-keep-ratio", "lod-keep"}, {"--lod-decimation-keep", "lod-keep"},
        {"--lod-refine-steps", "lod-refine-steps"}, {"--lod-image-scale", "lod-image-scale"},
        {"--resume", "resume"}, {"--start-iter", "start-iter"},
        {"--val", "val"}, {"--val-image", "val-image"}, {"--val-render", "val-render"},
        {"--eval", "eval"}, {"--test-every", "test-every"}, {"--eval-split-every", "eval-split-every"},
        {"--eval-every", "eval-every"}, {"--eval-save-to-disk", "eval-save-to-disk"},
        {"--final-quality", "final-quality"}, {"--no-final-quality", "final-quality"},
        {"-n", "total-train-iters"}, {"--num-iters", "total-train-iters"}, {"--total-train-iters", "total-train-iters"},
        {"-d", "downscale-factor"}, {"--downscale-factor", "downscale-factor"},
        {"--quality", "quality"},
        {"--max-resolution", "max-resolution"}, {"--max-frames", "max-frames"},
        {"--subsample-frames", "subsample-frames"}, {"--subsample-points", "subsample-points"},
        {"--alpha-mode", "alpha-mode"}, {"--num-downscales", "num-downscales"},
        {"--resolution-schedule", "resolution-schedule"}, {"--sh-degree", "sh-degree"},
        {"--sh-degree-interval", "sh-degree-interval"}, {"--sh-warmup-iters", "sh-warmup-iters"},
        {"--ssim-weight", "ssim-weight"},
        {"--refine-every", "refine-every"}, {"--warmup-length", "warmup-length"},
        {"--reset-alpha-every", "reset-alpha-every"},
        {"--densify-grad-thresh", "growth-grad-threshold"}, {"--growth-grad-threshold", "growth-grad-threshold"},
        {"--densify-size-thresh", "densify-size-thresh"},
        {"--stop-screen-size-at", "stop-screen-size-at"}, {"--growth-stop-iter", "growth-stop-iter"},
        {"--max-splats", "max-splats"}, {"--growth-select-fraction", "growth-select-fraction"},
        {"--split-screen-size", "split-at-screen-size"}, {"--split-at-screen-size", "split-at-screen-size"},
        {"--match-alpha-weight", "match-alpha-weight"}, {"--lpips-loss-weight", "lpips-loss-weight"},
        {"--aux-loss-time", "aux-loss-time"},
        {"--opac-decay", "opac-decay"}, {"--scale-decay", "scale-decay"},
        {"--mean-noise-weight", "mean-noise-weight"}, {"--lr-mean", "lr-mean"},
        {"--lr-mean-end", "lr-mean-end"}, {"--lr-scale", "lr-scale"},
        {"--lr-scale-end", "lr-scale-end"}, {"--lr-rotation", "lr-rotation"},
        {"--lr-coeffs-dc", "lr-coeffs-dc"}, {"--lr-coeffs-sh-scale", "lr-coeffs-sh-scale"},
        {"--lr-opac", "lr-opac"}, {"--random-init-scene-scale", "random-init-scene-scale"},
        {"--reduce-second-moment", "reduce-second-moment"},
        {"--background-noise-strength", "background-noise-strength"},
        {"--keep-crs", "keep-crs"}, {"--normalize-crs", "normalize-crs"},
        {"--render-mip", "render-mip"}, {"--render-mode", "render-mode"},
        {"--bg-color", "background-color"}, {"--background-color", "background-color"},
        {"--colmap-image-path", "colmap-image-path"},
        {"--with-viewer", "with-viewer"},
        {"--rerun-enabled", "rerun-enabled"},
        {"--rerun-log-train-stats-every", "rerun-log-train-stats-every"},
        {"--rerun-log-splats-every", "rerun-log-splats-every"},
        {"--rerun-max-img-size", "rerun-max-img-size"},
    };
    return names;
}

static std::string canonicalOptionKey(const std::string &token) {
    const auto &names = optionCanonicalNames();
    auto it = names.find(optionName(token));
    return it != names.end() ? it->second : "";
}

static int optionValueCount(const std::string &key) {
    static const std::unordered_set<std::string> flags = {
        "val", "eval", "eval-save-to-disk", "final-quality", "reduce-second-moment",
        "keep-crs", "normalize-crs", "render-mip", "rerun-enabled",
    };
    if (key.empty() || flags.count(key) > 0) return 0;
    if (key == "background-color") return 3;
    return 1;
}

static bool qualityPresetControlsKey(const std::string &key) {
    return key == "max-resolution"
        || key == "growth-grad-threshold"
        || key == "growth-select-fraction";
}

static int optionValueCountAfter(const std::vector<std::string> &args, size_t index, const std::string &key) {
    int count = optionValueCount(key);
    if (key == "background-color" && index + 1 < args.size() && args[index + 1].find(',') != std::string::npos) {
        return 1;
    }
    return count;
}

static bool isOptionToken(const std::string &token) {
    return token.size() > 1 && token[0] == '-';
}

static void skipOptionValues(const std::vector<std::string> &args, size_t &index, const std::string &key) {
    if (args[index].find('=') != std::string::npos) return;
    int count = optionValueCountAfter(args, index, key);
    while (count-- > 0 && index + 1 < args.size()) ++index;
}

static std::string findInputPath(int argc, char **argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--") return i + 1 < args.size() ? args[i + 1] : "";
        if (isOptionToken(args[i])) {
            skipOptionValues(args, i, canonicalOptionKey(args[i]));
            continue;
        }
        return args[i];
    }
    return "";
}

static std::unordered_set<std::string> explicitCliOptionKeys(int argc, char **argv) {
    std::unordered_set<std::string> keys;
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--") break;
        if (!isOptionToken(args[i])) continue;
        std::string key = canonicalOptionKey(args[i]);
        if (!key.empty()) keys.insert(key);
        skipOptionValues(args, i, key);
    }
    return keys;
}

static std::vector<std::string> filterArgsFileOptions(const std::vector<std::string> &fileArgs,
                                                       const std::unordered_set<std::string> &explicitKeys) {
    std::vector<std::string> filtered;
    for (size_t i = 0; i < fileArgs.size(); ++i) {
        if (!isOptionToken(fileArgs[i])) {
            filtered.push_back(fileArgs[i]);
            continue;
        }

        std::string key = canonicalOptionKey(fileArgs[i]);
        bool overridden = !key.empty()
            && (explicitKeys.count(key) > 0
                || (explicitKeys.count("quality") > 0 && qualityPresetControlsKey(key)));
        if (!overridden) filtered.push_back(fileArgs[i]);
        if (fileArgs[i].find('=') != std::string::npos) continue;

        int count = optionValueCountAfter(fileArgs, i, key);
        while (count-- > 0 && i + 1 < fileArgs.size()) {
            ++i;
            if (!overridden) filtered.push_back(fileArgs[i]);
        }
    }
    return filtered;
}

static std::vector<std::string> argvWithDatasetArgs(int argc, char **argv) {
    std::vector<std::string> merged(argv, argv + argc);
    std::string inputPath = findInputPath(argc, argv);
    if (inputPath.empty()) return merged;

    fs::path argsPath = fs::path(inputPath) / "args.txt";
    if (!fs::is_regular_file(argsPath)) return merged;

    std::ifstream file(argsPath);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::vector<std::string> fileArgs = splitArgsStr(buffer.str());
    if (fileArgs.empty()) return merged;

    std::vector<std::string> filtered = filterArgsFileOptions(fileArgs, explicitCliOptionKeys(argc, argv));
    if (filtered.empty()) return merged;

    std::vector<std::string> out;
    out.reserve(1 + filtered.size() + (size_t)argc - 1);
    out.push_back(argv[0]);
    out.insert(out.end(), filtered.begin(), filtered.end());
    for (int i = 1; i < argc; ++i) out.push_back(argv[i]);
    std::cerr << "Loaded settings from " << argsPath << std::endl;
    return out;
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
    int64_t start = inputData.pointsFromPlyOverride ? subsampleStep - 1 : 0;
    for (int64_t i = start; i < inputData.points.count; i += subsampleStep) {
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
    auto *outputOption = app.add_option("-o,--output", outputScene, "Output scene path");
    uint32_t seed = 42;
    app.add_option("--seed", seed, "Brush-style random seed");
    int saveEvery = 5000;
    app.add_option("-s,--save-every,--export-every", saveEvery, "Save/export every N steps (-1 to disable)");
    int progressEvery = 0;
    app.add_option("--progress-every", progressEvery, "Print training progress every N steps (0 = every 1%)")
        ->check(CLI::NonNegativeNumber);
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
    int lodDecimationKeep = 50;
    auto *lodDecimationKeepOption = app.add_option("--lod-decimation-keep", lodDecimationKeep,
                                                   "Brush-style LOD keep percentage")
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
    bool finalQualityRequested = false;
    bool skipFinalQualityRequested = false;
    app.add_flag("--final-quality", finalQualityRequested,
                 "Compute final train-quality metrics after saving output");
    app.add_flag("--no-final-quality", skipFinalQualityRequested,
                 "Skip final train-quality metrics after saving output (default)");

    // Training hyperparameters
    int numIters = 30000;
    app.add_option("-n,--num-iters,--total-train-iters", numIters, "Number of iterations")
        ->check(CLI::Range(1, 1000000));
    float downScaleFactor = 1.0f;
    app.add_option("-d,--downscale-factor", downScaleFactor, "Image downscale factor")
        ->check(CLI::Range(1.0f, 32.0f));
    bool logImageLoading = true;
    bool disableImageLoadingLog = false;
    app.add_flag("--log-image-loading", logImageLoading,
                 "Log lazy image decode, resize, and packed target preparation (default on)");
    app.add_flag("--no-log-image-loading", disableImageLoadingLog,
                 "Disable image loading status output");
    int imagePrefetchWorkers = 2;
    auto *imagePrefetchWorkersOption = app.add_option(
        "--image-prefetch-workers", imagePrefetchWorkers,
        "Background CPU image decode workers")
        ->check(CLI::Range(1, 8));
    std::string qualityPreset = "default";
    app.add_option("--quality", qualityPreset, "Quality preset: fast, default, brush")
        ->check(CLI::IsMember({"fast", "default", "brush"}));
    QualityPresetDefaults defaultQuality = qualityPresetDefaults(qualityPreset);
    int maxResolution = defaultQuality.maxResolution;
    auto *maxResolutionOption = app.add_option("--max-resolution", maxResolution, "Max loaded image resolution (0 disables)")
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
    int shWarmupIters = 5000;
    app.add_option("--sh-warmup-iters", shWarmupIters, "Brush-style SH degree warmup period in iterations")
        ->check(CLI::NonNegativeNumber);
    int shDegreeInterval = 0;
    auto *shDegreeIntervalOption = app.add_option("--sh-degree-interval", shDegreeInterval, "Increase SH degree every N steps (0 = full degree immediately)")
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
    float densifyGradThresh = defaultQuality.growthGradThreshold;
    auto *densifyGradThreshOption = app.add_option("--densify-grad-thresh,--growth-grad-threshold", densifyGradThresh, "Gradient threshold for split/dup");
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
    float growthSelectFraction = defaultQuality.growthSelectFraction;
    auto *growthSelectFractionOption = app.add_option("--growth-select-fraction", growthSelectFraction, "Fraction of high-gradient splats selected for growth")
        ->check(CLI::Range(0.0f, 1.0f));
    float splitScreenSize = 0.25f;
    app.add_option("--split-screen-size,--split-at-screen-size", splitScreenSize, "Screen-space split threshold");
    float matchAlphaWeight = 0.1f;
    app.add_option("--match-alpha-weight", matchAlphaWeight, "Alpha L1 loss weight for transparent targets")
        ->check(CLI::Range(0.0f, 100.0f));
    float lpipsLossWeight = 0.0f;
    app.add_option("--lpips-loss-weight", lpipsLossWeight, "LPIPS perceptual loss weight")
        ->check(CLI::Range(0.0f, 100.0f));
    std::string backwardRasterizer = "auto";
    auto *backwardRasterizerOption = app.add_option(
        "--backward-rasterizer",
        backwardRasterizer,
        "Training backward rasterizer: auto, pixel, persplat, or brush")
        ->check(CLI::IsMember({"auto", "pixel", "perpixel", "chunked", "persplat", "brush"}));
    float auxLossTime = 0.8f;
    app.add_option("--aux-loss-time", auxLossTime, "Brush compatibility option; accepted but currently unused")
        ->check(CLI::Range(0.0f, 1.0f));
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
        ->delimiter(',')
        ->expected(3);
    std::string colmapImagePath;
    app.add_option("--colmap-image-path", colmapImagePath, "Override COLMAP image directory");
    bool withViewer = false;
    app.add_option("--with-viewer", withViewer, "Brush viewer compatibility option; accepted but ignored");
    bool rerunEnabled = false;
    app.add_flag("--rerun-enabled", rerunEnabled, "Brush rerun compatibility option; accepted but ignored");
    int rerunLogTrainStatsEvery = 50;
    app.add_option("--rerun-log-train-stats-every", rerunLogTrainStatsEvery,
                   "Brush rerun compatibility option; accepted but ignored")
        ->check(CLI::PositiveNumber);
    int rerunLogSplatsEvery = 0;
    app.add_option("--rerun-log-splats-every", rerunLogSplatsEvery,
                   "Brush rerun compatibility option; accepted but ignored")
        ->check(CLI::NonNegativeNumber);
    int rerunMaxImgSize = 512;
    app.add_option("--rerun-max-img-size", rerunMaxImgSize,
                   "Brush rerun compatibility option; accepted but ignored")
        ->check(CLI::PositiveNumber);

    std::vector<std::string> mergedArgs = argvWithDatasetArgs(argc, argv);
    std::vector<char*> mergedArgv;
    mergedArgv.reserve(mergedArgs.size());
    for (std::string &arg : mergedArgs) mergedArgv.push_back(arg.data());
    try {
        app.parse(static_cast<int>(mergedArgv.size()), mergedArgv.data());
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    QualityPresetDefaults selectedQuality = qualityPresetDefaults(qualityPreset);
    if (maxResolutionOption->count() == 0) maxResolution = selectedQuality.maxResolution;
    if (densifyGradThreshOption->count() == 0) densifyGradThresh = selectedQuality.growthGradThreshold;
    if (growthSelectFractionOption->count() == 0) growthSelectFraction = selectedQuality.growthSelectFraction;
    if (maxResolution == 0 && imagePrefetchWorkersOption->count() == 0) imagePrefetchWorkers = 1;
    if (disableImageLoadingLog) logImageLoading = false;

    if (normalizeCrs) keepCrs = false;
    if (stopScreenSizeAtOption->count() == 0) stopScreenSizeAt = growthStopIter;
    if (backwardRasterizerOption->count() > 0) {
        setenv("MSPLAT_BACKWARD_RASTERIZER", backwardRasterizer.c_str(), 1);
    }
    if (shDegreeIntervalOption->count() == 0) {
        shDegreeInterval = (shWarmupIters > 0 && shDegree > 0) ? shWarmupIters / shDegree : 0;
    }
    if (lodDecimationKeepOption->count() > 0) lodKeepRatio = static_cast<float>(lodDecimationKeep) / 100.0f;
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
        const auto totalStart = CliClock::now();
        const auto datasetStart = CliClock::now();
        InputData inputData = inputDataFromX(projectRoot, colmapImagePath);
        filterCameras(inputData, maxFrames, subsampleFrames);
        subsamplePoints(inputData, subsamplePointStep);
        if (!validate && !inputData.evalCameras.empty()) evalMode = true;

        auto imageLoadCounter = logImageLoading
            ? std::make_shared<std::atomic<size_t>>(0)
            : nullptr;
        const size_t imageLoadTotal = inputData.cameras.size() + inputData.evalCameras.size();
        for (auto &cam : inputData.cameras)
            cam.configureLazyImageLoad(cameraDownscaleFactor(cam, downScaleFactor, maxResolution),
                                       alphaMode, logImageLoading,
                                       imageLoadCounter, imageLoadTotal);
        for (auto &cam : inputData.evalCameras)
            cam.configureLazyImageLoad(cameraDownscaleFactor(cam, downScaleFactor, maxResolution),
                                       alphaMode, logImageLoading,
                                       imageLoadCounter, imageLoadTotal);

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
        const auto datasetEnd = CliClock::now();

        constexpr unsigned brushSceneLoaderSeed = 42;
        CameraPrefetcher camsPrefetcher(cams, brushSceneLoaderSeed,
                                        static_cast<size_t>(imagePrefetchWorkers));

        const auto modelStart = CliClock::now();
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
        const auto modelEnd = CliClock::now();

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

            clearImageLoadingStatusLine();
            std::cout << "\n=== Evaluation (" << nTest << " test views";
            if (evalStep != numIters) std::cout << ", step " << evalStep;
            std::cout << ") ===" << std::endl;

            for (int i = 0; i < nTest; i++) {
                testCams[i].ensureImageLoaded();
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
                    imwriteRGB((imageDir / (fs::path(testCams[i].filePath).filename().string() + ".png")).string(), evalImg);
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
        const size_t firstTrainingStep = step;
        const size_t plannedTrainingSteps = numIters >= static_cast<int>(firstTrainingStep)
            ? static_cast<size_t>(numIters) - firstTrainingStep + 1
            : 0;
        const size_t progressInterval = progressEvery > 0
            ? static_cast<size_t>(progressEvery)
            : static_cast<size_t>(std::max(1, numIters / 100));

        bool benchmarking = std::getenv("BENCHMARK") != nullptr;
        const char *timingModeEnv = std::getenv("MSPLAT_BENCHMARK_TIMING_MODE");
        const std::string benchmarkTimingMode = timingModeEnv != nullptr
            ? std::string(timingModeEnv)
            : std::string("drain-each-iter");
        const bool benchmarkAsyncSubmit = benchmarkTimingMode == "async-submit";
        int bench_warmup = 50;
        std::vector<double> bench_iter_ms, bench_cpu_ms, bench_drain_ms;
        if (benchmarking) {
            bench_iter_ms.reserve(numIters);
            bench_cpu_ms.reserve(numIters);
            bench_drain_ms.reserve(numIters);
        }
        auto cpu_now = []() { return std::chrono::high_resolution_clock::now(); };

        auto bench_start = cpu_now();
        const auto trainingStart = CliClock::now();
        for (; step <= (size_t)numIters; step++) {
            Camera &cam = cams[camsPrefetcher.next()];

            auto iter_start = cpu_now();
            int downscale = model.getDownscaleFactor(step);
            std::array<float, 3> stepBg = sampleBackground();
            MTensor &gtPacked = cam.getGPUPackedImage(downscale);
            bool useLossMask = cam.hasLossMask();
            float lossMaskMean = 1.0f;
            if (useLossMask) {
                lossMaskMean = cam.getLossMaskMean(downscale);
            }
            bool useAlphaLoss = !useLossMask && cam.imageHasAlpha();
            bool compositeGt = cam.hasCompositeAlpha()
                && (stepBg[0] != 0.0f || stepBg[1] != 0.0f || stepBg[2] != 0.0f);
            model.fullIteration(cam, step, gtPacked, useLossMask, lossMaskMean,
                                useAlphaLoss, matchAlphaWeight, stepBg.data(), compositeGt,
                                ssimWeight, lpipsLossWeight);
            model.schedulersStep(step);
            model.afterTrain(step);
            msplat_commit();

            if (benchmarking && step > (size_t)bench_warmup) {
                auto pre_sync = cpu_now();
                if (!benchmarkAsyncSubmit) {
                    msplat_gpu_sync();
                }
                auto iter_end = cpu_now();
                double iter_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count() / 1000.0;
                double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(pre_sync - iter_start).count() / 1000.0;
                double drain_ms = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - pre_sync).count() / 1000.0;
                bench_iter_ms.push_back(iter_ms);
                bench_cpu_ms.push_back(cpu_ms);
                bench_drain_ms.push_back(drain_ms);
            }

            if (step == firstTrainingStep || step == static_cast<size_t>(numIters) ||
                step % progressInterval == 0) {
                const auto progressNow = cpu_now();
                const double elapsedSeconds =
                    std::chrono::duration_cast<std::chrono::duration<double>>(progressNow - bench_start).count();
                const size_t completedSteps = step >= firstTrainingStep
                    ? step - firstTrainingStep + 1
                    : 0;
                clearImageLoadingStatusLine();
                std::cout << formatProgressLine(step, numIters, completedSteps,
                                                model.means.size(0), elapsedSeconds)
                          << std::endl;
            }

            if (saveEvery > 0 && step % saveEvery == 0) {
                fs::path p = exportPathForStep(projectRoot, exportPath, exportName, outputScene, (int)step, numIters);
                if (p.has_parent_path()) fs::create_directories(p.parent_path());
                clearImageLoadingStatusLine();
                model.save(p.string(), step);
            }

            if (evalEvery > 0 && step % (size_t)evalEvery == 0) {
                runEvaluation((int)step, evalSaveToDisk);
            }

            if (!valRender.empty() && step % 10 == 0) {
                valCam->ensureImageLoaded();
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
        const auto trainingEnd = CliClock::now();
        const auto finalizationStart = CliClock::now();

        if (benchmarking && benchmarkAsyncSubmit) {
            msplat_gpu_sync();
        }

        if (benchmarking && !bench_iter_ms.empty()) {
            clearImageLoadingStatusLine();
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
            uint64_t forcedSyncs = msplat_drain_forced_sync_count();
            std::cout << "  forced syncs:  " << forcedSyncs << "\n";

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
                std::cout << "\n  --- Per-stage GPU time (synchronized command buffers) ---\n";
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

        // Brush evaluates the base model at the end of primary training and
        // skips evals during LOD phases.
        bool finalEvalAlreadyRun = evalEvery > 0 && numIters % evalEvery == 0;
        if (!finalEvalAlreadyRun) runEvaluation(numIters, evalSaveToDisk);

        bool outputExplicit = outputOption->count() > 0;
        fs::path baseOutputPath = outputExplicit
            ? fs::path(outputScene)
            : brushExportPathForName(projectRoot, exportPath, exportName, numIters, numIters);
        if (baseOutputPath.has_parent_path()) fs::create_directories(baseOutputPath.parent_path());
        inputData.saveCameras((baseOutputPath.parent_path() / "cameras.json").string(), keepCrs);
        clearImageLoadingStatusLine();
        model.save(baseOutputPath.string(), numIters);
        if (lodLevels > 0) {
            for (int level = 1; level <= lodLevels; level++) {
                int64_t sourceCount = model.means.size(0);
                int64_t targetCount = std::max<int64_t>(1, (int64_t)(sourceCount * lodKeepRatio));
                fs::path lodPath;
                if (outputExplicit) {
                    fs::path dir = baseOutputPath.parent_path();
                    if (dir.empty()) dir = ".";
                    lodPath = dir / (baseOutputPath.stem().string() + "_lod" + std::to_string(level) + ".ply");
                } else {
                    std::string lodExportName = exportName;
                    size_t plyPos = lodExportName.rfind(".ply");
                    if (plyPos != std::string::npos) {
                        lodExportName.replace(plyPos, 4, "_lod" + std::to_string(level) + ".ply");
                    } else {
                        lodExportName += "_lod" + std::to_string(level);
                    }
                    lodPath = brushExportPathForName(projectRoot, exportPath, lodExportName,
                                                     lodRefineSteps, lodRefineSteps);
                }
                std::cout << "LOD " << level << "/" << lodLevels
                          << ": computing PUP sensitivity scores..." << std::endl;
                std::vector<float> pupScores = model.computePupLodScores(cams);
                model.decimateToLod(targetCount, pupScores);
                float cumulativeScale = std::pow((float)lodImageScale / 100.0f, (float)level);
                std::cout << "LOD " << level << "/" << lodLevels << ": " << sourceCount
                          << " -> " << model.means.size(0) << " gaussians";
                if (lodRefineSteps > 0) {
                    std::vector<Camera> lodCams;
                    std::vector<Camera> *lodTrainCams = &cams;
                    if (cumulativeScale < 1.0f) {
                        lodCams = cams;
                        for (Camera &cam : lodCams) cam.applyImageScale(cumulativeScale);
                        lodTrainCams = &lodCams;
                    }
                    CameraPrefetcher lodCamsPrefetcher(*lodTrainCams, brushSceneLoaderSeed,
                                                       static_cast<size_t>(imagePrefetchWorkers));
                    std::cout << ", refining " << lodRefineSteps
                              << " steps at image scale " << (cumulativeScale * 100.0f) << "%";
                    std::cout << std::endl;

                    for (int refineStep = 1; refineStep <= lodRefineSteps; refineStep++) {
                        Camera &cam = (*lodTrainCams)[lodCamsPrefetcher.next()];
                        std::array<float, 3> stepBg = sampleBackground();
                        MTensor &gtPacked = cam.getGPUPackedImage(1);
                        bool useLossMask = cam.hasLossMask();
                        float lossMaskMean = 1.0f;
                        if (useLossMask) {
                            lossMaskMean = cam.getLossMaskMean(1);
                        }
                        bool useAlphaLoss = !useLossMask && cam.imageHasAlpha();
                        bool compositeGt = cam.hasCompositeAlpha()
                            && (stepBg[0] != 0.0f || stepBg[1] != 0.0f || stepBg[2] != 0.0f);
                        int globalStep = numIters + (level - 1) * lodRefineSteps + refineStep;
                        model.fullIteration(cam, globalStep, gtPacked, useLossMask, lossMaskMean,
                                            useAlphaLoss, matchAlphaWeight, stepBg.data(), compositeGt,
                                            ssimWeight, lpipsLossWeight, 1);
                        model.schedulersStep(refineStep);
                        model.afterTrain(globalStep, refineStep, lodRefineSteps);
                        msplat_commit();
                    }
                } else {
                    std::cout << std::endl;
                }
                clearImageLoadingStatusLine();
                model.save(lodPath.string(), numIters + level * lodRefineSteps);
            }
        }

        // Validation
        if (valCam) {
            const float evalBg[3] = {0.0f, 0.0f, 0.0f};
            valCam->ensureImageLoaded();
            MTensor rgb = model.render(*valCam, numIters, evalBg);
            msplat_gpu_sync();
            MTensor rgb_cpu = rgb.cpu();
            MTensor gt_cpu = valCam->getGPUImage(model.getDownscaleFactor(numIters), evalBg).cpu();
            quantizeRenderedForEval(rgb_cpu);

            clearImageLoadingStatusLine();
            std::cout << "\n=== Validation (" << valCam->filePath << ") ===" << std::endl;
            std::cout << "  PSNR:  " << psnr(rgb_cpu, gt_cpu)
                      << "  SSIM:  " << ssim_eval(rgb_cpu, gt_cpu)
                      << "  L1:  " << l1_loss(rgb_cpu, gt_cpu)
                      << "  Gaussians: " << model.means.size(0) << std::endl;
        }

        const auto finalizationEnd = CliClock::now();
        const double datasetSeconds = secondsBetween(datasetStart, datasetEnd);
        const double modelSeconds = secondsBetween(modelStart, modelEnd);
        const double trainingSeconds = secondsBetween(trainingStart, trainingEnd);
        const double finalizationSeconds = secondsBetween(finalizationStart, finalizationEnd);
        const double totalSeconds = secondsBetween(totalStart, finalizationEnd);

        clearImageLoadingStatusLine();
        std::cout << "\n=== Timings ===" << std::endl;
        std::cout << "  dataset/setup:   " << formatDuration(datasetSeconds) << std::endl;
        std::cout << "  model/init:      " << formatDuration(modelSeconds) << std::endl;
        std::cout << "  training loop:   " << formatDuration(trainingSeconds);
        if (plannedTrainingSteps > 0 && trainingSeconds > 0.0) {
            std::cout << "  (" << plannedTrainingSteps << " steps, " << std::fixed << std::setprecision(2)
                      << (static_cast<double>(plannedTrainingSteps) / trainingSeconds)
                      << " it/s)";
        }
        std::cout << std::endl;
        std::cout << "  finalize/export: " << formatDuration(finalizationSeconds) << std::endl;
        std::cout << "  total runtime:   " << formatDuration(totalSeconds) << std::endl;

        if (finalQualityRequested && !skipFinalQualityRequested) {
            const auto finalPsnrStart = CliClock::now();
            const float finalEvalBg[3] = {0.0f, 0.0f, 0.0f};
            FinalPsnrResult trainPsnr = computeTrainPsnr(model, cams, numIters, finalEvalBg);
            const double finalPsnrSeconds = secondsBetween(finalPsnrStart, CliClock::now());
            clearImageLoadingStatusLine();
            if (trainPsnr.views > 0) {
                std::cout << "\n=== Final Quality ===" << std::endl;
                std::cout << "  train PSNR:      " << std::fixed << std::setprecision(2)
                          << trainPsnr.psnr << " dB"
                          << "  (" << trainPsnr.views << " views, "
                          << formatDuration(finalPsnrSeconds)
                          << ", not included in runtime)" << std::endl;
                std::cout << "  train SSIM:      " << std::fixed << std::setprecision(4)
                          << trainPsnr.ssim << std::endl;
                std::cout << "  train L1:        " << std::fixed << std::setprecision(5)
                          << trainPsnr.l1 << std::endl;
            }
        }

        cleanup_msplat_metal();
        msplat_gpu_sync();
    } catch (const std::exception &e) {
        clearImageLoadingStatusLine();
        std::cerr << e.what() << std::endl;
        cleanup_msplat_metal();
        msplat_gpu_sync();
        return 1;
    }
}
