#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <utility>
#include <deque>
#include <cstdlib>
#include <stdexcept>
#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include "random_iter.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

static bool iequals(const std::string &a, const std::string &b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
        [](unsigned char ca, unsigned char cb) {
            return std::tolower(ca) == std::tolower(cb);
        });
}

static bool hasPathComponent(const fs::path &path, const std::string &component) {
    return std::any_of(path.begin(), path.end(), [&](const fs::path &part) {
        return iequals(part.string(), component);
    });
}

static std::vector<fs::path> maskSearchDirsForImage(const fs::path &image) {
    fs::path dir = image.parent_path();
    fs::path parent = dir.parent_path();
    std::vector<fs::path> dirs = {
        parent / "masks",
        dir / "masks",
        parent / "mask",
        dir / "mask",
    };

    fs::path suffix;
    for (fs::path cur = dir; !cur.empty(); suffix = cur.filename() / suffix, cur = cur.parent_path()) {
        if (iequals(cur.filename().string(), "images")) {
            dirs.push_back(cur.parent_path() / "masks" / suffix);
            dirs.push_back(cur.parent_path() / "mask" / suffix);
            break;
        }
        if (cur == cur.root_path()) break;
    }

    return dirs;
}

static bool maskNameMatches(const fs::path &candidate,
                            const std::string &imageName,
                            const std::string &imageStem,
                            const std::string &maskStem) {
    std::string candidateStem = candidate.stem().string();
    return iequals(candidateStem, imageName)
        || iequals(candidateStem, imageStem)
        || iequals(candidateStem, maskStem);
}

static bool pathEndsWith(const fs::path &path, const fs::path &suffix) {
    if (suffix.empty() || suffix == ".") return true;

    auto pathIt = path.end();
    auto suffixIt = suffix.end();
    while (suffixIt != suffix.begin()) {
        if (pathIt == path.begin()) return false;
        --pathIt;
        --suffixIt;
        if (!iequals(pathIt->string(), suffixIt->string())) return false;
    }
    return true;
}

static std::string findBrushStyleMaskPath(const fs::path &image,
                                          const fs::path &datasetRoot,
                                          const std::string &imageName,
                                          const std::string &imageStem,
                                          const std::string &maskStem) {
    const fs::path imageDir = image.parent_path();
    if (!datasetRoot.empty()) {
        const fs::path masksRoot = datasetRoot / "masks";
        if (!fs::is_directory(masksRoot)) return "";

        fs::recursive_directory_iterator it(
            masksRoot, fs::directory_options::skip_permission_denied);
        for (const fs::directory_entry &entry : it) {
            if (!entry.is_regular_file()) continue;
            if (!maskNameMatches(entry.path(), imageName, imageStem, maskStem)) continue;

            const fs::path maskSubdir =
                entry.path().parent_path().lexically_relative(masksRoot);
            if (pathEndsWith(imageDir, maskSubdir)) {
                return entry.path().string();
            }
        }
        return "";
    }

    for (fs::path root = imageDir; !root.empty(); root = root.parent_path()) {
        const fs::path masksRoot = root / "masks";
        if (fs::is_directory(masksRoot)) {
            fs::recursive_directory_iterator it(
                masksRoot, fs::directory_options::skip_permission_denied);
            for (const fs::directory_entry &entry : it) {
                if (!entry.is_regular_file()) continue;
                if (!maskNameMatches(entry.path(), imageName, imageStem, maskStem)) continue;

                const fs::path maskSubdir =
                    entry.path().parent_path().lexically_relative(masksRoot);
                if (pathEndsWith(imageDir, maskSubdir)) {
                    return entry.path().string();
                }
            }
        }

        if (root == root.root_path() || root.parent_path() == root) break;
    }
    return "";
}

static std::string findLocalMaskPath(const fs::path &image,
                                     const std::string &imageName,
                                     const std::string &imageStem,
                                     const std::string &maskStem) {
    for (const fs::path &root : maskSearchDirsForImage(image)) {
        if (!fs::is_directory(root)) continue;

        for (const fs::directory_entry &entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;

            if (maskNameMatches(entry.path(), imageName, imageStem, maskStem)) {
                return entry.path().string();
            }
        }
    }
    return "";
}

static std::string findMaskPath(const std::string &imagePath, const std::string &datasetRoot) {
    fs::path image(imagePath);
    std::string imageName = image.filename().string();
    std::string imageStem = image.stem().string();
    std::string maskStem = imageStem + ".mask";

    std::string path = findLocalMaskPath(image, imageName, imageStem, maskStem);
    if (!path.empty()) return path;

    return findBrushStyleMaskPath(image, fs::path(datasetRoot),
                                  imageName, imageStem, maskStem);
}

static float maskPixelValue(const Image &mask, int index) {
    if (mask.hasAlpha()) {
        return std::clamp(mask.alpha[index], 0.0f, 1.0f);
    }
    const float *p = &mask.data[index * 3];
    return std::clamp(p[0], 0.0f, 1.0f);
}

static uint8_t floatToByte(float value) {
    float scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f;
    return static_cast<uint8_t>(scaled + 0.5f);
}

static std::string imageLogName(const std::string &path) {
    std::string name = fs::path(path).filename().string();
    return name.empty() ? path : name;
}

static int64_t elapsedMillis(std::chrono::steady_clock::time_point start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

static int roundedScaledSize(int size, float factor) {
    return std::max(1, (int)std::round((float)size / factor));
}

static std::string imageProgressText(const char *label, size_t ordinal, size_t total) {
    if (ordinal == 0 || total == 0) return "";
    std::ostringstream out;
    double percent = 100.0 * static_cast<double>(std::min(ordinal, total)) /
                     static_cast<double>(total);
    out << label << " " << ordinal << "/" << total << " ("
        << std::fixed << std::setprecision(1) << percent << "%) ";
    return out.str();
}

static std::string imageLoadProgressText(size_t ordinal, size_t total) {
    return imageProgressText("image cache", ordinal, total);
}

static std::string currentImageProgressText(size_t ordinal, size_t total) {
    return imageProgressText("image", ordinal, total);
}

static std::mutex imageLoadingLogMutex;
static bool imageLoadingStatusVisible = false;

static size_t parseTerminalColumns(const char *value) {
    if (value == nullptr || *value == '\0') return 0;
    char *end = nullptr;
    long columns = std::strtol(value, &end, 10);
    if (end == value || columns <= 0) return 0;
    return static_cast<size_t>(columns);
}

static size_t terminalColumnsForStatusLine() {
    const size_t forcedColumns = parseTerminalColumns(std::getenv("MSPLAT_IMAGE_LOADING_COLUMNS"));
    if (forcedColumns > 0) return forcedColumns;

#if !defined(_WIN32)
    if (::isatty(STDERR_FILENO)) {
        struct winsize size {};
        if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0) {
            return static_cast<size_t>(size.ws_col);
        }
        return parseTerminalColumns(std::getenv("COLUMNS"));
    }
#endif
    return 0;
}

static std::string truncateMiddle(const std::string &message, size_t maxLength) {
    if (maxLength == 0 || message.size() <= maxLength) return message;
    if (maxLength <= 3) return message.substr(0, maxLength);

    const size_t available = maxLength - 3;
    const size_t left = (available + 1) / 2;
    const size_t right = available - left;
    return message.substr(0, left) + "..." + message.substr(message.size() - right);
}

static std::string fitStatusLineToTerminal(const std::string &message) {
    const size_t columns = terminalColumnsForStatusLine();
    if (columns == 0) return message;

    const size_t maxLength = columns > 1 ? columns - 1 : columns;
    return truncateMiddle(message, maxLength);
}

static void updateImageLoadingStatusLine(const std::string &message) {
    std::lock_guard<std::mutex> lock(imageLoadingLogMutex);
    std::cerr << '\r' << fitStatusLineToTerminal(message) << "\033[K" << std::flush;
    imageLoadingStatusVisible = true;
}

void clearImageLoadingStatusLine() {
    std::lock_guard<std::mutex> lock(imageLoadingLogMutex);
    if (!imageLoadingStatusVisible) return;
    std::cerr << "\r\033[K" << std::flush;
    imageLoadingStatusVisible = false;
}

static void copyMaskToAlpha(Image &image, const Image &mask) {
    if (image.empty() || mask.empty()) return;
    image.alpha.resize((size_t)image.width * (size_t)image.height);
    for (int i = 0; i < image.width * image.height; i++) {
        image.alpha[i] = maskPixelValue(mask, i);
    }
}

static void unpremultiplyAlpha(Image &image) {
    if (!image.hasAlpha()) return;
    for (int i = 0; i < image.width * image.height; i++) {
        float a = image.alpha[i];
        if (a > 1e-6f) {
            float inv = 1.0f / a;
            image.data[i * 3 + 0] = std::clamp(image.data[i * 3 + 0] * inv, 0.0f, 1.0f);
            image.data[i * 3 + 1] = std::clamp(image.data[i * 3 + 1] * inv, 0.0f, 1.0f);
            image.data[i * 3 + 2] = std::clamp(image.data[i * 3 + 2] * inv, 0.0f, 1.0f);
        }
    }
}

static void premultiplyAlpha(Image &image) {
    if (!image.hasAlpha()) return;
    for (int i = 0; i < image.width * image.height; i++) {
        float a = image.alpha[i];
        image.data[i * 3 + 0] *= a;
        image.data[i * 3 + 1] *= a;
        image.data[i * 3 + 2] *= a;
    }
}

void Camera::configureLazyImageLoad(float downscaleFactor, AlphaModeOverride alphaMode,
                                    bool logLoading,
                                    std::shared_ptr<std::atomic<size_t>> loadCounter,
                                    size_t loadTotal) {
    lazyImageDownscaleFactor = downscaleFactor;
    lazyAlphaMode = alphaMode;
    lazyImageLoadConfigured = true;
    logImageLoading = logLoading;
    imageLoadCounter = std::move(loadCounter);
    imageLoadTotal = loadTotal;
    imageLoadOrdinal = 0;
}

void Camera::ensureImageLoaded() {
    if (!lazyImageLoadConfigured || imageLoadAttempted || !image.empty()) return;
    loadImage(lazyImageDownscaleFactor, lazyAlphaMode);
}

void Camera::loadImage(float downscaleFactor, AlphaModeOverride alphaMode) {
    auto start = std::chrono::steady_clock::now();
    const std::string logName = imageLogName(filePath);
    if (imageLoadCounter && imageLoadTotal > 0) {
        imageLoadOrdinal = imageLoadCounter->fetch_add(1) + 1;
    }
    const std::string progressText = imageLoadProgressText(imageLoadOrdinal, imageLoadTotal);
    if (logImageLoading) {
        std::ostringstream out;
        out << "msplat: " << progressText << "loading image " << logName
            << " (downscale " << downscaleFactor << ")";
        updateImageLoadingStatusLine(out.str());
    }

    imageLoadAttempted = true;
    imagePyramids.clear();
    maskPyramids.clear();
    mtensorImageCache.clear();
    mtensorPackedImageCache.clear();
    mtensorCompositeImageCache.clear();
    mtensorCompositeImageCacheBackground.clear();
    mtensorLossMaskCache.clear();
    lossMaskMeanCache.clear();

    int maxDecodedPixelSize = 0;
    if (downscaleFactor > 1.0f && width > 0 && height > 0) {
        maxDecodedPixelSize = std::max(1, (int)std::ceil(
            (float)std::max(width, height) / downscaleFactor));
    }

    ImageReadResult loaded = imreadRGBWithMaxSize(filePath, maxDecodedPixelSize);
    Image raw = std::move(loaded.image);
    if (raw.empty()) return;
    const int sourceWidth = loaded.sourceWidth > 0 ? loaded.sourceWidth : raw.width;
    const int sourceHeight = loaded.sourceHeight > 0 ? loaded.sourceHeight : raw.height;
    const int decodedWidth = loaded.decodedWidth > 0 ? loaded.decodedWidth : raw.width;
    const int decodedHeight = loaded.decodedHeight > 0 ? loaded.decodedHeight : raw.height;
    alphaAsMask = false;

    if (maskPath.empty()) maskPath = findMaskPath(filePath, datasetRoot);
    Image rawMask;
    if (!maskPath.empty() && fs::exists(maskPath)) {
        rawMask = imreadRGBWithMaxSize(maskPath, maxDecodedPixelSize).image;
        if (!rawMask.empty() && (rawMask.width != raw.width || rawMask.height != raw.height)) {
            rawMask = resizeArea(rawMask, raw.width, raw.height);
        }
    }

    auto scaleIntrinsicsTo = [&](int newW, int newH) {
        if (width <= 0 || height <= 0) {
            width = newW;
            height = newH;
            return;
        }
        float sx = (float)newW / (float)width;
        float sy = (float)newH / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = newW; height = newH;
    };

    // If actual image dimensions differ from metadata, rescale intrinsics.
    if (width > 0 && height > 0 && (sourceWidth != width || sourceHeight != height)) {
        scaleIntrinsicsTo(sourceWidth, sourceHeight);
    } else if (width == 0 || height == 0) {
        width = sourceWidth; height = sourceHeight;
    }

    // Downscale only if ImageIO did not already decode to the requested size.
    if (downscaleFactor > 1.0f) {
        int newW = roundedScaledSize(width, downscaleFactor);
        int newH = roundedScaledSize(height, downscaleFactor);
        if (raw.width != newW || raw.height != newH) {
            raw = resizeArea(raw, newW, newH);
            if (!rawMask.empty()) rawMask = resizeArea(rawMask, newW, newH);
        }
        scaleIntrinsicsTo(newW, newH);
    } else if (raw.width != width || raw.height != height) {
        scaleIntrinsicsTo(raw.width, raw.height);
    }

    // Undistort if needed
    if (hasDistortion()) {
        auto result = undistortImage(raw, fx, fy, cx, cy, k1, k2, p1, p2, k3);
        if (!rawMask.empty()) {
            rawMask = undistortImage(rawMask, fx, fy, cx, cy, k1, k2, p1, p2, k3).image;
        }
        raw = std::move(result.image);
        fx = result.fx; fy = result.fy;
        cx = result.cx; cy = result.cy;
        width = result.width; height = result.height;
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    if (!rawMask.empty() && alphaMode == AlphaModeOverride::Transparent) {
        unpremultiplyAlpha(raw);
        copyMaskToAlpha(raw, rawMask);
        premultiplyAlpha(raw);
        rawMask = {};
    } else if (alphaMode == AlphaModeOverride::Masked
               || (!rawMask.empty() && alphaMode == AlphaModeOverride::Auto)) {
        if (raw.hasAlpha()) {
            unpremultiplyAlpha(raw);
        }
        alphaAsMask = rawMask.empty() && raw.hasAlpha();
        if (!rawMask.empty()) raw.alpha.clear();
    }

    image = std::move(raw);
    maskImage = std::move(rawMask);

    if (logImageLoading) {
        std::ostringstream out;
        out << "msplat: " << progressText << "loaded image " << logName
            << " source " << sourceWidth << "x" << sourceHeight
            << ", decoded " << decodedWidth << "x" << decodedHeight
            << ", final " << width << "x" << height
            << " in " << elapsedMillis(start) << " ms";
        updateImageLoadingStatusLine(out.str());
    }
}

void Camera::applyImageScale(float imageScale) {
    ensureImageLoaded();
    imageScale = std::clamp(imageScale, 0.0f, 1.0f);
    if (imageScale >= 1.0f || image.empty()) return;

    int newW = std::max(1, (int)((float)image.width * imageScale));
    int newH = std::max(1, (int)((float)image.height * imageScale));
    if (newW == image.width && newH == image.height) return;

    float sx = (float)newW / (float)image.width;
    float sy = (float)newH / (float)image.height;
    image = resizeArea(image, newW, newH);
    if (!maskImage.empty()) maskImage = resizeArea(maskImage, newW, newH);

    fx *= sx;
    fy *= sy;
    cx *= sx;
    cy *= sy;
    width = newW;
    height = newH;

    imagePyramids.clear();
    maskPyramids.clear();
    mtensorImageCache.clear();
    mtensorPackedImageCache.clear();
    mtensorCompositeImageCache.clear();
    mtensorCompositeImageCacheBackground.clear();
    mtensorLossMaskCache.clear();
    lossMaskMeanCache.clear();
    cachedViewMat.reset();
    cachedProjViewMat.reset();
}

Image Camera::getImage(int downscaleFactor) {
    ensureImageLoaded();
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = std::max(1, image.width / downscaleFactor);
    int newH = std::max(1, image.height / downscaleFactor);
    Image scaled = resizeArea(image, newW, newH);
    imagePyramids[downscaleFactor] = scaled;
    return scaled;
}

Image Camera::getMaskImage(int downscaleFactor) {
    ensureImageLoaded();
    if (downscaleFactor <= 1) return maskImage;

    auto it = maskPyramids.find(downscaleFactor);
    if (it != maskPyramids.end()) return it->second;

    int newW = std::max(1, maskImage.width / downscaleFactor);
    int newH = std::max(1, maskImage.height / downscaleFactor);
    Image scaled = resizeArea(maskImage, newW, newH);
    maskPyramids[downscaleFactor] = scaled;
    return scaled;
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    auto it = mtensorImageCache.find(downscaleFactor);
    if (it != mtensorImageCache.end()) return it->second;
    Image img = getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    memcpy(mt.data_ptr(), img.ptr(), img.width * img.height * 3 * sizeof(float));
    mtensorImageCache[downscaleFactor] = mt;
    return mtensorImageCache[downscaleFactor];
}

MTensor& Camera::getGPUImage(int downscaleFactor, const float background[3]) {
    if (background == nullptr) return getGPUImage(downscaleFactor);

    std::array<float, 3> bg = {background[0], background[1], background[2]};
    if (bg[0] == 0.0f && bg[1] == 0.0f && bg[2] == 0.0f) {
        return getGPUImage(downscaleFactor);
    }

    Image img = getImage(downscaleFactor);
    Image mask;
    const bool useExplicitMaskAlpha = !maskImage.empty();
    const bool useImageAlpha = img.hasAlpha();
    if (!useImageAlpha && !useExplicitMaskAlpha) return getGPUImage(downscaleFactor);
    if (useExplicitMaskAlpha) mask = getMaskImage(downscaleFactor);

    auto cacheIt = mtensorCompositeImageCache.find(downscaleFactor);
    auto bgIt = mtensorCompositeImageCacheBackground.find(downscaleFactor);
    if (cacheIt != mtensorCompositeImageCache.end() && bgIt != mtensorCompositeImageCacheBackground.end() && bgIt->second == bg) {
        return cacheIt->second;
    }

    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    float *dst = mt.data<float>();
    const float *src = img.ptr();
    for (int i = 0; i < img.width * img.height; i++) {
        float a = useExplicitMaskAlpha ? maskPixelValue(mask, i) : img.alpha[i];
        dst[i * 3 + 0] = src[i * 3 + 0] + background[0] * (1.0f - a);
        dst[i * 3 + 1] = src[i * 3 + 1] + background[1] * (1.0f - a);
        dst[i * 3 + 2] = src[i * 3 + 2] + background[2] * (1.0f - a);
    }
    mtensorCompositeImageCache[downscaleFactor] = mt;
    mtensorCompositeImageCacheBackground[downscaleFactor] = bg;
    return mtensorCompositeImageCache[downscaleFactor];
}

MTensor& Camera::getGPUPackedImage(int downscaleFactor) {
    auto it = mtensorPackedImageCache.find(downscaleFactor);
    if (it != mtensorPackedImageCache.end()) return it->second;

    auto start = std::chrono::steady_clock::now();
    Image img = getImage(downscaleFactor);
    Image mask;
    const bool useExplicitMaskAlpha = !maskImage.empty();
    if (useExplicitMaskAlpha) mask = getMaskImage(downscaleFactor);

    MTensor mt = gpu_empty({img.height, img.width}, DType::UInt32);
    uint32_t *dst = mt.data<uint32_t>();
    const float *src = img.ptr();
    for (int i = 0; i < img.width * img.height; i++) {
        uint32_t r = floatToByte(src[i * 3 + 0]);
        uint32_t g = floatToByte(src[i * 3 + 1]);
        uint32_t b = floatToByte(src[i * 3 + 2]);
        uint32_t a = 255;
        if (useExplicitMaskAlpha) {
            a = floatToByte(maskPixelValue(mask, i));
        } else if (img.hasAlpha()) {
            a = floatToByte(img.alpha[i]);
        }
        dst[i] = r | (g << 8) | (b << 16) | (a << 24);
    }
    mtensorPackedImageCache[downscaleFactor] = mt;
    if (logImageLoading) {
        std::ostringstream out;
        out << "msplat: " << currentImageProgressText(imageLoadOrdinal, imageLoadTotal)
            << "prepared target " << imageLogName(filePath)
            << " " << img.width << "x" << img.height
            << " in " << elapsedMillis(start) << " ms";
        updateImageLoadingStatusLine(out.str());
    }
    return mtensorPackedImageCache[downscaleFactor];
}

MTensor& Camera::getGPULossMask(int downscaleFactor) {
    auto it = mtensorLossMaskCache.find(downscaleFactor);
    if (it != mtensorLossMaskCache.end()) return it->second;

    Image img = !maskImage.empty() ? getMaskImage(downscaleFactor) : getImage(downscaleFactor);
    MTensor mt = gpu_empty({img.height, img.width}, DType::Float32);
    float *dst = mt.data<float>();
    double sum = 0.0;

    if (!maskImage.empty()) {
        for (int i = 0; i < img.width * img.height; i++) {
            float value = maskPixelValue(img, i);
            dst[i] = value;
            sum += value;
        }
    } else if (img.hasAlpha()) {
        for (int i = 0; i < img.width * img.height; i++) {
            float value = std::clamp(img.alpha[i], 0.0f, 1.0f);
            dst[i] = value;
            sum += value;
        }
    } else {
        for (int i = 0; i < img.width * img.height; i++) {
            dst[i] = 1.0f;
        }
        sum = img.width * img.height;
    }

    lossMaskMeanCache[downscaleFactor] = img.width > 0 && img.height > 0
        ? (float)(sum / (double)(img.width * img.height))
        : 1.0f;
    mtensorLossMaskCache[downscaleFactor] = mt;
    return mtensorLossMaskCache[downscaleFactor];
}

float Camera::getLossMaskMean(int downscaleFactor) {
    auto it = lossMaskMeanCache.find(downscaleFactor);
    if (it != lossMaskMeanCache.end()) return it->second;
    if (!hasLossMask()) return 1.0f;

    Image img = !maskImage.empty() ? getMaskImage(downscaleFactor) : getImage(downscaleFactor);
    double sum = 0.0;
    if (!maskImage.empty()) {
        for (int i = 0; i < img.width * img.height; i++) {
            sum += maskPixelValue(img, i);
        }
    } else if (img.hasAlpha()) {
        for (int i = 0; i < img.width * img.height; i++) {
            sum += std::clamp(img.alpha[i], 0.0f, 1.0f);
        }
    } else {
        sum = img.width * img.height;
    }
    lossMaskMeanCache[downscaleFactor] = img.width > 0 && img.height > 0
        ? (float)(sum / (double)(img.width * img.height))
        : 1.0f;
    return lossMaskMeanCache[downscaleFactor];
}

bool Camera::imageHasAlpha() {
    ensureImageLoaded();
    return image.hasAlpha() && !alphaAsMask;
}

bool Camera::hasLossMask() {
    ensureImageLoaded();
    return !maskImage.empty() || alphaAsMask;
}

bool Camera::hasExplicitMask() {
    ensureImageLoaded();
    return !maskImage.empty();
}

bool Camera::hasCompositeAlpha() {
    ensureImageLoaded();
    return !maskImage.empty() || image.hasAlpha();
}

// ── Camera prefetching ──────────────────────────────────────────────────────

static std::vector<size_t> cameraIndexList(size_t count) {
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

struct CameraPrefetcher::Impl {
    std::vector<Camera> &cameras;
    InfiniteRandomIterator<size_t> iterator;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable jobCv;
    bool stop = false;
    size_t inFlight = 0;
    size_t prefetchDepth = 1;
    std::deque<size_t> order;
    std::deque<size_t> jobs;
    std::vector<bool> ready;
    std::vector<bool> queuedOrLoading;

    Impl(std::vector<Camera> &cameras, unsigned seed, size_t workerCount)
        : cameras(cameras), iterator(cameraIndexList(cameras.size()), seed),
          prefetchDepth(std::max<size_t>(1, workerCount)),
          ready(cameras.size(), false),
          queuedOrLoading(cameras.size(), false) {
        if (cameras.empty()) return;

        workers.reserve(prefetchDepth);
        for (size_t i = 0; i < prefetchDepth; i++) {
            workers.emplace_back([this]() { run(); });
        }

        std::lock_guard<std::mutex> lock(mutex);
        fillPrefetchLocked();
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        jobCv.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    void fillPrefetchLocked() {
        while (!stop && order.size() + inFlight < prefetchDepth) {
            size_t index = iterator.next();
            order.push_back(index);
            if (ready[index]) {
                continue;
            }
            if (queuedOrLoading[index]) {
                continue;
            }
            if (cameras[index].imageLoaded()) {
                ready[index] = true;
                continue;
            }
            queuedOrLoading[index] = true;
            jobs.push_back(index);
            jobCv.notify_one();
        }
    }

    void run() {
        while (true) {
            size_t index = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                jobCv.wait(lock, [&]() { return stop || !jobs.empty(); });
                if (stop) return;
                index = jobs.front();
                jobs.pop_front();
                inFlight++;
            }

            cameras[index].ensureImageLoaded();

            {
                std::lock_guard<std::mutex> lock(mutex);
                inFlight--;
                ready[index] = true;
                queuedOrLoading[index] = false;
                fillPrefetchLocked();
            }
            cv.notify_all();
        }
    }

    size_t next() {
        if (cameras.empty()) {
            throw std::runtime_error("Cannot sample cameras from an empty training set");
        }

        size_t index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&]() {
                return stop || (!order.empty() && ready[order.front()]);
            });
            if (stop) {
                throw std::runtime_error("Camera prefetcher stopped");
            }
            index = order.front();
            order.pop_front();
            fillPrefetchLocked();
        }

        return index;
    }
};

CameraPrefetcher::CameraPrefetcher(std::vector<Camera> &cameras, unsigned seed, size_t workerCount)
    : impl(std::make_unique<Impl>(cameras, seed, workerCount)) {}

CameraPrefetcher::~CameraPrefetcher() = default;

size_t CameraPrefetcher::next() {
    return impl->next();
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    auto applyTransform = [&](Camera &cam) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    };

    // Center camera poses. Eval cameras share the train coordinate transform.
    for (auto &cam : data.cameras) {
        applyTransform(cam);
    }
    for (auto &cam : data.evalCameras) {
        applyTransform(cam);
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    auto applyScale = [&](Camera &cam) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    };

    // Apply scale to camera positions.
    for (auto &cam : data.cameras) {
        applyScale(cam);
    }
    for (auto &cam : data.evalCameras) {
        applyScale(cam);
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    if (!evalCameras.empty()) return {cameras, evalCameras};
    if (testEvery < 2) {
        throw std::invalid_argument("test_every must be at least 2");
    }
    if (cameras.empty()) {
        throw std::runtime_error("Cannot split an empty camera set");
    }

    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

static std::vector<fs::path> jsonFilesInDataset(const fs::path &root) {
    std::vector<fs::path> jsonFiles;
    for (const auto &entry : fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied)) {
        if (hasPathComponent(entry.path(), "__MACOSX")) continue;
        if (entry.is_regular_file() && iequals(entry.path().extension().string(), ".json")) {
            jsonFiles.push_back(entry.path());
        }
    }
    std::sort(jsonFiles.begin(), jsonFiles.end());
    return jsonFiles;
}

static bool pathEndsWithText(const fs::path &path, const std::string &suffix) {
    std::string text = path.generic_string();
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!text.empty() && text.front() != '/') text.insert(text.begin(), '/');
    std::string needle = suffix;
    std::transform(needle.begin(), needle.end(), needle.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!needle.empty() && needle.front() != '/') needle.insert(needle.begin(), '/');
    return text.size() >= needle.size()
        && text.compare(text.size() - needle.size(), needle.size(), needle) == 0;
}

static bool hasNerfstudioJson(const fs::path &root) {
    std::vector<fs::path> jsonFiles = jsonFilesInDataset(root);
    if (std::any_of(jsonFiles.begin(), jsonFiles.end(), [](const fs::path &path) {
            return pathEndsWithText(path, "transforms.json")
                || pathEndsWithText(path, "transforms_train.json");
        })) {
        return true;
    }

    if (jsonFiles.size() != 1) return false;

    std::ifstream f(jsonFiles.front());
    if (!f.is_open()) return false;
    json doc = json::parse(f, nullptr, false);
    return !doc.is_discarded() && doc.contains("frames") && doc["frames"].is_array();
}

static bool hasChildFileNamed(const fs::path &dir, const std::string &name) {
    if (fs::exists(dir / name)) return true;
    if (!fs::is_directory(dir)) return false;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && iequals(entry.path().filename().string(), name)) {
            return true;
        }
    }
    return false;
}

static bool hasColmapSparseModel(const fs::path &root) {
    auto hasModel = [](const fs::path &dir) {
        return (hasChildFileNamed(dir, "cameras.bin") && hasChildFileNamed(dir, "images.bin"))
            || (hasChildFileNamed(dir, "cameras.txt") && hasChildFileNamed(dir, "images.txt"));
    };

    for (const fs::path &dir : {root, root / "sparse" / "0", root / "sparse"}) {
        if (hasModel(dir)) return true;
    }
    for (const auto &entry : fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied)) {
        if (!entry.is_regular_file()) continue;
        if (hasPathComponent(entry.path(), "__MACOSX")) continue;
        fs::path dir = entry.path().parent_path();
        const fs::path name = entry.path().filename();
        if ((iequals(name.string(), "cameras.bin") || iequals(name.string(), "cameras.txt")) && hasModel(dir)) {
            return true;
        }
    }
    return false;
}

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    fs::path root(path);

    // Brush probes COLMAP before Nerfstudio, so mixed datasets prefer the SfM model.
    if (hasColmapSparseModel(root))
        return loaders::loadColmap(path, colmapImagePath);

    // Nerfstudio: transforms.json, split transforms_train.json, or a single JSON.
    if (hasNerfstudioJson(root))
        return loaders::loadNerfstudio(path);

    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json"))
        return loaders::loadPolycam(path);

    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin/cameras.txt), Nerfstudio (transforms.json/transforms_train.json/single json), Polycam (keyframes/)");
}
