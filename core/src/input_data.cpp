#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <cctype>

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

void Camera::loadImage(float downscaleFactor, AlphaModeOverride alphaMode) {
    Image raw = imreadRGB(filePath);
    if (raw.empty()) return;
    alphaAsMask = false;

    if (maskPath.empty()) maskPath = findMaskPath(filePath, datasetRoot);
    Image rawMask;
    if (!maskPath.empty() && fs::exists(maskPath)) {
        rawMask = imreadRGB(maskPath);
        if (!rawMask.empty() && (rawMask.width != raw.width || rawMask.height != raw.height)) {
            rawMask = resizeArea(rawMask, raw.width, raw.height);
        }
    }

    // If actual image dimensions differ from metadata, rescale intrinsics
    if (width > 0 && height > 0 && (raw.width != width || raw.height != height)) {
        float sx = (float)raw.width / (float)width;
        float sy = (float)raw.height / (float)height;
        fx *= sx; fy *= sy; cx *= sx; cy *= sy;
        width = raw.width; height = raw.height;
    } else if (width == 0 || height == 0) {
        width = raw.width; height = raw.height;
    }

    // Downscale
    if (downscaleFactor > 1.0f) {
        int newW = (int)(width / downscaleFactor);
        int newH = (int)(height / downscaleFactor);
        raw = resizeArea(raw, newW, newH);
        if (!rawMask.empty()) rawMask = resizeArea(rawMask, newW, newH);
        float s = 1.0f / downscaleFactor;
        fx *= s; fy *= s; cx *= s; cy *= s;
        width = newW; height = newH;
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
}

void Camera::applyImageScale(float imageScale) {
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
    mtensorCompositeImageCache.clear();
    mtensorCompositeImageCacheBackground.clear();
    mtensorLossMaskCache.clear();
    lossMaskMeanCache.clear();
    cachedViewMat.reset();
    cachedProjViewMat.reset();
}

Image Camera::getImage(int downscaleFactor) {
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramids.find(downscaleFactor);
    if (it != imagePyramids.end()) return it->second;

    int newW = image.width / downscaleFactor;
    int newH = image.height / downscaleFactor;
    Image scaled = resizeArea(image, newW, newH);
    imagePyramids[downscaleFactor] = scaled;
    return scaled;
}

Image Camera::getMaskImage(int downscaleFactor) {
    if (downscaleFactor <= 1) return maskImage;

    auto it = maskPyramids.find(downscaleFactor);
    if (it != maskPyramids.end()) return it->second;

    int newW = maskImage.width / downscaleFactor;
    int newH = maskImage.height / downscaleFactor;
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
    Image img = getImage(downscaleFactor);
    if (!img.hasAlpha() || background == nullptr || alphaAsMask) return getGPUImage(downscaleFactor);

    std::array<float, 3> bg = {background[0], background[1], background[2]};
    auto cacheIt = mtensorCompositeImageCache.find(downscaleFactor);
    auto bgIt = mtensorCompositeImageCacheBackground.find(downscaleFactor);
    if (cacheIt != mtensorCompositeImageCache.end() && bgIt != mtensorCompositeImageCacheBackground.end() && bgIt->second == bg) {
        return cacheIt->second;
    }

    MTensor mt = gpu_empty({img.height, img.width, 3}, DType::Float32);
    float *dst = mt.data<float>();
    const float *src = img.ptr();
    for (int i = 0; i < img.width * img.height; i++) {
        float a = img.alpha[i];
        dst[i * 3 + 0] = src[i * 3 + 0] + background[0] * (1.0f - a);
        dst[i * 3 + 1] = src[i * 3 + 1] + background[1] * (1.0f - a);
        dst[i * 3 + 2] = src[i * 3 + 2] + background[2] * (1.0f - a);
    }
    mtensorCompositeImageCache[downscaleFactor] = mt;
    mtensorCompositeImageCacheBackground[downscaleFactor] = bg;
    return mtensorCompositeImageCache[downscaleFactor];
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
    getGPULossMask(downscaleFactor);
    return lossMaskMeanCache[downscaleFactor];
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
