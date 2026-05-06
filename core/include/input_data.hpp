#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <array>
#include <memory>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;   // width * height * 3 floats, RGB, [0,1]
    std::vector<float> alpha;  // optional width * height floats, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    bool hasAlpha() const { return alpha.size() == (size_t)width * (size_t)height; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

enum class AlphaModeOverride {
    Auto,
    Masked,
    Transparent
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;
    std::string datasetRoot;

    Image image;
    Image maskImage;
    std::string maskPath;
    bool alphaAsMask = false;
    std::unordered_map<int, Image> imagePyramids;
    std::unordered_map<int, Image> maskPyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
    std::unordered_map<int, MTensor> mtensorCompositeImageCache;
    std::unordered_map<int, std::array<float, 3>> mtensorCompositeImageCacheBackground;
    std::unordered_map<int, MTensor> mtensorLossMaskCache;
    std::unordered_map<int, float> lossMaskMeanCache;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;
    float lazyImageDownscaleFactor = 1.0f;
    AlphaModeOverride lazyAlphaMode = AlphaModeOverride::Auto;
    bool lazyImageLoadConfigured = false;
    bool imageLoadAttempted = false;

    void loadImage(float downscaleFactor, AlphaModeOverride alphaMode = AlphaModeOverride::Auto);
    void configureLazyImageLoad(float downscaleFactor, AlphaModeOverride alphaMode = AlphaModeOverride::Auto);
    void ensureImageLoaded();
    bool imageLoaded() const { return !image.empty(); }
    void applyImageScale(float imageScale);
    Image getImage(int downscaleFactor);
    Image getMaskImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor);
    MTensor& getGPUImage(int downscaleFactor, const float background[3]);
    MTensor& getGPULossMask(int downscaleFactor);
    float getLossMaskMean(int downscaleFactor);
    bool imageHasAlpha();
    bool hasLossMask();
    bool hasExplicitMask();
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
};

class CameraPrefetcher {
public:
    CameraPrefetcher(std::vector<Camera> &cameras, unsigned seed = 42);
    ~CameraPrefetcher();

    CameraPrefetcher(const CameraPrefetcher&) = delete;
    CameraPrefetcher& operator=(const CameraPrefetcher&) = delete;

    size_t next();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    std::vector<Camera> evalCameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;
    std::string initialGaussianPlyPath;
    int initialGaussianSubsampleStep = 1;
    bool pointsFromPlyOverride = false;

    std::tuple<std::vector<Camera>, Camera*> getCameras(bool validate, const std::string &valImage = "random");
    std::tuple<std::vector<Camera>, std::vector<Camera>> splitTrainTest(int testEvery);
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");

#endif
