#include "loaders.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <optional>
#include <cctype>

namespace fs = std::filesystem;

// Quaternion [w,x,y,z] → row-major 3x3 rotation matrix
static void quatToRotMat(const double q[4], float R[9]) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    w /= n; x /= n; y /= n; z /= n;

    R[0] = (float)(1 - 2*(y*y + z*z));  R[1] = (float)(2*(x*y - w*z));      R[2] = (float)(2*(x*z + w*y));
    R[3] = (float)(2*(x*y + w*z));      R[4] = (float)(1 - 2*(x*x + z*z));  R[5] = (float)(2*(y*z - w*x));
    R[6] = (float)(2*(x*z - w*y));      R[7] = (float)(2*(y*z + w*x));      R[8] = (float)(1 - 2*(x*x + y*y));
}

enum ColmapModel {
    SIMPLE_PINHOLE = 0,
    PINHOLE = 1,
    SIMPLE_RADIAL = 2,
    RADIAL = 3,
    OPENCV = 4,
    OPENCV_FISHEYE = 5,
    FULL_OPENCV = 6,
    FOV = 7,
    SIMPLE_RADIAL_FISHEYE = 8,
    RADIAL_FISHEYE = 9,
    THIN_PRISM_FISHEYE = 10,
};

struct ColmapSparseModel {
    fs::path dir;
    bool binary = false;
};

struct ColmapCamera {
    uint32_t id;
    int model;
    int width, height;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
};

struct ColmapImage {
    uint32_t camId;
    double quat[4]; // w, x, y, z
    double t[3];    // world-to-camera translation
    std::string filename;
};

static bool hasBinaryModel(const fs::path &dir) {
    return fs::exists(dir / "cameras.bin") && fs::exists(dir / "images.bin");
}

static bool hasTextModel(const fs::path &dir) {
    return fs::exists(dir / "cameras.txt") && fs::exists(dir / "images.txt");
}

static std::optional<ColmapSparseModel> findSparseModel(const fs::path &root) {
    for (const fs::path &dir : {root, root / "sparse" / "0", root / "sparse"}) {
        if (hasBinaryModel(dir)) return ColmapSparseModel{dir, true};
        if (hasTextModel(dir)) return ColmapSparseModel{dir, false};
    }
    return std::nullopt;
}

static bool hasPathComponent(const fs::path &path, const std::string &component) {
    return std::any_of(path.begin(), path.end(), [&](const fs::path &part) {
        const std::string text = part.string();
        return text.size() == component.size() && std::equal(text.begin(), text.end(), component.begin(),
            [](unsigned char a, unsigned char b) {
                return std::tolower(a) == std::tolower(b);
            });
    });
}

static bool pathEndsWith(const fs::path &path, const fs::path &suffix) {
    auto pathIt = path.end();
    auto suffixIt = suffix.end();
    while (suffixIt != suffix.begin()) {
        if (pathIt == path.begin()) return false;
        --pathIt;
        --suffixIt;
        if (*pathIt != *suffixIt) return false;
    }
    return true;
}

static std::string findColmapImagePath(const fs::path &root, const fs::path &imageDir,
                                       const std::string &filename) {
    fs::path namePath(filename);
    if (namePath.is_absolute() && fs::exists(namePath)) return namePath.string();

    fs::path direct = imageDir / namePath;
    if (fs::exists(direct)) return direct.string();

    for (const auto &entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        if (hasPathComponent(entry.path(), "masks")) continue;

        fs::path relative = fs::relative(entry.path(), root);
        if (pathEndsWith(relative, namePath)) return entry.path().string();
    }

    return direct.string();
}

static size_t colmapModelParamCount(int model) {
    switch (model) {
        case SIMPLE_PINHOLE: return 3;
        case PINHOLE: return 4;
        case SIMPLE_RADIAL: return 4;
        case RADIAL: return 5;
        case OPENCV: return 8;
        case OPENCV_FISHEYE: return 8;
        case FULL_OPENCV: return 12;
        case FOV: return 5;
        case SIMPLE_RADIAL_FISHEYE: return 4;
        case RADIAL_FISHEYE: return 5;
        case THIN_PRISM_FISHEYE: return 12;
        default: throw std::runtime_error("Unsupported COLMAP camera model: " + std::to_string(model));
    }
}

static void applyCameraParams(ColmapCamera &c, const std::vector<double> &params) {
    if (params.size() != colmapModelParamCount(c.model)) {
        throw std::runtime_error("Invalid COLMAP camera parameter count");
    }

    switch (c.model) {
        case SIMPLE_PINHOLE:
            c.fx = c.fy = (float)params[0];
            c.cx = (float)params[1];
            c.cy = (float)params[2];
            break;
        case PINHOLE:
            c.fx = (float)params[0];
            c.fy = (float)params[1];
            c.cx = (float)params[2];
            c.cy = (float)params[3];
            break;
        case SIMPLE_RADIAL:
        case SIMPLE_RADIAL_FISHEYE:
            c.fx = c.fy = (float)params[0];
            c.cx = (float)params[1];
            c.cy = (float)params[2];
            c.k1 = c.model == SIMPLE_RADIAL ? (float)params[3] : 0.0f;
            break;
        case RADIAL:
        case RADIAL_FISHEYE:
            c.fx = c.fy = (float)params[0];
            c.cx = (float)params[1];
            c.cy = (float)params[2];
            c.k1 = c.model == RADIAL ? (float)params[3] : 0.0f;
            c.k2 = c.model == RADIAL ? (float)params[4] : 0.0f;
            break;
        case OPENCV:
        case OPENCV_FISHEYE:
        case FULL_OPENCV:
        case FOV:
        case THIN_PRISM_FISHEYE:
            c.fx = (float)params[0];
            c.fy = (float)params[1];
            c.cx = (float)params[2];
            c.cy = (float)params[3];
            if (c.model == OPENCV) {
                c.k1 = (float)params[4];
                c.k2 = (float)params[5];
                c.p1 = (float)params[6];
                c.p2 = (float)params[7];
            }
            break;
        default:
            throw std::runtime_error("Unsupported COLMAP camera model: " + std::to_string(c.model));
    }
}

static std::unordered_map<uint32_t, ColmapCamera> readCamerasBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::unordered_map<uint32_t, ColmapCamera> cams;
    for (uint64_t i = 0; i < n; i++) {
        ColmapCamera c = {};
        uint32_t model;
        uint64_t w, h;
        f.read(reinterpret_cast<char*>(&c.id), 4);
        f.read(reinterpret_cast<char*>(&model), 4);
        f.read(reinterpret_cast<char*>(&w), 8);
        f.read(reinterpret_cast<char*>(&h), 8);
        c.model = (int)model;
        c.width = (int)w;
        c.height = (int)h;

        auto rd = [&]() -> double { double v; f.read(reinterpret_cast<char*>(&v), 8); return v; };
        std::vector<double> params;
        params.reserve(colmapModelParamCount(c.model));
        for (size_t p = 0; p < colmapModelParamCount(c.model); p++) {
            params.push_back(rd());
        }
        applyCameraParams(c, params);
        cams[c.id] = c;
    }
    return cams;
}

static int colmapModelFromName(const std::string &name) {
    if (name == "SIMPLE_PINHOLE") return SIMPLE_PINHOLE;
    if (name == "PINHOLE") return PINHOLE;
    if (name == "SIMPLE_RADIAL") return SIMPLE_RADIAL;
    if (name == "RADIAL") return RADIAL;
    if (name == "OPENCV") return OPENCV;
    if (name == "OPENCV_FISHEYE") return OPENCV_FISHEYE;
    if (name == "FULL_OPENCV") return FULL_OPENCV;
    if (name == "FOV") return FOV;
    if (name == "SIMPLE_RADIAL_FISHEYE") return SIMPLE_RADIAL_FISHEYE;
    if (name == "RADIAL_FISHEYE") return RADIAL_FISHEYE;
    if (name == "THIN_PRISM_FISHEYE") return THIN_PRISM_FISHEYE;
    throw std::runtime_error("Unsupported COLMAP camera model: " + name);
}

static std::unordered_map<uint32_t, ColmapCamera> readCamerasTxt(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open cameras.txt: " + path);

    std::unordered_map<uint32_t, ColmapCamera> cams;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string model;
        ColmapCamera c = {};
        iss >> c.id >> model >> c.width >> c.height;
        if (!iss) throw std::runtime_error("Invalid COLMAP camera line: " + line);
        c.model = colmapModelFromName(model);

        std::vector<double> params;
        double value = 0.0;
        while (iss >> value) params.push_back(value);
        applyCameraParams(c, params);
        cams[c.id] = c;
    }
    return cams;
}

static std::vector<ColmapImage> readImagesBin(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);

    std::vector<ColmapImage> images;
    images.reserve(n);

    for (uint64_t i = 0; i < n; i++) {
        ColmapImage img;
        uint32_t imageId;
        f.read(reinterpret_cast<char*>(&imageId), 4);
        f.read(reinterpret_cast<char*>(img.quat), 32); // 4 doubles
        f.read(reinterpret_cast<char*>(img.t), 24);     // 3 doubles
        f.read(reinterpret_cast<char*>(&img.camId), 4);

        char ch;
        while (f.read(&ch, 1) && ch != '\0') img.filename += ch;

        uint64_t numPts2D;
        f.read(reinterpret_cast<char*>(&numPts2D), 8);
        f.seekg(numPts2D * 24, std::ios::cur);

        images.push_back(img);
    }
    return images;
}

static std::vector<ColmapImage> readImagesTxt(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open images.txt: " + path);

    std::vector<ColmapImage> images;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) parts.push_back(part);

        if (parts.size() == 10) {
            ColmapImage img = {};
            img.quat[0] = std::stod(parts[1]);
            img.quat[1] = std::stod(parts[2]);
            img.quat[2] = std::stod(parts[3]);
            img.quat[3] = std::stod(parts[4]);
            img.t[0] = std::stod(parts[5]);
            img.t[1] = std::stod(parts[6]);
            img.t[2] = std::stod(parts[7]);
            img.camId = (uint32_t)std::stoul(parts[8]);
            img.filename = parts[9];
            images.push_back(img);
        } else if (parts.size() % 3 != 0) {
            throw std::runtime_error("Invalid COLMAP image line: " + line);
        }
    }
    return images;
}

static Points readColmapPointsTxt(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open points3D.txt: " + path);

    Points pts;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::vector<std::string> parts;
        std::string part;
        while (iss >> part) parts.push_back(part);
        if (parts.size() < 8) {
            throw std::runtime_error("Invalid COLMAP points3D line: " + line);
        }

        pts.xyz.push_back((float)std::stod(parts[1]));
        pts.xyz.push_back((float)std::stod(parts[2]));
        pts.xyz.push_back((float)std::stod(parts[3]));
        pts.rgb.push_back((uint8_t)std::stoul(parts[4]));
        pts.rgb.push_back((uint8_t)std::stoul(parts[5]));
        pts.rgb.push_back((uint8_t)std::stoul(parts[6]));
    }
    pts.count = (int64_t)(pts.xyz.size() / 3);
    return pts;
}

// w2c rotation + translation → 4x4 c2w row-major with OpenGL Y/Z flip
static void w2cToCamToWorld(const double quat[4], const double t[3], float out[16]) {
    float R[9];
    quatToRotMat(quat, R);

    // R^T (transpose = inverse for rotation)
    float Ri[9] = { R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8] };

    // -R^T * t
    float Ti[3] = {
        -(Ri[0]*(float)t[0] + Ri[1]*(float)t[1] + Ri[2]*(float)t[2]),
        -(Ri[3]*(float)t[0] + Ri[4]*(float)t[1] + Ri[5]*(float)t[2]),
        -(Ri[6]*(float)t[0] + Ri[7]*(float)t[1] + Ri[8]*(float)t[2])
    };

    // OpenGL flip: negate columns 1,2 (camera Y-down→Y-up, Z-fwd→Z-back)
    out[0]  = Ri[0]; out[1]  = -Ri[1]; out[2]  = -Ri[2]; out[3]  = Ti[0];
    out[4]  = Ri[3]; out[5]  = -Ri[4]; out[6]  = -Ri[5]; out[7]  = Ti[1];
    out[8]  = Ri[6]; out[9]  = -Ri[7]; out[10] = -Ri[8]; out[11] = Ti[2];
    out[12] = 0;     out[13] = 0;      out[14] = 0;      out[15] = 1;
}

InputData loaders::loadColmap(const std::string &projectRoot, const std::string &imageSourcePath) {
    fs::path root(projectRoot);
    auto model = findSparseModel(root);
    if (!model) throw std::runtime_error("COLMAP model not found in: " + projectRoot);

    std::string imageDir = !imageSourcePath.empty() ? imageSourcePath
        : fs::exists(root / "images") ? (root / "images").string()
        : projectRoot;

    auto cameras = model->binary
        ? readCamerasBin((model->dir / "cameras.bin").string())
        : readCamerasTxt((model->dir / "cameras.txt").string());
    auto images = model->binary
        ? readImagesBin((model->dir / "images.bin").string())
        : readImagesTxt((model->dir / "images.txt").string());

    std::sort(images.begin(), images.end(),
        [](const ColmapImage &a, const ColmapImage &b) { return a.filename < b.filename; });

    InputData data;
    data.cameras.reserve(images.size());

    for (auto &img : images) {
        auto it = cameras.find(img.camId);
        if (it == cameras.end()) continue;
        auto &cc = it->second;

        Camera cam;
        cam.width = cc.width; cam.height = cc.height;
        cam.fx = cc.fx; cam.fy = cc.fy; cam.cx = cc.cx; cam.cy = cc.cy;
        cam.k1 = cc.k1; cam.k2 = cc.k2; cam.p1 = cc.p1; cam.p2 = cc.p2;
        cam.filePath = findColmapImagePath(root, imageDir, img.filename);
        if (!fs::exists(cam.filePath)) continue;
        w2cToCamToWorld(img.quat, img.t, cam.camToWorld);
        data.cameras.push_back(cam);
    }

    // Point cloud
    fs::path pointsBin = model->dir / "points3D.bin";
    fs::path pointsTxt = model->dir / "points3D.txt";
    fs::path pointsPly = model->dir / "points3D.ply";
    if (fs::exists(pointsBin))
        data.points = readColmapPoints(pointsBin.string());
    else if (fs::exists(pointsTxt))
        data.points = readColmapPointsTxt(pointsTxt.string());
    else if (fs::exists(pointsPly))
        data.points = readPly(pointsPly.string());
    loadDatasetPlyOverride(projectRoot, data.points);

    autoScaleAndCenter(data);
    return data;
}
