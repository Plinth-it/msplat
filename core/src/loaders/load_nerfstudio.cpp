#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace fs = std::filesystem;
using json = nlohmann::json;

static float fovToFocal(float fovRadians, int pixels) {
    return 0.5f * static_cast<float>(pixels) / std::tan(0.5f * fovRadians);
}

static float jsonFloat(const json &doc, const char *key, float fallback) {
    return doc.contains(key) && !doc[key].is_null() ? doc[key].get<float>() : fallback;
}

static bool isFiniteCamera(const Camera &cam) {
    if (cam.width <= 0 || cam.height <= 0) return false;
    if (!std::isfinite(cam.fx) || !std::isfinite(cam.fy)
        || !std::isfinite(cam.cx) || !std::isfinite(cam.cy)) {
        return false;
    }
    if (cam.fx <= 0.0f || cam.fy <= 0.0f) return false;
    for (float value : cam.camToWorld) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

static bool iequals(const std::string &a, const std::string &b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
        [](unsigned char a, unsigned char b) {
            return std::tolower(a) == std::tolower(b);
        });
}

static std::string lowercasePathText(const fs::path &path) {
    std::string text = path.generic_string();
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

static std::vector<fs::path> filesInDataset(const fs::path &root) {
    std::vector<fs::path> files;
    for (const auto &entry : fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied)) {
        if (entry.is_regular_file()) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<fs::path> pathWithImageExtensions(const fs::path &path) {
    if (path.has_extension()) return {path};
    return {path, path.string() + ".png", path.string() + ".jpg", path.string() + ".jpeg"};
}

struct DatasetPathIndex {
    fs::path root;
    std::unordered_map<std::string, fs::path> filesByRelativePath;
};

static DatasetPathIndex buildDatasetPathIndex(const fs::path &root,
                                              const std::vector<fs::path> &datasetFiles) {
    DatasetPathIndex index;
    index.root = fs::absolute(root).lexically_normal();
    for (const fs::path &file : datasetFiles) {
        fs::path relative = fs::absolute(file).lexically_normal().lexically_relative(index.root);
        index.filesByRelativePath.emplace(lowercasePathText(relative), file);
    }
    return index;
}

static fs::path resolveDatasetPath(const DatasetPathIndex &datasetIndex,
                                   const fs::path &path) {
    for (const fs::path &candidate : pathWithImageExtensions(path)) {
        if (fs::exists(candidate)) return candidate;
    }

    for (const fs::path &candidate : pathWithImageExtensions(path)) {
        const fs::path absoluteCandidate = fs::absolute(candidate).lexically_normal();
        fs::path relative = absoluteCandidate.lexically_relative(datasetIndex.root);
        auto firstComponent = relative.begin();
        if (relative.empty()
            || (firstComponent != relative.end() && firstComponent->string() == "..")) {
            continue;
        }

        const std::string wanted = lowercasePathText(relative);
        auto found = datasetIndex.filesByRelativePath.find(wanted);
        if (found != datasetIndex.filesByRelativePath.end()) return found->second;
    }

    return path;
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

static std::vector<fs::path> jsonFilesInDataset(const std::vector<fs::path> &datasetFiles) {
    std::vector<fs::path> jsonFiles;
    for (const fs::path &path : datasetFiles) {
        if (iequals(path.extension().string(), ".json")) jsonFiles.push_back(path);
    }
    std::sort(jsonFiles.begin(), jsonFiles.end());
    return jsonFiles;
}

static fs::path findTransformsJson(const std::vector<fs::path> &jsonFiles) {
    if (jsonFiles.size() == 1) return jsonFiles.front();

    auto transforms = std::find_if(jsonFiles.begin(), jsonFiles.end(), [](const fs::path &path) {
        return pathEndsWithText(path, "transforms.json");
    });
    if (transforms != jsonFiles.end()) return *transforms;

    auto train = std::find_if(jsonFiles.begin(), jsonFiles.end(), [](const fs::path &path) {
        return pathEndsWithText(path, "transforms_train.json");
    });
    if (train != jsonFiles.end()) return *train;

    return fs::path();
}

static fs::path findEvalTransformsJson(const std::vector<fs::path> &jsonFiles) {
    auto val = std::find_if(jsonFiles.begin(), jsonFiles.end(), [](const fs::path &path) {
        return pathEndsWithText(path, "transforms_val.json");
    });
    if (val != jsonFiles.end()) return *val;

    auto test = std::find_if(jsonFiles.begin(), jsonFiles.end(), [](const fs::path &path) {
        return pathEndsWithText(path, "transforms_test.json");
    });
    return test != jsonFiles.end() ? *test : fs::path();
}

InputData loaders::loadNerfstudio(const std::string &projectRoot) {
    fs::path root(projectRoot);
    std::vector<fs::path> datasetFiles = filesInDataset(root);
    DatasetPathIndex datasetIndex = buildDatasetPathIndex(root, datasetFiles);
    std::vector<fs::path> jsonFiles = jsonFilesInDataset(datasetFiles);
    fs::path transformsPath = findTransformsJson(jsonFiles);

    std::ifstream f(transformsPath.string());
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open Nerfstudio transforms file in: " + projectRoot);
    }
    json j = json::parse(f);
    fs::path transformsDir = transformsPath.parent_path();

    InputData data;
    auto appendFrames = [&](const json &doc, const fs::path &baseDir, std::vector<Camera> &out) {
        // Global defaults, overridden per frame when present.
        int gW = doc.value("w", 0), gH = doc.value("h", 0);
        float gFx = jsonFloat(doc, "fl_x", 0.0f), gFy = jsonFloat(doc, "fl_y", 0.0f);
        float gAngleX = jsonFloat(doc, "camera_angle_x", 0.0f);
        float gAngleY = jsonFloat(doc, "camera_angle_y", 0.0f);
        float gK1 = jsonFloat(doc, "k1", 0.0f), gK2 = jsonFloat(doc, "k2", 0.0f);
        float gK3 = jsonFloat(doc, "k3", 0.0f);
        float gP1 = jsonFloat(doc, "p1", 0.0f), gP2 = jsonFloat(doc, "p2", 0.0f);

        for (auto &frame : doc["frames"]) {
            Camera cam;
            cam.width  = frame.value("w", gW);  cam.height = frame.value("h", gH);
            cam.fx = jsonFloat(frame, "fl_x", gFx);  cam.fy = jsonFloat(frame, "fl_y", gFy);
            float angleX = jsonFloat(frame, "camera_angle_x", gAngleX);
            float angleY = jsonFloat(frame, "camera_angle_y", gAngleY);
            cam.k1 = jsonFloat(frame, "k1", gK1);     cam.k2 = jsonFloat(frame, "k2", gK2);
            cam.k3 = jsonFloat(frame, "k3", gK3);
            cam.p1 = jsonFloat(frame, "p1", gP1);     cam.p2 = jsonFloat(frame, "p2", gP2);

            std::string fp = frame["file_path"].get<std::string>();
            fs::path imagePath(fp);
            cam.filePath = imagePath.is_absolute()
                ? resolveDatasetPath(datasetIndex, imagePath).string()
                : resolveDatasetPath(datasetIndex, baseDir / imagePath).string();
            cam.datasetRoot = projectRoot;
            if (!fs::exists(cam.filePath)) continue;
            if (frame.contains("mask_path")) {
                std::string mp = frame["mask_path"].get<std::string>();
                fs::path maskPath(mp);
                cam.maskPath = maskPath.is_absolute()
                    ? resolveDatasetPath(datasetIndex, maskPath).string()
                    : resolveDatasetPath(datasetIndex, baseDir / maskPath).string();
            }

            if (cam.width <= 0 || cam.height <= 0) {
                Image image = imreadRGB(cam.filePath);
                cam.width = image.width;
                cam.height = image.height;
            }
            if (cam.fx <= 0.0f && angleX > 0.0f && cam.width > 0) {
                cam.fx = fovToFocal(angleX, cam.width);
            }
            if (cam.fy <= 0.0f && angleY > 0.0f && cam.height > 0) {
                cam.fy = fovToFocal(angleY, cam.height);
            }
            if (cam.fx <= 0.0f && cam.fy > 0.0f) cam.fx = cam.fy;
            if (cam.fy <= 0.0f && cam.fx > 0.0f) cam.fy = cam.fx;
            if (cam.fx <= 0.0f || cam.fy <= 0.0f) {
                throw std::runtime_error("Nerfstudio frame missing focal length: " + cam.filePath);
            }

            cam.cx = jsonFloat(frame, "cx", jsonFloat(doc, "cx", cam.width * 0.5f));
            cam.cy = jsonFloat(frame, "cy", jsonFloat(doc, "cy", cam.height * 0.5f));

            // transform_matrix is 4x4 c2w; flip camera Y/Z into msplat's pose convention.
            auto &tm = frame["transform_matrix"];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    cam.camToWorld[r*4+c] = tm[r][c].get<float>();
            for (int r = 0; r < 4; r++) {
                cam.camToWorld[r*4+1] *= -1.0f;
                cam.camToWorld[r*4+2] *= -1.0f;
            }
            if (!isFiniteCamera(cam)) continue;

            out.push_back(cam);
        }
    };

    appendFrames(j, transformsDir, data.cameras);

    fs::path evalPath = findEvalTransformsJson(jsonFiles);
    if (fs::exists(evalPath)) {
        std::ifstream evalFile(evalPath.string());
        json evalJson = json::parse(evalFile);
        appendFrames(evalJson, evalPath.parent_path(), data.evalCameras);
    }

    // Point cloud
    if (j.contains("ply_file_path")) {
        std::string p = j["ply_file_path"].get<std::string>();
        if (p[0] != '/') p = (fs::path(projectRoot) / p).string();
        if (fs::exists(p)) data.points = readPly(p);
    }
    if (data.points.count == 0) {
        for (auto p : {"sparse/0/points3D.ply", "points3D.ply"}) {
            auto path = (fs::path(projectRoot) / p).string();
            if (fs::exists(path)) { data.points = readPly(path); break; }
        }
    }
    loadDatasetPlyOverride(projectRoot, data.points, &data.initialGaussianPlyPath);

    autoScaleAndCenter(data);
    return data;
}
