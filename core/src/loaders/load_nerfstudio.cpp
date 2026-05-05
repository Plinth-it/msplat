#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Try adding common image extensions if file doesn't exist
static std::string resolveImagePath(const std::string &path) {
    if (fs::exists(path)) return path;
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"})
        if (fs::exists(path + ext)) return path + ext;
    return path;
}

static float fovToFocal(float fovRadians, int pixels) {
    return 0.5f * static_cast<float>(pixels) / std::tan(0.5f * fovRadians);
}

static float jsonFloat(const json &doc, const char *key, float fallback) {
    return doc.contains(key) && !doc[key].is_null() ? doc[key].get<float>() : fallback;
}

InputData loaders::loadNerfstudio(const std::string &projectRoot) {
    fs::path root(projectRoot);
    fs::path transformsPath = root / "transforms.json";
    if (!fs::exists(transformsPath)) transformsPath = root / "transforms_train.json";

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
                ? resolveImagePath(imagePath.string())
                : resolveImagePath((baseDir / imagePath).string());
            if (frame.contains("mask_path")) {
                std::string mp = frame["mask_path"].get<std::string>();
                fs::path maskPath(mp);
                cam.maskPath = maskPath.is_absolute()
                    ? resolveImagePath(maskPath.string())
                    : resolveImagePath((baseDir / maskPath).string());
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

            out.push_back(cam);
        }
    };

    appendFrames(j, transformsDir, data.cameras);

    std::sort(data.cameras.begin(), data.cameras.end(),
        [](const Camera &a, const Camera &b) { return a.filePath < b.filePath; });

    fs::path evalPath = root / "transforms_val.json";
    if (!fs::exists(evalPath)) evalPath = root / "transforms_test.json";
    if (fs::exists(evalPath)) {
        std::ifstream evalFile(evalPath.string());
        json evalJson = json::parse(evalFile);
        appendFrames(evalJson, evalPath.parent_path(), data.evalCameras);
        std::sort(data.evalCameras.begin(), data.evalCameras.end(),
            [](const Camera &a, const Camera &b) { return a.filePath < b.filePath; });
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
    loadDatasetPlyOverride(projectRoot, data.points);

    autoScaleAndCenter(data);
    return data;
}
