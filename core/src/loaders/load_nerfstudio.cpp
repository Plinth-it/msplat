#include "loaders.hpp"
#include <nlohmann/json.hpp>
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
        // Global defaults (overridden per-frame if present)
        int gW = doc.value("w", 0), gH = doc.value("h", 0);
        float gFx = doc.value("fl_x", 0.0f), gFy = doc.value("fl_y", 0.0f);
        float gCx = doc.value("cx", 0.0f), gCy = doc.value("cy", 0.0f);
        float gK1 = doc.value("k1", 0.0f), gK2 = doc.value("k2", 0.0f), gK3 = doc.value("k3", 0.0f);
        float gP1 = doc.value("p1", 0.0f), gP2 = doc.value("p2", 0.0f);

        for (auto &frame : doc["frames"]) {
            Camera cam;
            cam.width  = frame.value("w", gW);  cam.height = frame.value("h", gH);
            cam.fx = frame.value("fl_x", gFx);  cam.fy = frame.value("fl_y", gFy);
            cam.cx = frame.value("cx", gCx);     cam.cy = frame.value("cy", gCy);
            cam.k1 = frame.value("k1", gK1);     cam.k2 = frame.value("k2", gK2);
            cam.k3 = frame.value("k3", gK3);
            cam.p1 = frame.value("p1", gP1);     cam.p2 = frame.value("p2", gP2);

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

            // transform_matrix is 4x4 c2w (OpenGL convention)
            auto &tm = frame["transform_matrix"];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    cam.camToWorld[r*4+c] = tm[r][c].get<float>();

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

    autoScaleAndCenter(data);
    return data;
}
