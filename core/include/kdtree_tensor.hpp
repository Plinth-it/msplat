#ifndef KDTREE_TENSOR_H
#define KDTREE_TENSOR_H

#include <algorithm>
#include <array>
#include <nanoflann.hpp>
#include <vector>
#include <cmath>

// nanoflann adapter for a flat float array of 3D points.
// Used to compute per-point initial scales via KNN.
struct PointsTensor {
    const float *data;
    int64_t count;

    PointsTensor(const float *data, int64_t count) : data(data), count(count) {}

    // nanoflann interface
    size_t kdtree_get_point_count() const { return (size_t)count; }
    float kdtree_get_pt(size_t idx, size_t dim) const { return data[idx * 3 + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }

    using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointsTensor>,
        PointsTensor, 3, size_t>;

    static float percentileMedianSize(const float *points, int64_t pointCount, float percentile) {
        if (!points || pointCount <= 0) return 2.0f;

        std::array<float, 3> sizes{};
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> values;
            values.reserve(pointCount);
            for (int64_t i = 0; i < pointCount; ++i) {
                const float value = points[i * 3 + axis];
                if (std::isfinite(value)) values.push_back(value);
            }
            if (values.empty()) return 2.0f;

            std::sort(values.begin(), values.end());
            const size_t n = values.size();
            const size_t lo = static_cast<size_t>(
                (1.0f - percentile) * 0.5f * static_cast<float>(n));
            const size_t hi = std::min(n - 1, static_cast<size_t>(
                (1.0f + percentile) * 0.5f * static_cast<float>(n)));
            sizes[axis] = values[hi] - values[lo];
        }

        std::sort(sizes.begin(), sizes.end());
        return sizes[1];
    }

    // Brush-compatible KNN scale initializer for point-cloud splats.
    std::vector<float> scales() const {
        if (count <= 0) return {};
        if (count < 3) return std::vector<float>(count, 1.0f);

        KdTree index(3, *this, {10});

        std::vector<float> result(count);
        std::array<size_t, 3> indices{};
        std::array<float, 3> dists{};
        const float medianSize = std::max(percentileMedianSize(data, count, 0.75f), 0.01f);
        const float maxScale = medianSize * 0.1f;

        for (int64_t i = 0; i < count; i++) {
            index.knnSearch(&data[i * 3], indices.size(), indices.data(), dists.data());
            const float nearest = std::sqrt(dists[1]);
            const float secondNearest = std::sqrt(dists[2]);
            result[i] = std::clamp((nearest + secondNearest) * 0.25f, 1e-3f, maxScale);
        }
        return result;
    }
};

#endif
