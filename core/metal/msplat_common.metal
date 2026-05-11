#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define RAST_BLOCK_X 8
#define RAST_BLOCK_Y 8
#define RAST_BLOCK_SIZE (RAST_BLOCK_X * RAST_BLOCK_Y)
#define CHANNELS 3
#define MAX_REGISTER_CHANNELS 3

constant float SH_C0 = 0.28209479177387814f;
constant float MIN_QUAT_NORM_SQR = 1e-6f;
constant float MAX_PROJECT_Z = 1e10f;
constant float MAX_COV2D_ENTRY = 1e18f;
constant float INV_255 = 1.0f / 255.0f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

constant uint fc_project_sh_degrees_to_use [[function_constant(0)]];
constant bool fc_project_sh_use_mip_splatting [[function_constant(1)]];
constant bool fc_project_sh_reduce_second_moment [[function_constant(2)]];

static inline uint project_sh_degrees_to_use(const uint runtime_value) {
    return is_function_constant_defined(fc_project_sh_degrees_to_use)
        ? fc_project_sh_degrees_to_use
        : runtime_value;
}

static inline bool project_sh_use_mip_splatting(const uint runtime_value) {
    return is_function_constant_defined(fc_project_sh_use_mip_splatting)
        ? fc_project_sh_use_mip_splatting
        : runtime_value != 0;
}

static inline bool project_sh_reduce_second_moment(const uint runtime_value) {
    return is_function_constant_defined(fc_project_sh_reduce_second_moment)
        ? fc_project_sh_reduce_second_moment
        : runtime_value != 0;
}

static inline float packed_gt_alpha(constant uint *gt_packed, const uint pixel) {
    return float((gt_packed[pixel] >> 24u) & 0xffu) * INV_255;
}

static inline float packed_gt_channel(constant uint *gt_packed, const uint pixel, const uint channel) {
    return float((gt_packed[pixel] >> (channel * 8u)) & 0xffu) * INV_255;
}

static inline float packed_gt_effective(
    constant uint *gt_packed,
    const uint pixel,
    const uint channel,
    constant float *background,
    const uint composite_gt
) {
    float gt = packed_gt_channel(gt_packed, pixel, channel);
    if (composite_gt != 0) {
        gt += (1.0f - packed_gt_alpha(gt_packed, pixel)) * background[channel];
    }
    return gt;
}
static inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

static inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5;
}

static inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // Clamp axis-aligned bounding box to valid range [0, img_size).
    // Returns inclusive min, exclusive max.
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

static inline void get_tile_bbox(
    const float2 pix_center,
    const float2 pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // Convert pixel-space center/radius to tile coordinates and compute AABB.
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius.x / (float)BLOCK_X, pix_radius.y / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// Affine transform: mat (row-major 4x3) applied to point p.
static inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// Full 4x4 row-major transform, returns homogeneous coordinates.
static inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

struct QuaternionWXYZ {
    float w;
    float x;
    float y;
    float z;
};

// Training tensors store quaternions as [w, x, y, z]. Metal float4 fields are
// named x/y/z/w, so unpack once and keep the math in quaternion names.
static inline QuaternionWXYZ normalized_quaternion_wxyz(const float4 quat) {
    float s = rsqrt(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    return QuaternionWXYZ{
        quat.x * s,
        quat.y * s,
        quat.z * s,
        quat.w * s
    };
}

static inline float4 pack_quaternion_wxyz(const QuaternionWXYZ quat) {
    return float4(quat.w, quat.x, quat.y, quat.z);
}

// Normalized quaternion -> 3x3 rotation matrix (column-major for Metal).
static inline float3x3 quat_to_rotmat(const float4 quat) {
    QuaternionWXYZ q = normalized_quaternion_wxyz(quat);

    return float3x3(
        1.f - 2.f * (q.y * q.y + q.z * q.z),
        2.f * (q.x * q.y + q.w * q.z),
        2.f * (q.x * q.z - q.w * q.y),
        2.f * (q.x * q.y - q.w * q.z),
        1.f - 2.f * (q.x * q.x + q.z * q.z),
        2.f * (q.y * q.z + q.w * q.x),
        2.f * (q.x * q.z + q.w * q.y),
        2.f * (q.y * q.z - q.w * q.x),
        1.f - 2.f * (q.x * q.x + q.y * q.y)
    );
}

static inline bool finite_float3(const float3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

// Returns true if point is behind the near plane (should be culled).
static inline bool clip_near_plane(
    const float3 p, 
    constant float *viewmat, 
    thread float3 &p_view, 
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (!isfinite(p_view.x) || !isfinite(p_view.y) || !(p_view.z > thresh && p_view.z <= MAX_PROJECT_Z)) {
        return true;
    }
    return false;
}

static inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// Build 3D covariance matrix from scale + quaternion: cov = R*S*S^T*R^T.
// Stores upper triangle (6 floats) since the matrix is symmetric.
static inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);

    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Thread-local overload: writes cov3d to registers instead of device memory
static inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, thread float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// Project 3D covariance to 2D via EWA splatting.
// Takes pre-computed view-space position; exploits J sparsity (5/9 nonzero).
static inline float3 project_cov3d_ewa(
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    // Clamp view-space position to avoid extreme covariance at FOV edges
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    // T = J * W where J has only 5 nonzero entries.
    // Instead of full 3x3 matmul, compute T rows directly.
    // T_row0 = j00 * M_row0 + j20 * M_row2 (viewmat is row-major)
    // T_row1 = j11 * M_row1 + j21 * M_row2
    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;  // T row 0
    float3 t1 = j11 * mr1 + j21 * mr2;  // T row 1

    // cov2d = T * V * T^T, upper-left 2x2 only (3 values)
    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    float3 raw_cov2d = float3(dot(tv0, t0), dot(tv0, t1), dot(tv1, t1));
    float max_abs = max(max(abs(raw_cov2d.x), abs(raw_cov2d.y)), abs(raw_cov2d.z));
    if (max_abs > MAX_COV2D_ENTRY) {
        raw_cov2d *= MAX_COV2D_ENTRY / max_abs;
    }

    return raw_cov2d + float3(0.3f, 0.0f, 0.3f);
}

// Thread-local overload: reads cov3d from registers
static inline float3 project_cov3d_ewa(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float j00 = fx * rz;
    float j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2;
    float j21 = -fy * p_view.y * rz2;

    float3 mr0 = float3(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1 = float3(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2 = float3(viewmat[8], viewmat[9], viewmat[10]);

    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;

    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];

    float3 tv0 = float3(t0.x*v00 + t0.y*v01 + t0.z*v02,
                         t0.x*v01 + t0.y*v11 + t0.z*v12,
                         t0.x*v02 + t0.y*v12 + t0.z*v22);
    float3 tv1 = float3(t1.x*v00 + t1.y*v01 + t1.z*v02,
                         t1.x*v01 + t1.y*v11 + t1.z*v12,
                         t1.x*v02 + t1.y*v12 + t1.z*v22);

    float3 raw_cov2d = float3(dot(tv0, t0), dot(tv0, t1), dot(tv1, t1));
    float max_abs = max(max(abs(raw_cov2d.x), abs(raw_cov2d.y)), abs(raw_cov2d.z));
    if (max_abs > MAX_COV2D_ENTRY) {
        raw_cov2d *= MAX_COV2D_ENTRY / max_abs;
    }

    return raw_cov2d + float3(0.3f, 0.0f, 0.3f);
}

static inline bool compute_cov2d_bounds(
    const float3 cov2d, 
    thread float3 &conic, 
    thread float &radius
) {
    // Invert 2x2 covariance (upper triangle in cov2d.xyz) to get the conic,
    // and compute the gaussian's screen-space radius from eigenvalues (3-sigma).
    if (!finite_float3(cov2d)) {
        return false;
    }
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (!(det > 0.f) || !isfinite(det))
        return false;
    float inv_det = 1.f / det;

    // Conic = inverse of 2x2 covariance
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = sqrt(max(0.1f, b * b - det));
    // 3-sigma radius from the larger eigenvalue
    radius = ceil(3.f * sqrt(b + disc));
    return true;
}

static inline float mip_opacity_compensation(const float3 cov2d) {
    const float3 cov_orig = cov2d - float3(0.3f, 0.0f, 0.3f);
    const float det_orig = max(cov_orig.x * cov_orig.z - cov_orig.y * cov_orig.y, 0.0f);
    const float det_blurred = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det_blurred <= 0.0f) {
        return 1.0f;
    }
    return sqrt(det_orig / det_blurred);
}

static inline float2 compute_bbox_extent(const float3 conic, const float power_threshold) {
    if (!finite_float3(conic) || !isfinite(power_threshold)) {
        return float2(-1.0f);
    }
    float det = conic.x * conic.z - conic.y * conic.y;
    if (!(det > 0.0f) || !isfinite(det) || power_threshold < 0.0f) {
        return float2(-1.0f);
    }
    float inv_det = 1.0f / det;
    return float2(
        sqrt(2.0f * power_threshold * conic.z * inv_det),
        sqrt(2.0f * power_threshold * conic.x * inv_det)
    );
}

static inline float gaussian_sigma(const float2 pixel_coord, const float3 conic, const float2 xy) {
    float2 delta = pixel_coord - xy;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y)
        + conic.y * delta.x * delta.y;
}

static inline float4 tile_rect(const uint2 tile) {
    float2 rect_min = float2(tile) * float2((float)BLOCK_X, (float)BLOCK_Y);
    float2 rect_max = rect_min + float2((float)BLOCK_X, (float)BLOCK_Y);
    return float4(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

static inline bool will_primitive_contribute(
    const float4 rect,
    const float2 mean,
    const float3 conic,
    const float power_threshold
) {
    bool x_left = mean.x < rect.x;
    bool x_right = mean.x > rect.z;
    bool in_x_range = !(x_left || x_right);

    bool y_above = mean.y < rect.y;
    bool y_below = mean.y > rect.w;
    bool in_y_range = !(y_above || y_below);

    if (in_x_range && in_y_range) {
        return true;
    }

    float2 closest_corner = float2(x_left ? rect.x : rect.z, y_above ? rect.y : rect.w);
    float width = rect.z - rect.x;
    float height = rect.w - rect.y;
    float2 d = float2(x_left ? width : -width, y_above ? height : -height);
    float2 diff = mean - closest_corner;

    float tx = in_y_range ? 0.0f
        : clamp((d.x * conic.x * diff.x + d.x * conic.y * diff.y)
            / (d.x * conic.x * d.x), 0.0f, 1.0f);
    float ty = in_x_range ? 0.0f
        : clamp((d.y * conic.y * diff.x + d.y * conic.z * diff.y)
            / (d.y * conic.z * d.y), 0.0f, 1.0f);
    float2 max_contribution_point = closest_corner + float2(tx, ty) * d;
    return gaussian_sigma(mean, conic, max_contribution_point) <= power_threshold;
}

// Project 3D point to pixel coordinates via projection matrix.
static inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

// Metal pads vector types in arrays (e.g. float3 → 16 bytes). These helpers
// read/write contiguous packed data by indexing into the underlying scalar buffer.

static inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

static inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

static inline void write_packed_int2x(device int* arr, int idx, int x) {
    arr[2*idx] = x;
}

static inline void write_packed_int2y(device int* arr, int idx, int y) {
    arr[2*idx+1] = y;
}

static inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

static inline float2 read_packed_float2(device float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

static inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

static inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

static inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

static inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

static inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

static inline float3 read_packed_float3(device const float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

static inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

static inline float3 read_packed_sorted_float3(
    constant float* packed_float,
    constant half* packed_half,
    int idx,
    uint use_half_sorted_buffers
) {
    if (use_half_sorted_buffers != 0) {
        return float3(packed_half[3*idx], packed_half[3*idx+1], packed_half[3*idx+2]);
    }
    return read_packed_float3(packed_float, idx);
}

static inline float read_packed_sorted_float(
    constant float* packed_float,
    constant half* packed_half,
    int idx,
    uint use_half_sorted_buffers
) {
    return use_half_sorted_buffers != 0 ? float(packed_half[idx]) : packed_float[idx];
}

static inline void write_packed_sorted_float3(
    device float* packed_float,
    device half* packed_half,
    int idx,
    float3 val,
    uint use_half_sorted_buffers
) {
    if (use_half_sorted_buffers != 0) {
        packed_half[3*idx] = half(val.x);
        packed_half[3*idx+1] = half(val.y);
        packed_half[3*idx+2] = half(val.z);
    } else {
        write_packed_float3(packed_float, idx, val);
    }
}

static inline void write_packed_sorted_float(
    device float* packed_float,
    device half* packed_half,
    int idx,
    float val,
    uint use_half_sorted_buffers
) {
    if (use_half_sorted_buffers != 0) {
        packed_half[idx] = half(val);
    } else {
        packed_float[idx] = val;
    }
}

static inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

static inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

static inline float clean_raw_sh_channel(float raw) {
    float color = isfinite(raw) ? raw + 0.5f : 0.0f;
    return clamp(color, -100.0f, 100.0f) - 0.5f;
}

static inline bool valid_quaternion(float4 quat) {
    return isfinite(quat.x) && isfinite(quat.y) && isfinite(quat.z) && isfinite(quat.w)
        && dot(quat, quat) >= MIN_QUAT_NORM_SQR;
}


// Shared SH helpers.
static inline void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *dc_coeffs,
    constant float *rest_coeffs,
    device float *colors
) {
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * dc_coeffs[c];
    }
    if (degree < 1) {
        for (int c = 0; c < CHANNELS; ++c) {
            colors[c] = clean_raw_sh_channel(colors[c]);
        }
        return;
    }

    // viewdir is already normalized by caller (normalize() in project_and_sh_forward_kernel etc.)
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * rest_coeffs[0 * CHANNELS + c] +
                              z * rest_coeffs[1 * CHANNELS + c] -
                              x * rest_coeffs[2 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * rest_coeffs[3 * CHANNELS + c] +
             SH_C2[1] * yz * rest_coeffs[4 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * rest_coeffs[5 * CHANNELS + c] +
             SH_C2[3] * xz * rest_coeffs[6 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * rest_coeffs[7 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * rest_coeffs[8 * CHANNELS + c] +
             SH_C3[1] * xy * z * rest_coeffs[9 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * rest_coeffs[10 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 rest_coeffs[11 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * rest_coeffs[12 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * rest_coeffs[13 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * rest_coeffs[14 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * rest_coeffs[15 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * rest_coeffs[16 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * rest_coeffs[17 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * rest_coeffs[18 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 rest_coeffs[19 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * rest_coeffs[20 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 rest_coeffs[21 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * rest_coeffs[22 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 rest_coeffs[23 * CHANNELS + c]);
    }
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = clean_raw_sh_channel(colors[c]);
    }
}

static inline void sh_coeffs_to_color_vjp(
    const uint degree,
    const float3 viewdir,
    constant float *v_colors,
    device float *v_dc_coeffs,
    device float *v_rest_coeffs
) {
    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        v_dc_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    // viewdir is already normalized by caller
    float x = viewdir.x;
    float y = viewdir.y;
    float z = viewdir.z;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    #pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_rest_coeffs[0 * CHANNELS + c] = v1 * v_colors[c];
        v_rest_coeffs[1 * CHANNELS + c] = v2 * v_colors[c];
        v_rest_coeffs[2 * CHANNELS + c] = v3 * v_colors[c];
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_rest_coeffs[3 * CHANNELS + c] = v4 * v_colors[c];
        v_rest_coeffs[4 * CHANNELS + c] = v5 * v_colors[c];
        v_rest_coeffs[5 * CHANNELS + c] = v6 * v_colors[c];
        v_rest_coeffs[6 * CHANNELS + c] = v7 * v_colors[c];
        v_rest_coeffs[7 * CHANNELS + c] = v8 * v_colors[c];
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_rest_coeffs[8 * CHANNELS + c] = v9 * v_colors[c];
        v_rest_coeffs[9 * CHANNELS + c] = v10 * v_colors[c];
        v_rest_coeffs[10 * CHANNELS + c] = v11 * v_colors[c];
        v_rest_coeffs[11 * CHANNELS + c] = v12 * v_colors[c];
        v_rest_coeffs[12 * CHANNELS + c] = v13 * v_colors[c];
        v_rest_coeffs[13 * CHANNELS + c] = v14 * v_colors[c];
        v_rest_coeffs[14 * CHANNELS + c] = v15 * v_colors[c];
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_rest_coeffs[15 * CHANNELS + c] = v16 * v_colors[c];
        v_rest_coeffs[16 * CHANNELS + c] = v17 * v_colors[c];
        v_rest_coeffs[17 * CHANNELS + c] = v18 * v_colors[c];
        v_rest_coeffs[18 * CHANNELS + c] = v19 * v_colors[c];
        v_rest_coeffs[19 * CHANNELS + c] = v20 * v_colors[c];
        v_rest_coeffs[20 * CHANNELS + c] = v21 * v_colors[c];
        v_rest_coeffs[21 * CHANNELS + c] = v22 * v_colors[c];
        v_rest_coeffs[22 * CHANNELS + c] = v23 * v_colors[c];
        v_rest_coeffs[23 * CHANNELS + c] = v24 * v_colors[c];
    }
}

// Shared SIMD-group reductions.
static inline int warp_reduce_all_max(int val, const int warp_size) {
    return simd_max(val);
}

static inline int warp_reduce_all_or(int val, const int warp_size) {
    return simd_or(val);
}

static inline float3 warpSum3(float3 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    val.z = simd_sum(val.z);
    return val;
}

static inline float2 warpSum2(float2 val, const int warp_size, const uint lane) {
    val.x = simd_sum(val.x);
    val.y = simd_sum(val.y);
    return val;
}

static inline float warpSum(float val, const int warp_size, const uint lane) {
    return simd_sum(val);
}

// Shared projection backward helpers.
static inline float3 project_pix_vjp(
    constant float *mat, const float3 p, const uint2 img_size, const float2 v_xy
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);

    float3 v_ndc = {0.5f * img_size.x * v_xy.x, 0.5f * img_size.y * v_xy.y, 0.0f};
    float4 v_proj = {
        v_ndc.x * rw, v_ndc.y * rw, 0., -(v_ndc.x + v_ndc.y) * rw * rw
    };
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    return {
        mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z,
        mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z,
        mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z
    };
}

// compute vjp from df/d_conic to df/c_cov2d
static inline void cov2d_to_conic_vjp(
    float3 conic, 
    float3 v_conic, 
    device float* v_cov2d // float3
) {
    // conic = inverse cov2d
    // df/d_cov2d = -conic * df/d_conic * conic
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// Thread-local overload
static inline void cov2d_to_conic_vjp(
    float3 conic,
    float3 v_conic,
    thread float* v_cov2d
) {
    float2x2 X = float2x2(conic.x, conic.y, conic.y, conic.z);
    float2x2 G = float2x2(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1. * X * G * X;
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

// output space: 2D covariance, input space: cov3d
static inline void project_cov3d_ewa_vjp(
    constant float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    device float* v_cov3d,
    float3 p_view
) {
    // Apply same fov clipping as forward
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

// Thread-local overload: reads cov3d from registers, writes v_cov3d to registers
static inline void project_cov3d_ewa_vjp(
    thread float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float3 v_cov2d,
    device float* v_mean3d,
    thread float* v_cov3d,
    float3 p_view
) {
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * min(lim_x, max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * min(lim_y, max(-lim_y, p_view.y / p_view.z));

    float rz = 1.f / p_view.z;
    float rz2 = rz * rz;

    float3x3 W = float3x3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );

    float3x3 J = float3x3(
        fx * rz,                0.f,                0.f,
        0.f,                    fy * rz,            0.f,
        -fx * p_view.x * rz2,  -fy * p_view.y * rz2, 0.f
    );
    float3x3 V = float3x3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    float3x3 v_cov = float3x3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );

    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;

    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2;
    float fy_rz2 = fy * rz2;
    float rz3 = rz2 * rz;
    float3 v_t = float3(
        -fx_rz2 * v_J[2][0],
        -fy_rz2 * v_J[2][1],
        -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
            fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]
    );
    v_mean3d[0] += (float)dot(v_t, W[0]);
    v_mean3d[1] += (float)dot(v_t, W[1]);
    v_mean3d[2] += (float)dot(v_t, W[2]);
}

static inline float4 quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R) {
    QuaternionWXYZ q = normalized_quaternion_wxyz(quat);
    QuaternionWXYZ v_quat;
    // v_R is COLUMN MAJOR
    v_quat.w =
        2.f * (
                  q.x * (v_R[1][2] - v_R[2][1]) +
                  q.y * (v_R[2][0] - v_R[0][2]) +
                  q.z * (v_R[0][1] - v_R[1][0])
              );
    v_quat.x =
        2.f *
        (
            -2.f * q.x * (v_R[1][1] + v_R[2][2]) +
            q.y * (v_R[0][1] + v_R[1][0]) +
            q.z * (v_R[0][2] + v_R[2][0]) +
            q.w * (v_R[1][2] - v_R[2][1])
        );
    v_quat.y =
        2.f *
        (
            q.x * (v_R[0][1] + v_R[1][0]) -
            2.f * q.y * (v_R[0][0] + v_R[2][2]) +
            q.z * (v_R[1][2] + v_R[2][1]) +
            q.w * (v_R[2][0] - v_R[0][2])
        );
    v_quat.z =
        2.f *
        (
            q.x * (v_R[0][2] + v_R[2][0]) +
            q.y * (v_R[1][2] + v_R[2][1]) -
            2.f * q.z * (v_R[0][0] + v_R[1][1]) +
            q.w * (v_R[0][1] - v_R[1][0])
        );
    return pack_quaternion_wxyz(v_quat);
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
static inline void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const device float* v_cov3d,
    device float* v_scale, // float3
    device float* v_quat // float4
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

// Thread-local overload: reads v_cov3d from registers
static inline void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const thread float* v_cov3d,
    device float* v_scale,
    device float* v_quat
) {
    float3x3 v_V = float3x3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 v_M = 2.f * v_V * M;
    v_scale[0] = (float)dot(R[0], v_M[0]);
    v_scale[1] = (float)dot(R[1], v_M[1]);
    v_scale[2] = (float)dot(R[2], v_M[2]);

    float3x3 v_R = v_M * S;
    float4 out_v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = out_v_quat.x;
    v_quat[1] = out_v_quat.y;
    v_quat[2] = out_v_quat.z;
    v_quat[3] = out_v_quat.w;
}

// Shared deterministic hash helpers.
static inline uint msplat_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline float msplat_hash_unit(uint seed, uint idx, uint channel) {
    uint h = msplat_hash_u32(seed ^ (idx * 0x9e3779b9u) ^ (channel * 0x85ebca6bu));
    return (((float)(h & 0x00ffffffu)) + 0.5f) * (1.0f / 16777216.0f);
}

static inline float msplat_normal_sample(uint seed, uint idx, uint channel) {
    float u1 = max(msplat_hash_unit(seed, idx, channel * 2u), 1e-7f);
    float u2 = msplat_hash_unit(seed, idx, channel * 2u + 1u);
    return sqrt(-2.0f * log(u1)) * cos(6.283185307179586f * u2);
}
