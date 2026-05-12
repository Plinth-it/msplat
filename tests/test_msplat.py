"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os
import json
import gc
import sys
import struct
import zlib
import binascii
import subprocess
from pathlib import Path

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)
NATIVE_CLI = Path(__file__).resolve().parents[1] / "build" / "msplat"


def _read_metal_sources(repo_root):
    metal_dir = repo_root / "core" / "metal"
    sources = [
        "msplat_common.metal",
        "msplat_project.metal",
        "msplat_raster_backward.metal",
        "msplat_project_backward.metal",
        "msplat_project_sh.metal",
        "msplat_sort.metal",
        "msplat_loss.metal",
        "msplat_chunked_raster.metal",
        "msplat_densify.metal",
    ]
    return "\n".join((metal_dir / name).read_text(encoding="utf-8") for name in sources)


@pytest.fixture(autouse=True)
def _release_msplat_gpu_cache_after_test():
    yield
    gc.collect()
    msplat = sys.modules.get("msplat")
    if msplat is None:
        return
    msplat.sync()
    msplat._cleanup_raw()


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_rgba_png(path, width, height, pixels):
    raw = bytearray()
    row_bytes = width * 4
    for y in range(height):
        raw.append(0)  # filter type 0
        start = y * row_bytes
        raw.extend(pixels[start:start + row_bytes])

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)


def _write_minimal_nerfstudio_dataset(root, alpha, mask=False, points=True, point_rows=None):
    image_path = os.path.join(root, "image.png")
    pixels = bytes([
        255, 0, 0, 255,
        0, 0, 255, 128 if alpha else 255,
    ])
    _write_rgba_png(image_path, 2, 1, pixels)
    if mask:
        mask_path = os.path.join(root, "mask.png")
        _write_rgba_png(mask_path, 2, 1, bytes([
            255, 255, 255, 255,
            0, 0, 0, 255,
        ]))

    if points:
        if point_rows is None:
            point_rows = [("0", "0", "0", "255", "255", "255")]
        with open(os.path.join(root, "points3D.ply"), "w", encoding="utf-8") as f:
            f.write(
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {len(point_rows)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            for row in point_rows:
                f.write("{} {} {} {} {} {}\n".format(*row))

    transforms = {
        "w": 2,
        "h": 1,
        "fl_x": 2.0,
        "fl_y": 2.0,
        "cx": 1.0,
        "cy": 0.5,
        "frames": [{
            "file_path": "image.png",
            **({"mask_path": "mask.png"} if mask else {}),
            "transform_matrix": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        }],
    }
    with open(os.path.join(root, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(transforms, f)


def _write_binary_float64_point_ply(path, rows):
    with open(path, "wb") as f:
        f.write(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(rows)}\n"
                "property double x\n"
                "property double y\n"
                "property double z\n"
                "property double red\n"
                "property double green\n"
                "property double blue\n"
                "end_header\n"
            ).encode("ascii")
        )
        for row in rows:
            f.write(struct.pack("<6d", *row))


def _write_minimal_colmap_text_dataset(root, mask_filename=None, point_rows=None, image_filename="image.png"):
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "sparse"))

    _write_rgba_png(os.path.join(root, "images", image_filename), 2, 1, bytes([
        255, 0, 0, 255,
        0, 0, 255, 255,
    ]))

    if mask_filename is not None:
        os.makedirs(os.path.join(root, "masks"))
        _write_rgba_png(os.path.join(root, "masks", mask_filename), 2, 1, bytes([
            255, 255, 255, 255,
            0, 0, 0, 255,
        ]))

    with open(os.path.join(root, "sparse", "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 2 1 2.0 2.0 1.0 0.5\n")

    with open(os.path.join(root, "sparse", "images.txt"), "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write(f"1 1 0 0 0 0 0 0 1 {image_filename}\n")
        f.write("\n")

    with open(os.path.join(root, "sparse", "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        if point_rows is None:
            point_rows = [("1", "0", "0", "0", "255", "255", "255")]
        for row in point_rows:
            f.write("{} {} {} {} {} {} {} 0 1 0\n".format(*row))


def _ply_vertex_count(path):
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise AssertionError("PLY vertex count not found")


def _ply_comments(path):
    comments = []
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii").strip()
            if line.startswith("comment "):
                comments.append(line)
            if line == "end_header":
                break
    return comments


def _read_ply_vertices(path):
    properties = []
    vertex_count = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.startswith("property float "):
                properties.append(line.split()[-1])
            if line == "end_header":
                break
        row_size = 4 * len(properties)
        return [
            dict(zip(properties, struct.unpack("<" + "f" * len(properties), f.read(row_size))))
            for _ in range(vertex_count)
        ]


def _read_first_ply_vertex(path):
    return _read_ply_vertices(path)[0]


def _write_gaussian_ply(path, rows):
    properties = [
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "opacity",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment Exported from Brush\n")
        f.write(b"comment SplatRenderMode: default\n")
        f.write(f"element vertex {len(rows)}\n".encode("ascii"))
        for name in properties:
            f.write(f"property float {name}\n".encode("ascii"))
        f.write(b"end_header\n")
        for row in rows:
            f.write(struct.pack("<" + "f" * len(properties), *(row[name] for name in properties)))


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import():
    import msplat
    assert hasattr(msplat, "GaussianTrainer")
    assert hasattr(msplat, "TrainingConfig")
    assert hasattr(msplat, "Dataset")
    assert hasattr(msplat, "load_dataset")


def test_native_cli_exposes_quality_presets():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    result = subprocess.run(
        [str(NATIVE_CLI), "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "--quality" in result.stdout
    assert "fast, default, brush" in result.stdout
    assert "--image-prefetch-workers" in result.stdout
    assert "--no-log-image-loading" in result.stdout
    assert "--final-quality" in result.stdout
    assert "--no-final-quality" in result.stdout
    assert "--backward-rasterizer" in result.stdout


def test_metal_uses_dynamic_global_intersections():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)
    tensor_header = (repo_root / "core" / "include" / "metal_tensor.hpp").read_text(encoding="utf-8")

    assert "map_gaussian_to_intersects_kernel_cpso" in host_source
    assert "radix_sort_histogram_kernel_cpso" in host_source
    assert "get_tile_bin_edges_kernel_cpso" in host_source
    assert "should_use_dynamic_intersections" in host_source
    assert "MSPLAT_INTERSECTION_SORT" in host_source
    assert "needs_fixed_tile_bins" in host_source
    assert "dynamic_capacity_valid" in host_source
    assert "did_dynamic_count_prepass" in host_source
    assert "padded_dynamic_intersection_capacity" in host_source
    assert "DType::UInt64" in host_source
    assert "IntersectionKeyBitsMode::Auto32WhenSafe" in host_source
    assert "IntersectionKeyBitsMode::Force64" in host_source
    assert '!mode || std::strcmp(mode, "auto") == 0' in host_source
    assert 'std::strcmp(mode, "auto") == 0' in host_source
    assert 'std::strcmp(mode, "64") == 0' in host_source
    assert "use_dynamic_u32_keys ? 32u : 64u" in host_source
    assert "UInt64" in tensor_header
    assert "case DType::UInt64:  return 8;" in tensor_header
    assert "device uint64_t* isect_ids" in shader_source
    assert "constant uint64_t* isect_ids_sorted" in shader_source
    assert "device const uint64_t* keys_in" in shader_source
    assert "device uint64_t* keys_out" in shader_source
    assert "(tile_id << 32) | depth_bits" in shader_source
    assert "write_packed_sorted_float(" in shader_source


def test_tile_bin_edges_close_last_tile_transition():
    repo_root = Path(__file__).resolve().parents[1]
    shader_source = _read_metal_sources(repo_root)

    def kernel_body(name):
        body = shader_source.split(f"kernel void {name}", 1)[1]
        stops = [pos for token in ("\nkernel void ", "\ninline ")
                 if (pos := body.find(token)) >= 0]
        return body[:min(stops)] if stops else body

    for name in ("get_tile_bin_edges_kernel", "get_tile_bin_edges_u32_kernel"):
        body = kernel_body(name)
        transition = body.index("prev_tile_idx != cur_tile_idx")
        final_end = body.index("idx == num_intersects - 1")

        assert "idx == 0 || idx == num_intersects - 1" not in body
        assert "write_packed_int2x(tile_bins, cur_tile_idx" in body
        assert "write_packed_int2y(tile_bins, prev_tile_idx, idx)" in body
        assert "write_packed_int2y(tile_bins, cur_tile_idx, num_intersects)" in body
        assert transition < final_end


def test_metal_exposes_brush_style_persplat_backward():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)

    assert "return std::max(img_width, img_height) > 2560 || num_tiles > 25000;" in host_source
    assert "rasterize_backward_persplat_kernel_cpso" in host_source
    assert 'loadWithEmptyConstants(@"rasterize_backward_persplat_kernel")' in host_source
    assert "MSPLAT_BACKWARD_RASTERIZER" in host_source
    assert "MSPLAT_BACKWARD_DEBUG" in host_source
    assert "pixel atomic estimate" in host_source
    assert "pixel_warp_atomic_groups" in host_source
    assert "copy_int_buffer_kernel_cpso" in host_source
    assert "--backward-rasterizer" in (repo_root / "cli" / "msplat.cpp").read_text(encoding="utf-8")
    assert "rasterize_backward_persplat_kernel" in shader_source
    assert "copy_int_buffer_kernel" in shader_source
    assert "SPLAT_BATCH" in shader_source
    assert "pix_state" in shader_source
    assert "diagonal" in shader_source.lower()
    assert "max_useful_isect" in shader_source


def test_persplat_backward_scales_loss_gradient_before_half_pack():
    repo_root = Path(__file__).resolve().parents[1]
    shader_source = _read_metal_sources(repo_root)
    start = shader_source.index("kernel void rasterize_backward_persplat_kernel")
    end = shader_source.index("kernel void nd_rasterize_backward_kernel", start)
    body = shader_source[start:end]

    assert "threadgroup half4 pix_v_out_tail" in body
    assert "pix_v_out_tail_scale" in body
    assert "inv_pix_v_out_tail_scale" in body
    assert "float4(pix_v_out_tail[pix_rank]) * inv_pix_v_out_tail_scale" in body


def test_project_sh_kernels_use_function_constant_specialization():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)

    assert "fc_project_sh_degrees_to_use [[function_constant(0)]]" in shader_source
    assert "fc_project_sh_use_mip_splatting [[function_constant(1)]]" in shader_source
    assert "fc_project_sh_reduce_second_moment [[function_constant(2)]]" in shader_source
    assert "project_sh_degrees_to_use(degrees_to_use)" in shader_source
    assert "project_sh_reduce_second_moment(adam_hp.reduce_second_moment)" in shader_source

    assert "MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION" in host_source
    assert "loadWithEmptyConstants(@\"project_and_sh_forward_kernel\")" in host_source
    assert "loadWithEmptyConstants(@\"project_and_sh_backward_kernel\")" in host_source
    assert "project_sh_forward_pipeline(" in host_source
    assert "project_sh_backward_pipeline(" in host_source
    assert "newFunctionWithName:function_name" in host_source


def test_loss_kernels_have_opt_in_function_constant_specialization():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "fc_loss_composite_gt [[function_constant(3)]]" in shader_source
    assert "fc_loss_use_mask [[function_constant(4)]]" in shader_source
    assert "fc_loss_use_alpha_loss [[function_constant(5)]]" in shader_source
    assert "loss_composite_gt(composite_gt)" in shader_source
    assert "loss_use_mask(use_loss_mask)" in shader_source
    assert "loss_use_alpha_loss(use_alpha_loss)" in shader_source

    assert "MSPLAT_ENABLE_LOSS_SPECIALIZATION" in host_source
    assert "loss_l1_specializations" in host_source
    assert "loss_ssim_h_specializations" in host_source
    assert "loss_ssim_fused_specializations" in host_source
    assert "loss_ssim_v_bwd_specializations" in host_source
    assert "loadWithEmptyConstants(@\"l1_loss_fwd_bwd_kernel\")" in host_source
    assert "loadWithEmptyConstants(@\"ssim_h_fwd_kernel\")" in host_source
    assert '@"l1_loss_fwd_bwd_kernel"' in host_source
    assert '@"ssim_fused_v_fwd_h_bwd_kernel"' in host_source
    assert "MSPLAT_ENABLE_LOSS_SPECIALIZATION=1" in readme


def test_raster_backward_has_opt_in_function_constant_specialization():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "fc_raster_use_alpha_loss [[function_constant(6)]]" in shader_source
    assert "fc_raster_use_half_sorted_buffers [[function_constant(7)]]" in shader_source
    assert "fc_raster_use_warp_merge [[function_constant(8)]]" in shader_source
    assert "raster_use_alpha_loss(use_alpha_loss)" in shader_source
    assert "raster_use_half_sorted_buffers(use_half_sorted_buffers)" in shader_source
    assert "raster_use_warp_merge()" in shader_source

    assert "MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION" in host_source
    assert "MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE" in host_source
    assert "raster_backward_specializations" in host_source
    assert "raster_backward_persplat_specializations" in host_source
    assert "raster_backward_chunked_specializations" in host_source
    assert "loadWithEmptyConstants(@\"rasterize_backward_kernel\")" in host_source
    assert "raster_backward_pipeline(" in host_source
    assert "MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION=1" in readme
    assert "MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE=1" in readme


def test_raster_backward_has_no_duplicate_refine_calculation():
    repo_root = Path(__file__).resolve().parents[1]
    shader_source = (repo_root / "core" / "metal" / "msplat_raster_backward.metal").read_text(encoding="utf-8")

    refine_work = "v_refine_local = length(v_xy_local * float2((float)img_size.x, (float)img_size.y)) / final_alpha;"
    assert shader_source.count(refine_work) == 1


def test_quaternion_rotation_uses_named_wxyz_layout():
    repo_root = Path(__file__).resolve().parents[1]
    shader_source = _read_metal_sources(repo_root)

    assert "struct QuaternionWXYZ" in shader_source
    assert "normalized_quaternion_wxyz" in shader_source
    assert "Training tensors store quaternions as [w, x, y, z]" in shader_source
    assert "pack_quaternion_wxyz(v_quat)" in shader_source

    start = shader_source.index("static inline float4 quat_to_rotmat_vjp")
    end = shader_source.index("// given cotangent", start)
    body = shader_source[start:end]

    assert "QuaternionWXYZ q = normalized_quaternion_wxyz(quat)" in body
    assert "QuaternionWXYZ v_quat" in body
    assert "v_quat.w" in body
    assert "v_quat.x" in body
    assert "v_quat.y" in body
    assert "v_quat.z" in body
    assert "float w = quat.x" not in body
    assert "w element stored in x field" not in body


def test_metal_supports_opt_in_half_sorted_buffers():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader_source = _read_metal_sources(repo_root)
    tensor_header = (repo_root / "core" / "include" / "metal_tensor.hpp").read_text(encoding="utf-8")

    assert "Float16" in tensor_header
    assert "case DType::Float16: return 2;" in tensor_header
    assert "MSPLAT_HALF_SORTED_BUFFERS" in host_source
    assert "packed_sorted_half" in host_source
    assert "bind_sorted_half_buffers" in host_source
    assert "use_half_sorted_buffers_u32" in host_source
    assert "DType sorted_dtype = use_half_sorted_buffers ? DType::Float16 : DType::Float32" in host_source

    assert "read_packed_sorted_float3" in shader_source
    assert "write_packed_sorted_float3" in shader_source
    assert "constant half* packed_conic_half" in shader_source
    assert "device half* packed_conic_half" in shader_source
    assert "constant uint& use_half_sorted_buffers" in shader_source


def test_metal_sources_compile_as_logical_units():
    repo_root = Path(__file__).resolve().parents[1]
    cmake_source = (repo_root / "CMakeLists.txt").read_text(encoding="utf-8")
    manifest = (repo_root / "core" / "metal" / "msplat_metal.metal").read_text(encoding="utf-8")

    assert "METAL_SOURCES" in cmake_source
    assert "METAL_COMMON_SOURCE" in cmake_source
    assert "foreach(METAL_SOURCE ${METAL_SOURCES})" in cmake_source
    assert "metallib" in cmake_source
    assert "msplat_project.metal" in cmake_source
    assert "msplat_loss.metal" in cmake_source
    assert "msplat_densify.metal" in cmake_source
    assert "MSPLAT_ENABLE_METAL4_SHADERS" in cmake_source
    assert "MSPLAT_METAL_FLAGS" in cmake_source
    assert "-std=metal4.0" in cmake_source
    assert "msplat_metal.metal" not in cmake_source

    assert "Metal source manifest" in manifest
    assert '#include "msplat_common.metal"' in manifest


def test_backward_rasterizer_benchmark_script_wires_profile_ab():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "benchmark_backward_rasterizers.py").read_text(encoding="utf-8")

    assert "BENCHMARK" in script
    assert "PROFILE_STAGES" in script
    assert "MSPLAT_BACKWARD_DEBUG" in script
    assert '"auto", "pixel", "persplat"' in script
    assert "--backward-rasterizer" in script
    assert "--quality-metrics" in script
    assert "--final-quality" in script
    assert "pixel_atomic_groups_m" in script
    assert "merge ceiling %" in script
    assert "replay_diagonal_m" in script
    assert "replay skip M" in script
    assert "saturated_pixels" in script
    assert "sat px" in script
    assert "print_baseline_deltas" in script
    assert "Baseline deltas vs" in script
    assert "rast_bwd" in script
    assert "train PSNR" in script
    assert "train SSIM" in script


def test_backward_rasterizer_benchmark_script_supports_production_ab():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "benchmark_backward_rasterizers.py").read_text(encoding="utf-8")

    assert "--no-profile-stages" in script
    assert "--roofline-dimensions" in script
    assert "--no-debug" in script
    assert "--project-sh-specialization" in script
    assert "MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION" in script
    assert "project_sh_specialization_variants" in script
    assert "--loss-specialization" in script
    assert "MSPLAT_ENABLE_LOSS_SPECIALIZATION" in script
    assert "loss_specialization_variants" in script
    assert "--raster-backward-specialization" in script
    assert "MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION" in script
    assert "raster_specialization_variants" in script
    assert "--half-sorted-buffers" in script
    assert "MSPLAT_HALF_SORTED_BUFFERS" in script
    assert "half_sorted_buffer_variants" in script
    assert "--warp-merge" in script
    assert "MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE" in script
    assert "warp_merge_variants" in script
    assert "--intersection-key-bits" in script
    assert "MSPLAT_INTERSECTION_KEY_BITS" in script
    assert "intersection_key_bit_variants" in script
    assert "--refine-flag-mode" in script
    assert "MSPLAT_REFINE_FLAG_MODE" in script
    assert "refine_flag_mode_variants" in script
    assert "PROFILE_STAGES_REPORT_EVERY" in script
    assert "parse_duration" in script
    assert "training_ips" in script
    assert "wall_mean_ms" in script
    assert "Sync diagnostics" in script
    assert "flag_sync_readback_max_ms" in script
    assert "densify_count_readback_max_ms" in script
    assert "cpu_pre_refine_drain_max_ms" in script


def test_benchmark_script_supports_async_submit_timing_mode():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "benchmark_backward_rasterizers.py").read_text(encoding="utf-8")
    cli = (repo_root / "cli" / "msplat.cpp").read_text(encoding="utf-8")
    metal_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    model_header = (repo_root / "core" / "include" / "model.hpp").read_text(encoding="utf-8")
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")

    assert "--timing-mode" in script
    assert "--drain-interval" in script
    assert "--pre-refine-drain" in script
    assert "--overflow-poll-interval" in script
    assert "drain-each-iter" in script
    assert "MSPLAT_BENCHMARK_TIMING_MODE" in script
    assert "MSPLAT_BENCHMARK_DRAIN_INTERVAL" in script
    assert "MSPLAT_BENCHMARK_PRE_REFINE_DRAIN" in script
    assert "MSPLAT_OVERFLOW_POLL_INTERVAL" in script
    assert "MSPLAT_REFINE_FLAG_MODE" in script
    assert "MSPLAT_BENCHMARK_TIMING_MODE" in cli
    assert "MSPLAT_BENCHMARK_DRAIN_INTERVAL" in cli
    assert "MSPLAT_BENCHMARK_PRE_REFINE_DRAIN" in cli
    assert "wall-only" in cli
    assert "async-submit" in cli
    assert "drain-every-n" in cli
    assert "pre-refine-drain" in cli
    assert "timing mode:" in cli
    assert "wall-only production throughput" in cli
    assert "wall includes final GPU drain" in cli
    assert "bounded async submit" in cli
    assert "msplat_consume_training_overflow_flag_after_sync" in cli
    assert "CPU submit phases" in cli
    assert "full_iteration" in cli
    assert "pre_refine_drain" in cli
    assert "training_overflow_poll_interval" in metal_source
    assert "MSPLAT_OVERFLOW_POLL_INTERVAL" in metal_source
    assert "flag_sync_readback" in script
    assert "shouldRefineAfterTrain" in model_header
    assert "after_train refine subphases" in cli
    assert "refine_prepare_flags" in cli
    assert "densify_count_readback" in cli
    assert "refine flag preparation subphases" in cli
    assert "flag_sync_readback" in cli
    assert "flag_grow_sample" in cli
    assert "benchmarkRefinePrepareFlagsMs" in model_header
    assert "benchmarkRefineDensifyCountReadbackMs" in model_header
    assert "benchmarkRefineFlagGrowSampleMs" in model_header
    assert "gpuRefineFlagsEnabled" in model_source
    assert "static const bool enabled = std::getenv(\"BENCHMARK\") != nullptr" in model_source
    assert "std::getenv(\"MSPLAT_REFINE_FLAG_MODE\")" in model_source
    assert "useGpuRefineFlags ? 0 : 1" in model_source


def test_metal_benchmark_diagnostics_cache_env_lookup():
    repo_root = Path(__file__).resolve().parents[1]
    metal_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")

    assert "static bool roofline_diagnostics_enabled()" in metal_source
    assert "static const bool enabled = std::getenv(\"MSPLAT_PRINT_ROOFLINE\") != nullptr" in metal_source
    assert "roofline_diagnostics_enabled() && (diag_count == 100" in metal_source


def test_backward_rasterizer_benchmark_defaults_to_production_throughput():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "benchmark_backward_rasterizers.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_binary = tmp_path / "fake_msplat.py"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
(output.parent / "env.json").write_text(json.dumps({
    "profile_stages": os.environ.get("PROFILE_STAGES"),
    "roofline": os.environ.get("MSPLAT_PRINT_ROOFLINE"),
    "backward_debug": os.environ.get("MSPLAT_BACKWARD_DEBUG"),
    "timing_mode": os.environ.get("MSPLAT_BENCHMARK_TIMING_MODE"),
    "drain_interval": os.environ.get("MSPLAT_BENCHMARK_DRAIN_INTERVAL"),
    "overflow_poll_interval": os.environ.get("MSPLAT_OVERFLOW_POLL_INTERVAL"),
    "pre_refine_drain": os.environ.get("MSPLAT_BENCHMARK_PRE_REFINE_DRAIN"),
    "refine_flag_mode": os.environ.get("MSPLAT_REFINE_FLAG_MODE"),
}), encoding="utf-8")
print("=== Benchmark fake ===")
print("wall mean: 1.0 ms/iter")
print("Progress: 100.0% (1/1)  10 gaussians  1.0 it/s")
print("  training loop: 1.0 s (1 steps, 1.0 it/s)")
""",
            encoding="utf-8",
        )
        os.chmod(fake_binary, 0o755)
        output_dir = tmp_path / "bench"

        subprocess.run(
            [
                "python3",
                str(script),
                "/tmp/dataset",
                "--binary",
                str(fake_binary),
                "--output-dir",
                str(output_dir),
                "--modes",
                "auto",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        env = json.loads((output_dir / "auto" / "env.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert env["profile_stages"] is None
    assert env["roofline"] is None
    assert env["backward_debug"] is None
    assert env["timing_mode"] == "wall-only"
    assert env["drain_interval"] == "4"
    assert env["overflow_poll_interval"] == "100"
    assert env["pre_refine_drain"] is None
    assert env["refine_flag_mode"] is None
    assert summary["profile_stages"] is False
    assert summary["profile_mode"] == "production"
    assert summary["debug"] is False
    assert summary["timing_mode"] == "wall-only"
    assert summary["drain_interval"] == 4
    assert summary["overflow_poll_interval"] == 100
    assert summary["pre_refine_drain"] is False
    assert summary["refine_flag_mode"] == "brush"


def test_backward_rasterizer_summary_extracts_sync_diagnostics():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "benchmark_backward_rasterizers.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_binary = tmp_path / "fake_msplat.py"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
print("=== Benchmark (1 iters, 0 warmup, 1.0s total) ===")
print("  mean:   3.0 ms/iter")
print("  median: 0.5 ms/iter")
print("  forced syncs:  9")
print("  forced sync reasons:")
print("    overflow-check: 3")
print("    pre-refine-drain: 2")
print("    refine-flags-readback: 2")
print("    densify-count-readback: 2")
print("")
print("  --- CPU submit phases ---")
print("  full_iteration: mean=1.0  median=0.1  p95=2.0  max=99.0 ms")
print("  after_train: mean=0.2  median=0.0  p95=0.0  max=8.0 ms")
print("  pre_refine_drain: mean=0.3  median=0.0  p95=0.0  max=123.0 ms")
print("")
print("  --- after_train refine subphases (refine events only) ---")
print("  densify_count_readback: mean=4.0  median=4.0  p95=6.0  max=6.0 ms")
print("")
print("  --- refine flag preparation subphases (refine events only) ---")
print("  flag_sync_readback: mean=5.0  median=5.0  p95=7.0  max=7.0 ms")
print("Progress: 100.0% (1/1)  10 gaussians  1.0 it/s")
print("  training loop: 1.0 s (1 steps, 1.0 it/s)")
""",
            encoding="utf-8",
        )
        os.chmod(fake_binary, 0o755)
        output_dir = tmp_path / "bench"

        result = subprocess.run(
            [
                "python3",
                str(script),
                "/tmp/dataset",
                "--binary",
                str(fake_binary),
                "--output-dir",
                str(output_dir),
                "--modes",
                "auto",
                "--timing-mode",
                "async-submit",
                "--pre-refine-drain",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    row = summary["results"][0]
    assert summary["pre_refine_drain"] is True
    assert row["forced_syncs"] == 9
    assert row["forced_sync_reasons"]["pre-refine-drain"] == 2
    assert row["cpu_pre_refine_drain_max_ms"] == 123.0
    assert row["cpu_after_train_max_ms"] == 8.0
    assert row["cpu_full_iteration_max_ms"] == 99.0
    assert row["flag_sync_readback_mean_ms"] == 5.0
    assert row["flag_sync_readback_max_ms"] == 7.0
    assert row["densify_count_readback_mean_ms"] == 4.0
    assert row["densify_count_readback_max_ms"] == 6.0
    assert "Sync diagnostics" in result.stdout
    assert "| auto | 9 | 7.000 | 6.000 | 123.000 | 8.000 | 99.000 | 3 | 2 | 2 | 2 |" in result.stdout


def test_backward_rasterizer_stage_report_interval_enables_stage_profile():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "benchmark_backward_rasterizers.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_binary = tmp_path / "fake_msplat.py"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
(output.parent / "env.json").write_text(json.dumps({
    "profile_stages": os.environ.get("PROFILE_STAGES"),
    "stage_report_every": os.environ.get("PROFILE_STAGES_REPORT_EVERY"),
    "roofline": os.environ.get("MSPLAT_PRINT_ROOFLINE"),
}), encoding="utf-8")
print("=== Benchmark fake ===")
print("mean: 1.0 ms/iter")
print("median: 1.0 ms/iter")
print("Progress: 100.0% (1/1)  10 gaussians  1.0 it/s")
print("  training loop: 1.0 s (1 steps, 1.0 it/s)")
""",
            encoding="utf-8",
        )
        os.chmod(fake_binary, 0o755)
        output_dir = tmp_path / "bench"

        subprocess.run(
            [
                "python3",
                str(script),
                "/tmp/dataset",
                "--binary",
                str(fake_binary),
                "--output-dir",
                str(output_dir),
                "--modes",
                "auto",
                "--stage-report-every",
                "600",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        env = json.loads((output_dir / "auto" / "env.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert env["profile_stages"] == "1"
    assert env["stage_report_every"] == "600"
    assert env["roofline"] == "1"
    assert summary["profile_stages"] is True
    assert summary["profile_mode"] == "stage"


def test_train_step_forced_syncs_are_named_and_counted():
    repo_root = Path(__file__).resolve().parents[1]
    host = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    bindings = (repo_root / "core" / "metal" / "bindings.h").read_text(encoding="utf-8")

    assert "record_forced_sync" in host
    assert "msplat_drain_forced_sync_count" in host
    assert "msplat_drain_forced_sync_counts" in host
    assert "msplat_drain_forced_sync_count" in bindings
    assert "msplat_drain_forced_sync_counts" in bindings
    assert "ForcedSyncStat" in bindings
    assert "msplat_gpu_sync_named" in bindings
    assert "msplat_consume_training_overflow_flag_after_sync" in bindings
    assert 'record_forced_sync("overflow-check")' in host
    assert 'record_forced_sync("dynamic-count-prepass")' in host
    assert 'record_forced_sync("densify-count-readback")' in host


def test_overflow_check_uses_delayed_polling_window():
    repo_root = Path(__file__).resolve().parents[1]
    host = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")

    assert "training_overflow_poll_interval" in host
    assert "MSPLAT_OVERFLOW_POLL_INTERVAL" in host
    assert "g_pending_train_overflow_poll" in host
    assert "overflow_poll_due" in host
    assert "msplat_consume_training_overflow_flag_after_sync" in host
    assert "overflow_poll_interval > 0" in host
    assert "num_points_changed || (iter_count_oc % kOverflowPollInterval)" not in host


def test_refine_decay_runs_on_gpu_without_readback_sync():
    repo_root = Path(__file__).resolve().parents[1]
    model_header = (repo_root / "core" / "include" / "model.hpp").read_text(encoding="utf-8")
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")
    host = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader = (repo_root / "core" / "metal" / "msplat_densify.metal").read_text(encoding="utf-8")
    bindings = (repo_root / "core" / "metal" / "bindings.h").read_text(encoding="utf-8")

    assert "applyRefineDecay(int step)" in model_header
    assert "applyRefineDecay(step)" in model_source
    assert "gpuAlreadySynced" not in model_source
    assert "msplat_consume_training_overflow_flag_after_sync()" in model_source
    assert "msplat_apply_refine_decay(num_active, opacities, scales" in model_source
    assert "void msplat_apply_refine_decay" in bindings
    assert "apply_refine_decay_kernel" in host
    assert "kernel void apply_refine_decay_kernel" in shader


def test_refine_flag_preparation_reuses_cpu_scratch():
    repo_root = Path(__file__).resolve().parents[1]
    model_header = (repo_root / "core" / "include" / "model.hpp").read_text(encoding="utf-8")
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")

    assert "refineScratchSampleKeys" in model_header
    assert "refineScratchWeights" not in model_header
    assert "auto pruneWeight = [&](int i)" in model_source
    assert "auto growWeight = [&](int i)" in model_source
    assert "weightedSampleWithoutReplacement(N, prunedCount, selected, rng, refineScratchSampleKeys, pruneWeight)" in model_source
    assert "weightedSampleWithoutReplacement(N, growCount, selected, rng, refineScratchSampleKeys, growWeight)" in model_source
    assert "std::fill(split, split + N, 0)" not in model_source
    assert "std::fill(dup, dup + N, 0)" not in model_source
    assert "split[i] = selected[i] ? 1 : 0;" in model_source
    assert "dup[i] = 0;" in model_source


def test_refine_reuses_bounds_for_scene_scale_update():
    repo_root = Path(__file__).resolve().parents[1]
    model_header = (repo_root / "core" / "include" / "model.hpp").read_text(encoding="utf-8")
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")

    assert "float *sceneScaleOut" in model_header
    assert "std::array<float, 3> extents" in model_source
    assert "prepareBrushRefineFlags(step, check_screen, allowGrowth, cullCenter, &refineSceneScale)" in model_source
    assert "updateMeanLrSceneScale(refineSceneScale, step)" in model_source


def test_densify_split_offsets_generated_on_gpu():
    repo_root = Path(__file__).resolve().parents[1]
    model_header = (repo_root / "core" / "include" / "model.hpp").read_text(encoding="utf-8")
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")
    host = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader = (repo_root / "core" / "metal" / "msplat_densify.metal").read_text(encoding="utf-8")
    bindings = (repo_root / "core" / "metal" / "bindings.h").read_text(encoding="utf-8")

    assert "densify_random_samples" not in model_header
    assert "densify_random_samples" not in model_source
    assert "densify_random_samples" not in bindings
    assert "ENC_SCALAR(enc, growth_seed_u32, 3)" in host
    assert "constant uint& split_seed" in shader
    assert "msplat_normal_sample(split_seed" in shader


def test_opacity_reset_runs_on_gpu_without_readback_sync():
    repo_root = Path(__file__).resolve().parents[1]
    model_source = (repo_root / "core" / "src" / "model.cpp").read_text(encoding="utf-8")
    host = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    shader = (repo_root / "core" / "metal" / "msplat_densify.metal").read_text(encoding="utf-8")
    bindings = (repo_root / "core" / "metal" / "bindings.h").read_text(encoding="utf-8")

    assert 'msplat_gpu_sync_named("opacity-reset-readback")' not in model_source
    assert "msplat_reset_opacity(num_active, opacities, resetLogit)" in model_source
    assert "msplat_zero_tensor(adam_exp_avg[5])" in model_source
    assert "adam_exp_avg[5].zero()" not in model_source
    assert "void msplat_reset_opacity" in bindings
    assert "void msplat_zero_tensor" in bindings
    assert "void msplat_zero_tensor(MTensor &tensor)" in host
    assert "reset_opacity_kernel_cpso" in host
    assert "kernel void reset_opacity_kernel" in shader


def test_backward_rasterizer_benchmark_script_supports_auto_quality_metrics():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "benchmark_backward_rasterizers.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_binary = tmp_path / "fake_msplat.py"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
(output.parent / "argv.json").write_text(json.dumps(args), encoding="utf-8")
(output.parent / "env.json").write_text(json.dumps({
    "project_sh_specialization": os.environ.get("MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION"),
    "loss_specialization": os.environ.get("MSPLAT_ENABLE_LOSS_SPECIALIZATION"),
    "raster_specialization": os.environ.get("MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION"),
    "half_sorted_buffers": os.environ.get("MSPLAT_HALF_SORTED_BUFFERS"),
    "warp_merge": os.environ.get("MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE"),
    "intersection_key_bits": os.environ.get("MSPLAT_INTERSECTION_KEY_BITS"),
}), encoding="utf-8")
key_mode = os.environ.get("MSPLAT_INTERSECTION_KEY_BITS")
splats = 12 if key_mode == "auto" else 10
psnr = 29.50 if key_mode == "auto" else 30.00
ssim = 0.8800 if key_mode == "auto" else 0.9000
l1 = 0.01100 if key_mode == "auto" else 0.01000
print("=== Benchmark fake ===")
print("mean: 1.0 ms/iter")
print("median: 1.0 ms/iter")
print(f"Progress: 100.0% (1/1)  {splats} gaussians  1.0 it/s")
print("  training loop: 1.0 s (1 steps, 1.0 it/s)")
print("  loss_fwd_bwd     median=0.2ms mean=0.2ms")
print("  rast_bwd median=0.9ms mean=1.0ms")
print("  rast_bwd median=0.5ms mean=0.6ms")
print("  proj_sh_bwd_adam median=0.3ms mean=0.3ms")
print("  TOTAL (sum medians) 1.0ms")
print(f"  train PSNR:      {psnr:.2f} dB")
print(f"  train SSIM:      {ssim:.4f}")
print(f"  train L1:        {l1:.5f}")
""",
            encoding="utf-8",
        )
        os.chmod(fake_binary, 0o755)
        output_dir = tmp_path / "bench"

        result = subprocess.run(
            [
                "python3",
                str(script),
                "/tmp/dataset",
                "--binary",
                str(fake_binary),
                "--output-dir",
                str(output_dir),
                "--modes",
                "auto",
                "--raster-backward-specialization",
                "both",
                "--half-sorted-buffers",
                "both",
                "--warp-merge",
                "both",
                "--intersection-key-bits",
                "default",
                "auto",
                "--quality-metrics",
                "--",
                "--alpha-mode",
                "transparent",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        launched_args = json.loads((output_dir / "auto" / "argv.json").read_text(encoding="utf-8"))
        base_env = json.loads((output_dir / "auto" / "env.json").read_text(encoding="utf-8"))
        half_env = json.loads((output_dir / "auto-half" / "env.json").read_text(encoding="utf-8"))
        key_env = json.loads((output_dir / "auto-key-auto" / "env.json").read_text(encoding="utf-8"))
        warp_env = json.loads((output_dir / "auto-warp-merge" / "env.json").read_text(encoding="utf-8"))
        half_key_env = json.loads((output_dir / "auto-half-key-auto" / "env.json").read_text(encoding="utf-8"))
        specialized_env = json.loads((output_dir / "auto-rb-spec" / "env.json").read_text(encoding="utf-8"))
        combined_env = json.loads((output_dir / "auto-rb-spec-half" / "env.json").read_text(encoding="utf-8"))
        combined_key_env = json.loads((output_dir / "auto-rb-spec-half-key-auto" / "env.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    mode_index = launched_args.index("--backward-rasterizer") + 1
    assert launched_args[mode_index] == "auto"
    assert "--final-quality" in launched_args
    assert "--quality-metrics" not in launched_args
    assert "--alpha-mode" in launched_args
    assert base_env["project_sh_specialization"] is None
    assert base_env["loss_specialization"] is None
    assert base_env["raster_specialization"] is None
    assert base_env["half_sorted_buffers"] is None
    assert base_env["intersection_key_bits"] is None
    assert half_env["raster_specialization"] is None
    assert half_env["half_sorted_buffers"] == "1"
    assert half_env["intersection_key_bits"] is None
    assert key_env["intersection_key_bits"] == "auto"
    assert warp_env["warp_merge"] == "1"
    assert warp_env["intersection_key_bits"] is None
    assert half_key_env["half_sorted_buffers"] == "1"
    assert half_key_env["intersection_key_bits"] == "auto"
    assert specialized_env["raster_specialization"] == "1"
    assert specialized_env["half_sorted_buffers"] is None
    assert specialized_env["intersection_key_bits"] is None
    assert combined_env["raster_specialization"] == "1"
    assert combined_env["half_sorted_buffers"] == "1"
    assert combined_env["intersection_key_bits"] is None
    assert combined_key_env["raster_specialization"] == "1"
    assert combined_key_env["half_sorted_buffers"] == "1"
    assert combined_key_env["intersection_key_bits"] == "auto"
    assert "auto" in result.stdout
    assert "auto-rb-spec" in result.stdout
    assert "auto-half" in result.stdout
    assert "auto-rb-spec-half" in result.stdout
    assert "auto-warp-merge" in result.stdout
    assert "auto-key-auto" in result.stdout
    assert "auto-rb-spec-half-key-auto" in result.stdout
    assert "Baseline deltas vs auto" in result.stdout
    assert "| auto | 1.00 | 1.00 | 1000.000 | 1.000 | 1.000 | 0.200 | 0.500 | 0.300 | 30.00 | 0.9000 | 0.01000 | 10 |" in result.stdout
    assert "Large quality/count drift vs auto" in result.stdout
    assert "auto-key-auto: PSNR -0.50 dB" in result.stdout
    assert "splats +20.0%" in result.stdout
    assert "Summary:" in result.stdout
    assert "30.00" in result.stdout
    assert summary["schema_version"] == 1
    assert summary["dataset"] == "/tmp/dataset"
    assert summary["iters"] == 120
    assert summary["quality_metrics"] is True
    assert summary["project_sh_specialization"] == "off"
    assert summary["loss_specialization"] == "off"
    assert summary["raster_backward_specialization"] == "both"
    assert summary["half_sorted_buffers"] == "both"
    assert summary["warp_merge"] == "both"
    assert summary["intersection_key_bits"] == ["default", "auto"]
    assert summary["msplat_args"] == ["--alpha-mode", "transparent"]
    assert [row["mode"] for row in summary["results"]] == [
        "auto",
        "auto-key-auto",
        "auto-warp-merge",
        "auto-warp-merge-key-auto",
        "auto-half",
        "auto-half-key-auto",
        "auto-half-warp-merge",
        "auto-half-warp-merge-key-auto",
        "auto-rb-spec",
        "auto-rb-spec-key-auto",
        "auto-rb-spec-warp-merge",
        "auto-rb-spec-warp-merge-key-auto",
        "auto-rb-spec-half",
        "auto-rb-spec-half-key-auto",
        "auto-rb-spec-half-warp-merge",
        "auto-rb-spec-half-warp-merge-key-auto",
    ]
    assert summary["results"][0]["log"].endswith("/auto/run.log")
    assert summary["results"][0]["wall_mean_ms"] == 1000.0
    assert summary["results"][0]["gpu_stage_total_median_ms"] == 1.0
    assert summary["results"][0]["loss_fwd_bwd_median_ms"] == 0.2
    assert summary["results"][0]["proj_sh_bwd_adam_median_ms"] == 0.3
    assert summary["quality_warnings"][0]["mode"] == "auto-key-auto"
    assert "PSNR -0.50 dB" in summary["quality_warnings"][0]["reasons"]


def test_benchmark_summary_extracts_gpu_stage_total():
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "benchmark_backward_rasterizers.py").read_text(encoding="utf-8")
    compare = (repo_root / "scripts" / "compare_benchmark_summaries.py").read_text(encoding="utf-8")

    assert "gpu_stage_total_median_ms" in script
    assert "wall_mean_ms" in script
    assert "wall mean delta" in compare
    assert "loss_fwd_bwd_median_ms" in script
    assert "proj_sh_bwd_adam_median_ms" in script
    assert "gpu_stage_total_median_ms" in compare


def test_backward_rasterizer_benchmark_script_supports_project_and_loss_specialization_ab():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "benchmark_backward_rasterizers.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fake_binary = tmp_path / "fake_msplat.py"
        fake_binary.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
(output.parent / "env.json").write_text(json.dumps({
    "project_sh_specialization": os.environ.get("MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION"),
    "loss_specialization": os.environ.get("MSPLAT_ENABLE_LOSS_SPECIALIZATION"),
}), encoding="utf-8")
print("=== Benchmark fake ===")
print("mean: 1.0 ms/iter")
print("median: 1.0 ms/iter")
print("Progress: 100.0% (1/1)  10 gaussians  1.0 it/s")
print("  training loop: 1.0 s (1 steps, 1.0 it/s)")
""",
            encoding="utf-8",
        )
        os.chmod(fake_binary, 0o755)
        output_dir = tmp_path / "bench"

        subprocess.run(
            [
                "python3",
                str(script),
                "/tmp/dataset",
                "--binary",
                str(fake_binary),
                "--output-dir",
                str(output_dir),
                "--modes",
                "auto",
                "--project-sh-specialization",
                "both",
                "--loss-specialization",
                "both",
                "--no-profile-stages",
                "--no-debug",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        base_env = json.loads((output_dir / "auto" / "env.json").read_text(encoding="utf-8"))
        loss_env = json.loads((output_dir / "auto-loss-spec" / "env.json").read_text(encoding="utf-8"))
        project_env = json.loads((output_dir / "auto-project-sh-spec" / "env.json").read_text(encoding="utf-8"))
        both_env = json.loads((output_dir / "auto-project-sh-spec-loss-spec" / "env.json").read_text(encoding="utf-8"))
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert base_env["project_sh_specialization"] is None
    assert base_env["loss_specialization"] is None
    assert loss_env["project_sh_specialization"] is None
    assert loss_env["loss_specialization"] == "1"
    assert project_env["project_sh_specialization"] == "1"
    assert project_env["loss_specialization"] is None
    assert both_env["project_sh_specialization"] == "1"
    assert both_env["loss_specialization"] == "1"
    assert summary["project_sh_specialization"] == "both"
    assert summary["loss_specialization"] == "both"
    assert [row["mode"] for row in summary["results"]] == [
        "auto",
        "auto-loss-spec",
        "auto-project-sh-spec",
        "auto-project-sh-spec-loss-spec",
    ]


def test_compare_benchmark_summaries_script_prints_cross_run_table():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "compare_benchmark_summaries.py"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_a = tmp_path / "garden" / "summary.json"
        summary_b = tmp_path / "playroom" / "summary.json"
        summary_a.parent.mkdir()
        summary_b.parent.mkdir()
        base_summary = {
            "dataset": "/datasets/garden",
            "output_dir": "/tmp/msplat-garden",
            "results": [
                {
                    "mode": "pixel",
                    "training_ips": 100.0,
                    "wall_mean_ms": 10.0,
                    "iter_median_ms": 3.0,
                    "gpu_stage_total_median_ms": 2.0,
                    "loss_fwd_bwd_median_ms": 0.4,
                    "rast_bwd_median_ms": 1.0,
                    "proj_sh_bwd_adam_median_ms": 0.5,
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "l1": 0.04,
                    "splats": 100,
                },
                {
                    "mode": "pixel-key-auto",
                    "training_ips": 110.0,
                    "wall_mean_ms": 9.0,
                    "iter_median_ms": 2.7,
                    "gpu_stage_total_median_ms": 1.8,
                    "loss_fwd_bwd_median_ms": 0.36,
                    "rast_bwd_median_ms": 0.9,
                    "proj_sh_bwd_adam_median_ms": 0.45,
                    "psnr": 19.7,
                    "ssim": 0.79,
                    "l1": 0.045,
                    "splats": 106,
                },
            ],
            "quality_warnings": [
                {"mode": "pixel-key-auto", "reasons": ["PSNR -0.30 dB", "splats +6.0%"]}
            ],
        }
        summary_a.write_text(json.dumps(base_summary), encoding="utf-8")
        summary_b.write_text(
            json.dumps(
                {
                    **base_summary,
                    "dataset": "/datasets/playroom",
                    "output_dir": "/tmp/msplat-playroom",
                    "quality_warnings": [],
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            ["python3", str(script), str(summary_a), str(summary_b)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    assert "## msplat-garden" in result.stdout
    assert "dataset: /datasets/garden" in result.stdout
    assert "| pixel-key-auto | 110.00 | +10.0% | -10.0% | -10.0% | -10.0% | -10.0% | -10.0% | -10.0% | -0.30 | -0.0100 | +12.5% | +6.0% | PSNR -0.30 dB, splats +6.0% |" in result.stdout
    assert "## msplat-playroom" in result.stdout


def test_stage_profiler_uses_synchronized_command_buffers():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")

    assert "run_profiled_stage" in host_source
    assert "PROFILE_STAGES: synchronized command-buffer profiling enabled" in host_source
    assert "stage_cb.GPUEndTime - stage_cb.GPUStartTime" in host_source


def test_fused_ssim_threadgroup_memory_budget_is_explicit():
    repo_root = Path(__file__).resolve().parents[1]
    host_source = (repo_root / "core" / "metal" / "msplat_metal.mm").read_text(encoding="utf-8")
    loss_source = (repo_root / "core" / "metal" / "msplat_loss.metal").read_text(encoding="utf-8")

    assert "#define SSIM_FUSED_TG_HP_BYTES" in loss_source
    assert "#define SSIM_FUSED_TG_DERIV_BYTES" in loss_source
    assert "#define SSIM_FUSED_TG_REDUCTION_BYTES" in loss_source
    assert "Keep this fused kernel one channel at a time" in loss_source
    assert "SSIM_STATS_PER_CHANNEL" in loss_source

    assert "requireStaticThreadgroupMemoryFits" in host_source
    assert "staticThreadgroupMemoryLength" in host_source
    assert "maxThreadgroupMemoryLength" in host_source
    assert "ssim_fused_v_fwd_h_bwd_kernel" in host_source


def test_native_cli_can_log_image_loading():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--log-image-loading",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"image cache 1/1 (100.0%)" in combined_output
    assert b"loading image image.png" in combined_output
    assert b"loaded image image.png" in combined_output
    assert b"source 2x1, decoded 2x1, final 2x1" in combined_output
    assert b"prepared target image.png" in combined_output
    assert b"image 1/1 (100.0%) prepared target image.png" in combined_output
    assert b"\rmsplat: image cache 1/1 (100.0%) loading image image.png" in result.stderr
    assert b"\nmsplat: loaded image image.png" not in result.stderr
    assert b"\nmsplat: prepared target image.png" not in result.stderr


def test_native_cli_logs_image_loading_by_default():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"image cache 1/1 (100.0%)" in combined_output
    assert b"image 1/1 (100.0%) prepared target image.png" in combined_output


def test_native_cli_can_disable_default_image_loading_log():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--no-log-image-loading",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"image cache" not in combined_output
    assert b"prepared target image.png" not in combined_output


def test_native_cli_can_skip_final_quality_pass():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--no-log-image-loading",
                "--no-final-quality",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"=== Timings ===" in combined_output
    assert b"Saved " in combined_output
    assert b"=== Final Quality ===" not in combined_output
    assert b"train PSNR" not in combined_output


def test_native_cli_skips_final_quality_by_default():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--no-log-image-loading",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"=== Timings ===" in combined_output
    assert b"Saved " in combined_output
    assert b"=== Final Quality ===" not in combined_output
    assert b"train PSNR" not in combined_output


def test_native_cli_can_enable_final_quality_pass():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--no-log-image-loading",
                "--final-quality",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"=== Final Quality ===" in combined_output
    assert b"train PSNR" in combined_output


def test_native_cli_clamps_image_loading_status_to_terminal_width():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        image_filename = "very-long-gallery-installation-image-name-that-would-wrap-a-terminal-row.png"
        _write_minimal_colmap_text_dataset(tmp, image_filename=image_filename)
        env = os.environ.copy()
        env["MSPLAT_IMAGE_LOADING_COLUMNS"] = "96"

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--log-image-loading",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    status_segments = [
        segment.replace(b"\033[K", b"")
        for segment in result.stderr.split(b"\r")
        if segment.startswith(b"msplat:")
    ]
    assert status_segments
    assert all(len(segment) <= 96 for segment in status_segments)


def test_native_cli_decodes_image_at_max_resolution():
    if not NATIVE_CLI.exists():
        pytest.skip("native CLI is not built")

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp)
        pixels = bytes([
            255, 0, 0, 255,  0, 255, 0, 255,  0, 0, 255, 255,  255, 255, 255, 255,
            255, 0, 255, 255,  0, 255, 255, 255,  255, 255, 0, 255,  0, 0, 0, 255,
        ])
        _write_rgba_png(os.path.join(tmp, "images", "image.png"), 4, 2, pixels)
        with open(os.path.join(tmp, "sparse", "cameras.txt"), "w", encoding="utf-8") as f:
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("1 PINHOLE 4 2 4.0 4.0 2.0 1.0\n")

        result = subprocess.run(
            [
                str(NATIVE_CLI),
                tmp,
                "--output", os.path.join(tmp, "out.ply"),
                "--total-train-iters", "1",
                "--num-downscales", "0",
                "--ssim-weight", "0.0",
                "--progress-every", "1",
                "--save-every", "-1",
                "--max-resolution", "2",
                "--log-image-loading",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    combined_output = result.stdout + result.stderr
    assert b"source 4x2, decoded 2x1, final 2x1" in combined_output


def test_training_config_defaults():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.iterations == 30000
    assert cfg.sh_degree == 3
    assert cfg.sh_degree_interval == 0
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.num_downscales == 0
    assert cfg.refine_every == 200
    assert cfg.warmup_length == 0
    assert cfg.reset_alpha_every == 0
    assert cfg.densify_grad_thresh == pytest.approx(0.0025)
    assert cfg.stop_screen_size_at == 15000
    assert cfg.split_screen_size == pytest.approx(0.25)
    assert cfg.growth_stop_iter == 15000
    assert cfg.max_splats == 10000000
    assert cfg.growth_select_fraction == pytest.approx(0.25)
    assert cfg.match_alpha_weight == pytest.approx(0.1)
    assert cfg.bg_color == pytest.approx([0.0, 0.0, 0.0])
    assert cfg.lpips_loss_weight == pytest.approx(0.0)
    assert cfg.background_noise_strength == pytest.approx(0.1)
    assert cfg.opac_decay == pytest.approx(0.004)
    assert cfg.scale_decay == pytest.approx(0.002)
    assert cfg.mean_noise_weight == pytest.approx(50.0)
    assert cfg.lr_mean == pytest.approx(0.00002)
    assert cfg.lr_mean_end == pytest.approx(0.0000002)
    assert cfg.lr_scale == pytest.approx(0.007)
    assert cfg.lr_scale_end == pytest.approx(0.005)
    assert cfg.lr_rotation == pytest.approx(0.002)
    assert cfg.lr_coeffs_dc == pytest.approx(0.002)
    assert cfg.lr_coeffs_sh_scale == pytest.approx(10.0)
    assert cfg.lr_opac == pytest.approx(0.012)
    assert cfg.random_init_scene_scale == pytest.approx(0.0)
    assert cfg.reduce_second_moment is False


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0, lpips_loss_weight=0.25)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0
    assert cfg.lpips_loss_weight == pytest.approx(0.25)


def test_training_config_mutable():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    cfg.iterations = 500
    assert cfg.iterations == 500


# ── Dataset tests ────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


def test_dataset_rejects_invalid_eval_split_period():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False)

        with pytest.raises(ValueError, match="test_every must be at least 2"):
            Dataset(tmp, eval_mode=True, test_every=0)


def test_dataset_detects_transparent_png_alpha():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=True)
        ds = Dataset(tmp)

        assert ds.camera_has_alpha(0) is True


def test_nerfstudio_pose_keeps_opengl_forward_direction():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False)
        ds = Dataset(tmp)

        pose = ds.camera_pose(0)

        np.testing.assert_allclose(pose[:3, 2], [0.0, 0.0, 1.0], atol=1e-6)


def test_dataset_ignores_fully_opaque_png_alpha_channel():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False)
        ds = Dataset(tmp)

        assert ds.camera_has_alpha(0) is False


def test_dataset_detects_explicit_mask_path():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False, mask=True)
        ds = Dataset(tmp)

        assert ds.camera_has_alpha(0) is False
        assert ds.camera_has_mask(0) is True


def test_dataset_detects_colmap_text_with_image_extension_mask_suffix():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(tmp, mask_filename="image.png.png")
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.camera_has_alpha(0) is False
        assert ds.camera_has_mask(0) is True


def test_dataset_filters_nonfinite_point_cloud_rows():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(
            tmp,
            alpha=False,
            point_rows=[
                ("0", "0", "0", "255", "255", "255"),
                ("nan", "0", "0", "255", "0", "0"),
                ("1", "2", "3", "0", "255", "0"),
                ("0", "inf", "0", "0", "0", "255"),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 2


def test_dataset_keeps_point_rows_with_nonfinite_colors():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(
            tmp,
            alpha=False,
            point_rows=[
                ("0", "0", "0", "nan", "255", "255"),
                ("1", "2", "3", "0", "inf", "0"),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 2


def test_dataset_binary_float64_ply_filters_points_and_clamps_colors():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False, points=False)
        _write_binary_float64_point_ply(
            os.path.join(tmp, "points3D.ply"),
            [
                (0.0, 0.0, 0.0, float("nan"), 2.0, -1.0),
                (float("nan"), 1.0, 1.0, 0.0, 1.0, 0.0),
                (1.0, 2.0, 3.0, 0.25, float("inf"), 0.75),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 2


def test_dataset_all_nonfinite_points_falls_back_to_random_init():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(
            tmp,
            alpha=False,
            point_rows=[
                ("nan", "0", "0", "255", "0", "0"),
                ("0", "-inf", "0", "0", "255", "0"),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 0


def test_colmap_text_filters_nonfinite_point_rows():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(
            tmp,
            point_rows=[
                ("1", "0", "0", "0", "255", "255", "255"),
                ("2", "nan", "0", "0", "255", "0", "0"),
                ("3", "2", "3", "4", "0", "255", "0"),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 2


def test_colmap_text_keeps_point_rows_with_nonfinite_colors():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_colmap_text_dataset(
            tmp,
            point_rows=[
                ("1", "0", "0", "0", "nan", "255", "-10"),
                ("2", "1", "2", "3", "300", "inf", "0"),
            ],
        )
        ds = Dataset(tmp)

        assert ds.num_train == 1
        assert ds.initial_point_count == 2


def test_train_one_step_with_explicit_mask():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False, mask=True)
        ds = Dataset(tmp)
        trainer = GaussianTrainer(ds, TrainingConfig(iterations=1, num_downscales=0, ssim_weight=0.0))

        stats = trainer.step()

        assert ds.camera_has_mask(0) is True
        assert stats.iteration == 1
        assert stats.splat_count == 1


def test_train_one_step_with_transparent_alpha():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=True)
        ds = Dataset(tmp)
        trainer = GaussianTrainer(ds, TrainingConfig(iterations=1, num_downscales=0, ssim_weight=0.0))

        stats = trainer.step()

        assert ds.camera_has_alpha(0) is True
        assert ds.camera_has_mask(0) is False
        assert stats.iteration == 1
        assert stats.splat_count == 1


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden fixture not available")
def test_lpips_loss_weight_runs_one_step():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0, lpips_loss_weight=0.1)
    trainer = GaussianTrainer(ds, cfg)

    stats = trainer.step()

    assert stats.iteration == 1
    assert stats.splat_count > 0


def test_random_init_when_dataset_has_no_point_cloud():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False, points=False)
        ds = Dataset(tmp)
        trainer = GaussianTrainer(ds, TrainingConfig(iterations=1, random_init_scene_scale=0.5))

        assert trainer.splat_count == 10000


def test_reduce_second_moment_one_step():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False)
        ds = Dataset(tmp)
        trainer = GaussianTrainer(
            ds,
            TrainingConfig(iterations=1, num_downscales=0, reduce_second_moment=True),
        )

        stats = trainer.step()

    assert stats.iteration == 1
    assert stats.splat_count == 1


def test_tiny_image_safe_with_zero_schedule_and_progressive_downscale():
    from msplat import Dataset, GaussianTrainer, TrainingConfig

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=False)
        ds = Dataset(tmp)
        cfg = TrainingConfig(
            iterations=1,
            num_downscales=2,
            resolution_schedule=0,
            refine_every=0,
            ssim_weight=0.0,
        )
        trainer = GaussianTrainer(ds, cfg)

        stats = trainer.step()
        img = trainer.render(0)

        assert stats.iteration == 1
        assert img.shape[0] >= 1
        assert img.shape[1] >= 1


# ── Training tests ───────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_short():
    """Train 50 steps at 4x downscale — verify it runs without error."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    steps_seen = []
    trainer.train(lambda s: steps_seen.append(s.iteration), callback_every=10)

    assert trainer.iteration == 50
    assert trainer.splat_count > 100000
    assert steps_seen == [10, 20, 30, 40, 50]


@pytest.mark.gpu_heavy
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_default_refine_grows_after_warmup():
    """Default refine settings should grow garden after the strict warmup gate."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=600, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)
    initial_count = trainer.splat_count

    for _ in range(600):
        trainer.step()

    assert trainer.iteration == 600
    assert trainer.splat_count > initial_count


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_step_by_step():
    """Manual step loop works."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        stats = trainer.step()

    assert stats.iteration == 10
    assert stats.splat_count > 0
    assert stats.ms_per_step > 0


# ── Render tests ─────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render():
    """Render produces valid image array."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    img = trainer.render(0)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Values should be in [0, 1] range (approximately)
    assert img.min() >= -0.1
    assert img.max() <= 1.5


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render_from_pose():
    """Pose render uses reference intrinsics without copying the full camera payload."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)
    pose = ds.camera_pose(0)

    img = trainer.render_from_pose(pose, ref_cam_idx=0)

    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_one_step_with_mip_splatting():
    """MIP splatting path trains and renders without invalid pixels."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0, render_mip=True)
    trainer = GaussianTrainer(ds, cfg)

    stats = trainer.step()
    img = trainer.render(0)

    assert stats.iteration == 1
    assert np.isfinite(img).all()
    assert img.shape[2] == 3


# ── Export tests ─────────────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply():
    """PLY export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000  # non-trivial file
        comments = _ply_comments(path)
        assert "comment Generated by msplat at iteration 10" in comments
        assert "comment SplatRenderMode: default" in comments
    finally:
        os.unlink(path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply_persists_mip_render_mode():
    """PLY export records mip render mode like Brush."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0, render_mip=True)
    trainer = GaussianTrainer(ds, cfg)
    trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert "comment SplatRenderMode: mip" in _ply_comments(path)
    finally:
        os.unlink(path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_ply_restores_step_and_mip_render_mode():
    """PLY import restores saved iteration and render mode metadata."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    source = GaussianTrainer(
        ds,
        TrainingConfig(iterations=1, num_downscales=0, render_mip=True),
    )
    source.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        src_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        dst_path = f.name

    try:
        source.export_ply(src_path)

        loaded = GaussianTrainer(
            ds,
            TrainingConfig(iterations=1, num_downscales=0, render_mip=False),
        )
        loaded_step = loaded.load_ply(src_path)
        assert loaded_step == 1
        assert loaded.iteration == 1
        assert loaded.splat_count == source.splat_count

        loaded.export_ply(dst_path)
        assert "comment SplatRenderMode: mip" in _ply_comments(dst_path)
    finally:
        os.unlink(src_path)
        os.unlink(dst_path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_ply_accepts_brush_property_order():
    """Gaussian PLY import follows property names, not msplat's export order."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)

    fields = [
        ("x", 1.0),
        ("y", 2.0),
        ("z", 3.0),
        ("scale_0", -0.1),
        ("scale_1", -0.2),
        ("scale_2", -0.3),
        ("opacity", 0.25),
        ("rot_0", 1.0),
        ("rot_1", 0.0),
        ("rot_2", 0.0),
        ("rot_3", 0.0),
        ("f_dc_0", 0.11),
        ("f_dc_1", 0.22),
        ("f_dc_2", 0.33),
    ]

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        src_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        dst_path = f.name

    try:
        with open(src_path, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(b"comment Exported from Brush\n")
            f.write(b"comment SplatRenderMode: default\n")
            f.write(b"element vertex 1\n")
            for name, _ in fields:
                f.write(f"property float {name}\n".encode("ascii"))
            f.write(b"end_header\n")
            f.write(struct.pack("<" + "f" * len(fields), *(value for _, value in fields)))

        trainer = GaussianTrainer(ds, TrainingConfig(iterations=1, num_downscales=0))
        assert trainer.load_ply(src_path) == 0
        assert trainer.splat_count == 1

        trainer.export_ply(dst_path)
        vertex = _read_first_ply_vertex(dst_path)
        for name, expected in dict(fields).items():
            assert vertex[name] == pytest.approx(expected)
    finally:
        os.unlink(src_path)
        os.unlink(dst_path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_refine_prunes_near_zero_quaternions():
    """Refine pruning matches Brush's render-time near-zero quaternion cull."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    valid = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "scale_0": -2.0, "scale_1": -2.0, "scale_2": -2.0,
        "opacity": 0.0,
        "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
        "f_dc_0": 0.1, "f_dc_1": 0.1, "f_dc_2": 0.1,
    }
    invalid = dict(valid)
    invalid.update({"rot_0": 1e-4, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0})

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        out_path = f.name

    try:
        _write_gaussian_ply(path, [valid, invalid])
        cfg = TrainingConfig(
            iterations=1,
            refine_every=1,
            warmup_length=0,
            reset_alpha_every=0,
            densify_grad_thresh=999.0,
            growth_select_fraction=0.0,
            split_screen_size=0.0,
            growth_stop_iter=10,
            max_splats=3,
            opac_decay=0.0,
            scale_decay=0.0,
            num_downscales=0,
        )
        trainer = GaussianTrainer(ds, cfg)
        trainer.load_ply(path)
        assert trainer.splat_count == 2

        trainer.step()

        trainer.export_ply(out_path)
        for vertex in _read_ply_vertices(out_path):
            quat_norm_sqr = sum(vertex[f"rot_{i}"] ** 2 for i in range(4))
            assert quat_norm_sqr >= 1e-6
    finally:
        os.unlink(path)
        os.unlink(out_path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_lod_ply():
    """LOD PLY export writes an importance-ranked subset."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)
    trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_lod_ply(path, 128)
        assert os.path.exists(path)
        assert _ply_vertex_count(path) == 128
        assert "comment msplat_lod_score_source training-stats" in _ply_comments(path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_decimate_to_lod_can_continue_training():
    """In-memory LOD decimation updates the active model and keeps it trainable."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=1, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)
    trainer.step()

    trainer.decimate_to_lod(128)
    assert trainer.splat_count == 128

    stats = trainer.step()
    assert stats.splat_count == 128

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert _ply_vertex_count(path) == 128
    finally:
        os.unlink(path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_lod_refine_step_accepts_forced_downscale():
    """Python LOD refinement can train at a caller-selected image downscale."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=2, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)
    trainer.step()

    trainer.decimate_to_lod(128)
    stats = trainer.step(forced_downscale=2, apply_refine=False)

    assert stats.iteration == 2
    assert stats.splat_count == 128


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_splat():
    """Splat export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name

    try:
        trainer.export_splat(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000
    finally:
        os.unlink(path)


# ── Eval tests ───────────────────────────────────────────────────────────────


@pytest.mark.gpu_heavy
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_evaluate():
    """Evaluation returns valid metrics dict."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    trainer.train(lambda s: None, callback_every=50)
    metrics = trainer.evaluate()

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "l1" in metrics
    assert "num_test" in metrics
    assert metrics["num_test"] > 0
    assert metrics["psnr"] > 10  # sanity — should be at least somewhat trained
    assert 0 < metrics["ssim"] < 1
    assert metrics["l1"] > 0


# ── Checkpoint tests ────────────────────────────────────────────────────────


@pytest.mark.gpu_heavy
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_save_load():
    """Save checkpoint, load it, verify state is preserved."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    splats_at_50 = trainer.splat_count

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 1000

        # Load into a fresh trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.iteration == 50
        assert trainer2.splat_count == splats_at_50
    finally:
        os.unlink(ckpt_path)


@pytest.mark.gpu_heavy
@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_resume_training():
    """Train 50 → save → load → train 50 more. Verify it completes."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)

        # Resume in a new trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        for _ in range(50):
            stats = trainer2.step()

        assert trainer2.iteration == 100
        assert stats.splat_count > 0
        assert stats.ms_per_step > 0
    finally:
        os.unlink(ckpt_path)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_persists_scale_lr_schedule():
    """Checkpoint format stores custom scale LR schedule values for resume."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(
        iterations=10,
        num_downscales=0,
        lr_scale=0.007,
        lr_scale_end=0.003,
    )
    trainer = GaussianTrainer(ds, cfg)
    trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        with open(ckpt_path, "rb") as f:
            header = f.read(6 * 4 + 10 * 4)

        magic, version, step, num_active, _, _ = struct.unpack_from("<6I", header, 0)
        assert magic == 0x4C50534D
        assert version >= 2
        assert step == 1
        assert num_active == trainer.splat_count

        values = struct.unpack_from("<10f", header, 6 * 4)
        assert values[8] == pytest.approx(0.007)
        assert values[9] == pytest.approx(0.003)
    finally:
        os.unlink(ckpt_path)
