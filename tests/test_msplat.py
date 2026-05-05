"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os
import json
import struct
import zlib
import binascii

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)


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


def _write_minimal_nerfstudio_dataset(root, alpha, mask=False, points=True):
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
        with open(os.path.join(root, "points3D.ply"), "w", encoding="utf-8") as f:
            f.write(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 1\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
                "0 0 0 255 255 255\n"
            )

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


def _write_minimal_colmap_text_dataset(root, mask_filename=None):
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "sparse"))

    _write_rgba_png(os.path.join(root, "images", "image.png"), 2, 1, bytes([
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
        f.write("1 1 0 0 0 0 0 0 1 image.png\n")
        f.write("\n")

    with open(os.path.join(root, "sparse", "points3D.txt"), "w", encoding="utf-8") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("1 0 0 0 255 255 255 0 1 0\n")


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


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


def test_dataset_detects_transparent_png_alpha():
    from msplat import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_nerfstudio_dataset(tmp, alpha=True)
        ds = Dataset(tmp)

        assert ds.camera_has_alpha(0) is True


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


# ── Training tests ───────────────────────────────────────────────────────────


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
