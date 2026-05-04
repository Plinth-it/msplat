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


def _write_minimal_nerfstudio_dataset(root, alpha, mask=False):
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
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.refine_every == 100
    assert cfg.warmup_length == 500


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0


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
    finally:
        os.unlink(path)


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
