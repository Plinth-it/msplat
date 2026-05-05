"""msplat: Metal-accelerated 3D Gaussian Splatting."""

import atexit
from pathlib import Path

from msplat._core import (
    TrainingConfig,
    TrainingStats,
    Dataset,
    GaussianTrainer,
    _set_lpips_weights_path,
    _set_metallib_path,
    sync,
    cleanup as _cleanup_raw,
)

_cleaned_up = False


def _configure_resources():
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[1] if len(package_dir.parents) > 1 else package_dir
    for metallib in (
        package_dir / "default.metallib",
        repo_root / "build" / "default.metallib",
    ):
        if metallib.exists():
            _set_metallib_path(str(metallib))
            break
    for lpips_weights in (
        package_dir / "lpips_vgg.bin",
        repo_root / "core" / "resources" / "lpips_vgg.bin",
    ):
        if lpips_weights.exists():
            _set_lpips_weights_path(str(lpips_weights))
            break


_configure_resources()


def cleanup():
    """Release all cached GPU resources. Safe to call multiple times."""
    global _cleaned_up
    if not _cleaned_up:
        _cleaned_up = True
        _cleanup_raw()


atexit.register(cleanup)

__all__ = [
    "TrainingConfig",
    "TrainingStats",
    "Dataset",
    "GaussianTrainer",
    "sync",
    "cleanup",
    "load_dataset",
]

__version__ = "1.1.3"


def load_dataset(
    path: str,
    downscale_factor: float = 1.0,
    eval_mode: bool = False,
    test_every: int = 8,
) -> Dataset:
    """Load a dataset (auto-detects COLMAP, Nerfstudio, Polycam)."""
    return Dataset(path, downscale_factor, eval_mode, test_every)
