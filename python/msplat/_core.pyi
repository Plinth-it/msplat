"""Type stubs for msplat._core (compiled nanobind extension)."""

import numpy as np
from numpy.typing import NDArray

class TrainingConfig:
    iterations: int
    sh_degree: int
    sh_degree_interval: int
    ssim_weight: float
    num_downscales: int
    resolution_schedule: int
    refine_every: int
    warmup_length: int
    reset_alpha_every: int
    densify_grad_thresh: float
    densify_size_thresh: float
    stop_screen_size_at: int
    growth_stop_iter: int
    max_splats: int
    growth_select_fraction: float
    split_screen_size: float
    match_alpha_weight: float
    lpips_loss_weight: float
    opac_decay: float
    scale_decay: float
    mean_noise_weight: float
    lr_mean: float
    lr_mean_end: float
    lr_scale: float
    lr_scale_end: float
    lr_rotation: float
    lr_coeffs_dc: float
    lr_coeffs_sh_scale: float
    lr_opac: float
    random_init_scene_scale: float
    reduce_second_moment: bool
    keep_crs: bool
    render_mip: bool
    downscale_factor: float
    output: str
    save_every: int
    bg_color: list[float]
    """Background color as [R, G, B] floats in [0, 1]. Default black [0, 0, 0]."""
    background_noise_strength: float

    def __init__(
        self,
        iterations: int = 30000,
        sh_degree: int = 3,
        sh_degree_interval: int = 1,
        ssim_weight: float = 0.2,
        num_downscales: int = 0,
        resolution_schedule: int = 3000,
        refine_every: int = 200,
        warmup_length: int = 0,
        reset_alpha_every: int = 0,
        densify_grad_thresh: float = 0.0020,
        densify_size_thresh: float = 0.01,
        stop_screen_size_at: int = 15000,
        split_screen_size: float = 0.25,
        keep_crs: bool = False,
        render_mip: bool = False,
        downscale_factor: float = 1.0,
        output: str = "splat.ply",
        save_every: int = -1,
        bg_color: list[float] = ...,
        match_alpha_weight: float = 0.1,
        background_noise_strength: float = 0.1,
        opac_decay: float = 0.004,
        scale_decay: float = 0.002,
        mean_noise_weight: float = 50.0,
        growth_stop_iter: int = 15000,
        max_splats: int = 10000000,
        growth_select_fraction: float = 0.25,
        lr_mean: float = 0.00002,
        lr_mean_end: float = 0.0000002,
        lr_scale: float = 0.007,
        lr_scale_end: float = 0.005,
        lr_rotation: float = 0.002,
        lr_coeffs_dc: float = 0.002,
        lr_coeffs_sh_scale: float = 10.0,
        lr_opac: float = 0.012,
        lpips_loss_weight: float = 0.0,
        random_init_scene_scale: float = 0.0,
        reduce_second_moment: bool = False,
    ) -> None: ...

class TrainingStats:
    """Per-step training statistics returned by GaussianTrainer.step()."""

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Number of active Gaussians."""
        ...

    @property
    def ms_per_step(self) -> float:
        """Wall-clock time for this step in milliseconds."""
        ...

class Dataset:
    """A loaded dataset of camera images. Auto-detects COLMAP, Nerfstudio, and Polycam formats."""

    def __init__(
        self,
        path: str,
        downscale_factor: float = 1.0,
        eval_mode: bool = False,
        test_every: int = 8,
    ) -> None: ...

    @property
    def num_train(self) -> int:
        """Number of training cameras."""
        ...

    @property
    def num_test(self) -> int:
        """Number of test cameras (0 unless eval_mode=True)."""
        ...

    @property
    def initial_point_count(self) -> int:
        """Number of finite point-cloud points loaded for initialization."""
        ...

    def camera_pose(self, index: int) -> NDArray[np.float32]:
        """Get camera-to-world pose (4x4 row-major, OpenGL convention) as numpy array."""
        ...

    def camera_has_alpha(self, index: int) -> bool:
        """Return true when the loaded training image has transparent pixels."""
        ...

    def camera_has_mask(self, index: int) -> bool:
        """Return true when the dataset provides an explicit mask image."""
        ...

class GaussianTrainer:
    """3D Gaussian Splatting trainer. All computation runs on the Metal GPU."""

    def __init__(self, dataset: Dataset, config: TrainingConfig) -> None: ...

    def step(
        self,
        forced_downscale: int = 0,
        apply_refine: bool = True,
    ) -> TrainingStats:
        """Run a single training iteration. Returns TrainingStats."""
        ...

    def train(
        self,
        callback: object,
        callback_every: int = 100,
    ) -> None:
        """Run training to completion, calling callback(stats) every callback_every steps."""
        ...

    def evaluate(self) -> dict[str, float | int]:
        """Evaluate on held-out test cameras. Returns dict with psnr, ssim, l1 keys.

        Requires the dataset to have been loaded with eval_mode=True.
        """
        ...

    def render(
        self,
        cam_idx: int,
        use_test: bool = False,
    ) -> NDArray[np.float32]:
        """Render a camera view. Returns a numpy array of shape (H, W, 3), float32, RGB [0,1]."""
        ...

    def render_from_pose(
        self,
        cam_to_world: NDArray[np.float32],
        ref_cam_idx: int = 0,
    ) -> NDArray[np.float32]:
        """Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).

        Uses intrinsics from ref_cam_idx. Returns numpy (H, W, 3) float32.
        """
        ...

    def export_ply(self, path: str) -> None:
        """Export the current Gaussians as a PLY file."""
        ...

    def export_lod_ply(self, path: str, target_count: int) -> None:
        """Export an importance-ranked LOD PLY with at most target_count Gaussians."""
        ...

    def decimate_to_lod(self, target_count: int) -> None:
        """Decimate the active model in memory to at most target_count Gaussians."""
        ...

    def export_splat(self, path: str) -> None:
        """Export the current Gaussians as a .splat file."""
        ...

    def load_ply(self, path: str) -> int:
        """Load Gaussians from a trained PLY file and resume from its saved iteration."""
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint."""
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and resume from the saved iteration."""
        ...

    @property
    def splat_count(self) -> int:
        """Current number of active Gaussians."""
        ...

    @property
    def iteration(self) -> int:
        """Current training iteration."""
        ...

def sync() -> None:
    """Synchronize GPU (wait for all commands to complete)."""
    ...

def cleanup() -> None:
    """Release all cached GPU resources."""
    ...

def _set_metallib_path(path: str) -> None:
    """Set the default.metallib resource path."""
    ...

def _set_lpips_weights_path(path: str) -> None:
    """Set the LPIPS weights resource path."""
    ...
