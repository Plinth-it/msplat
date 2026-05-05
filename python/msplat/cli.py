"""msplat-train CLI entry point."""

import sys
from pathlib import Path


def main():
    try:
        import tyro
    except ImportError:
        print("Install tyro for CLI support: pip install msplat[cli]", file=sys.stderr)
        sys.exit(1)

    from dataclasses import dataclass, field

    @dataclass
    class Args:
        """Train 3D Gaussian Splatting on a dataset."""

        input: str
        """Path to dataset (COLMAP, Nerfstudio, etc.)"""

        output: str = "splat.ply"
        """Output PLY file path"""

        num_iters: int = 30000
        """Number of training iterations"""

        downscale_factor: float = 1.0
        """Image downscale factor"""

        num_downscales: int = 2
        """Number of progressive downscales"""

        resolution_schedule: int = 3000
        """Double resolution every N steps"""

        sh_degree: int = 3
        """Max spherical harmonics degree"""

        sh_degree_interval: int = 1
        """Steps between SH degree increases"""

        ssim_weight: float = 0.2
        """SSIM loss weight"""

        refine_every: int = 100
        """Densification interval"""

        warmup_length: int = 500
        """Steps before densification starts"""

        reset_alpha_every: int = 30
        """Reset opacity every N refinements"""

        densify_grad_thresh: float = 0.008
        """Gradient threshold for densification"""

        densify_size_thresh: float = 0.01
        """Size threshold for split vs clone"""

        stop_screen_size_at: int = 4000
        """Stop screen-size split after this step"""

        growth_stop_iter: int = 15000
        """Stop splat growth after this iteration"""

        max_splats: int = 10000000
        """Maximum splat count allowed during growth"""

        growth_select_fraction: float = 0.25
        """Fraction of high-gradient splats selected for growth"""

        split_screen_size: float = 0.05
        """Screen-space split threshold"""

        match_alpha_weight: float = 0.1
        """Alpha L1 loss weight for transparent targets"""

        lpips_loss_weight: float = 0.0
        """LPIPS perceptual loss weight"""

        opac_decay: float = 0.004
        """Opacity shrink applied at refinement steps"""

        scale_decay: float = 0.002
        """Scale shrink applied at refinement steps"""

        mean_noise_weight: float = 50.0
        """Low-opacity mean noise weight during growth"""

        lr_mean: float = 0.00256
        """Initial learning rate for mean parameters"""

        lr_mean_end: float = 0.0000256
        """Final learning rate for mean parameters"""

        lr_scale: float = 0.022
        """Initial learning rate for scale parameters"""

        lr_scale_end: float = 0.022
        """Final learning rate for scale parameters"""

        lr_rotation: float = 0.002
        """Learning rate for rotation parameters"""

        lr_coeffs_dc: float = 0.012
        """Learning rate for base SH coefficients"""

        lr_coeffs_sh_scale: float = 10.0
        """Divisor for higher-order SH coefficient learning rate"""

        lr_opac: float = 0.035
        """Learning rate for opacity parameters"""

        random_init_scene_scale: float = 0.0
        """Scene scale for random init when no point cloud exists; 0 estimates from cameras"""

        reduce_second_moment: bool = False
        """Use Brush-style scalar second moment for SH Adam updates"""

        keep_crs: bool = False
        """Keep input coordinate reference system"""

        render_mip: bool = False
        """Use MIP splatting opacity compensation during training and rendering"""

        save_every: int = -1
        """Save every N steps (-1 to disable)"""

        lod_levels: int = 0
        """Export N importance-ranked LOD PLY files after training"""

        lod_keep_ratio: float = 0.5
        """Fraction of splats to keep per LOD level"""

        lod_refine_steps: int = 0
        """Optimize each decimated LOD for N extra steps"""

        lod_image_scale: int = 50
        """Percentage to scale source images at each LOD refinement level"""

        eval: bool = False
        """Evaluate on held-out test views"""

        test_every: int = 8
        """Hold out every Nth image for eval"""

        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Background RGB used for rendering and transparent image compositing"""

        background_noise_strength: float = 0.1
        """Uniform background jitter strength per training step"""

    args = tyro.cli(Args)

    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync, cleanup

    config = TrainingConfig(
        iterations=args.num_iters,
        sh_degree=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        ssim_weight=args.ssim_weight,
        num_downscales=args.num_downscales,
        resolution_schedule=args.resolution_schedule,
        refine_every=args.refine_every,
        warmup_length=args.warmup_length,
        reset_alpha_every=args.reset_alpha_every,
        densify_grad_thresh=args.densify_grad_thresh,
        densify_size_thresh=args.densify_size_thresh,
        stop_screen_size_at=args.stop_screen_size_at,
        growth_stop_iter=args.growth_stop_iter,
        max_splats=args.max_splats,
        growth_select_fraction=args.growth_select_fraction,
        split_screen_size=args.split_screen_size,
        match_alpha_weight=args.match_alpha_weight,
        lpips_loss_weight=args.lpips_loss_weight,
        opac_decay=args.opac_decay,
        scale_decay=args.scale_decay,
        mean_noise_weight=args.mean_noise_weight,
        lr_mean=args.lr_mean,
        lr_mean_end=args.lr_mean_end,
        lr_scale=args.lr_scale,
        lr_scale_end=args.lr_scale_end,
        lr_rotation=args.lr_rotation,
        lr_coeffs_dc=args.lr_coeffs_dc,
        lr_coeffs_sh_scale=args.lr_coeffs_sh_scale,
        lr_opac=args.lr_opac,
        random_init_scene_scale=args.random_init_scene_scale,
        reduce_second_moment=args.reduce_second_moment,
        keep_crs=args.keep_crs,
        render_mip=args.render_mip,
        downscale_factor=args.downscale_factor,
        output=args.output,
        save_every=args.save_every,
        bg_color=list(args.bg_color),
        background_noise_strength=args.background_noise_strength,
    )

    dataset = Dataset(
        args.input,
        downscale_factor=args.downscale_factor,
        eval_mode=args.eval,
        test_every=args.test_every,
    )
    print(f"Loaded {dataset.num_train} train cameras", end="")
    if args.eval:
        print(f", {dataset.num_test} test cameras")
    else:
        print()

    trainer = GaussianTrainer(dataset, config)

    def on_step(stats):
        print(
            f"step={stats.iteration:>6}  "
            f"splats={stats.splat_count:>8,}  "
            f"ms={stats.ms_per_step:.1f}"
        )

    trainer.train(on_step, callback_every=100)

    trainer.export_ply(args.output)
    print(f"Saved {args.output}")
    if args.lod_levels > 0:
        output_path = Path(args.output)
        for level in range(1, args.lod_levels + 1):
            source_count = trainer.splat_count
            target_count = max(1, round(source_count * args.lod_keep_ratio))
            lod_path = output_path.with_name(f"{output_path.stem}_lod{level}.ply")
            if args.lod_refine_steps > 0:
                trainer.decimate_to_lod(target_count)
                cumulative_scale = (args.lod_image_scale / 100.0) ** level
                lod_downscale = max(1, round(1.0 / max(cumulative_scale, 0.01)))
                for _ in range(args.lod_refine_steps):
                    trainer.step(forced_downscale=lod_downscale, apply_refine=False)
                trainer.export_ply(str(lod_path))
            else:
                trainer.export_lod_ply(str(lod_path), target_count)
            print(f"Saved {lod_path} ({target_count:,} splats)")

    if args.eval:
        metrics = trainer.evaluate()
        print(f"\n=== Evaluation ({metrics['num_test']} test views) ===")
        print(f"  PSNR:  {metrics['psnr']:.4f}")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  L1:    {metrics['l1']:.4f}")
        print(f"  Gaussians: {metrics['num_gaussians']:,}")

    cleanup()


if __name__ == "__main__":
    main()
