#!/usr/bin/env python3
"""Run throughput and stage-profile A/Bs for backward rasterizer modes."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare msplat backward rasterizer modes."
    )
    parser.add_argument("dataset", type=Path, help="Dataset path accepted by ./build/msplat")
    parser.add_argument("--binary", type=Path, default=Path("build/msplat"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--debug-interval", type=int)
    parser.add_argument(
        "--profile-stages",
        dest="profile_stages",
        action="store_true",
        default=False,
        help="Enable synchronized per-stage GPU profiling.",
    )
    parser.add_argument(
        "--no-profile-stages",
        dest="profile_stages",
        action="store_false",
        help="Disable per-stage profiling for production throughput measurements (default).",
    )
    parser.add_argument(
        "--stage-report-every",
        type=int,
        help="PROFILE_STAGES_REPORT_EVERY value when stage profiling is enabled.",
    )
    parser.add_argument(
        "--roofline-dimensions",
        action="store_true",
        help="Print periodic roofline dimension diagnostics. Stage profiling enables this automatically.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["wall-only", "drain-each-iter", "async-submit", "drain-every-n"],
        default="wall-only",
        help="Benchmark wall-clock production throughput, per-iteration GPU drains, async submit, or bounded async submit.",
    )
    parser.add_argument(
        "--drain-interval",
        type=int,
        default=4,
        help="Iteration interval for --timing-mode drain-every-n.",
    )
    parser.add_argument(
        "--pre-refine-drain",
        action="store_true",
        help="Drain at the end of the iteration before refine to diagnose queued GPU backlog.",
    )
    parser.add_argument(
        "--overflow-poll-interval",
        type=int,
        default=100,
        help="MSPLAT_OVERFLOW_POLL_INTERVAL value; 0 disables periodic overflow polling for diagnostic A/Bs.",
    )
    parser.add_argument(
        "--refine-flag-mode",
        choices=["brush", "gpu", "both"],
        default="brush",
        help="A/B CPU Brush-style refine flag preparation against GPU-native refine classification.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Enable backward raster debug counters.",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable backward raster debug counters for production throughput measurements (default).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["auto", "pixel", "persplat"],
        choices=["auto", "pixel", "perpixel", "chunked", "persplat", "brush"],
    )
    parser.add_argument(
        "--project-sh-specialization",
        choices=["off", "on", "both"],
        default="off",
        help="A/B the opt-in MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION hook.",
    )
    parser.add_argument(
        "--loss-specialization",
        choices=["off", "on", "both"],
        default="off",
        help="A/B the opt-in MSPLAT_ENABLE_LOSS_SPECIALIZATION hook.",
    )
    parser.add_argument(
        "--raster-backward-specialization",
        choices=["off", "on", "both"],
        default="off",
        help="A/B the opt-in MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION hook.",
    )
    parser.add_argument(
        "--half-sorted-buffers",
        choices=["off", "on", "both"],
        default="off",
        help="A/B the opt-in MSPLAT_HALF_SORTED_BUFFERS hook.",
    )
    parser.add_argument(
        "--warp-merge",
        choices=["off", "on", "both"],
        default="off",
        help="A/B the opt-in MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE hook.",
    )
    parser.add_argument(
        "--intersection-key-bits",
        nargs="+",
        default=["default"],
        choices=["default", "64", "32", "auto"],
        help="Run with one or more MSPLAT_INTERSECTION_KEY_BITS modes; default leaves the env unset.",
    )
    parser.add_argument(
        "--quality-metrics",
        "--final-quality",
        dest="quality_metrics",
        action="store_true",
        default=False,
        help="Pass --final-quality to msplat so the table includes final train metrics.",
    )
    parser.add_argument(
        "--no-quality-metrics",
        "--no-final-quality",
        dest="quality_metrics",
        action="store_false",
        help="Do not request final train metrics from msplat (default).",
    )
    args, extra_args = parser.parse_known_args()
    args.msplat_args = clean_extra_args(extra_args)
    if args.stage_report_every is not None:
        args.profile_stages = True
    if args.drain_interval <= 0:
        parser.error("--drain-interval must be positive")
    if args.overflow_poll_interval < 0:
        parser.error("--overflow-poll-interval must be non-negative")
    return args


def clean_extra_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, re.MULTILINE)
    return float(matches[-1]) if matches else None


def parse_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_duration(text: str) -> float | None:
    if match := re.fullmatch(r"([0-9.]+) s", text):
        return float(match.group(1))
    if match := re.fullmatch(r"([0-9]+)m ([0-9.]+)s", text):
        return int(match.group(1)) * 60.0 + float(match.group(2))
    return None


def parse_phase_stat(log_text: str, phase: str, stat: str) -> float | None:
    match = re.search(
        rf"^\s*{re.escape(phase)}:\s+mean=([0-9.e+-]+)\s+median=([0-9.e+-]+)\s+p95=([0-9.e+-]+)\s+max=([0-9.e+-]+) ms",
        log_text,
        re.MULTILINE,
    )
    if not match:
        return None
    field_index = {"mean": 1, "median": 2, "p95": 3, "max": 4}[stat]
    return float(match.group(field_index))


def parse_forced_sync_reasons(log_text: str) -> dict[str, int]:
    section = re.search(
        r"^\s*forced sync reasons:\n((?:^\s{4}.+\n?)+)",
        log_text,
        re.MULTILINE,
    )
    if not section:
        return {}
    reasons: dict[str, int] = {}
    for reason, count in re.findall(r"^\s{4}([^:]+):\s+([0-9]+)", section.group(1), re.MULTILINE):
        reasons[reason] = int(count)
    return reasons


def parse_metrics(log_text: str) -> dict[str, object]:
    benchmark = re.search(
        r"=== Benchmark .*?mean:\s+([0-9.]+) ms/iter\s+median:\s+([0-9.]+) ms/iter",
        log_text,
        re.DOTALL,
    )
    wall_only_benchmark = re.search(
        r"=== Benchmark .*?wall mean:\s+([0-9.]+) ms/iter",
        log_text,
        re.DOTALL,
    )
    training = re.search(
        r"training loop:\s+(.+?)\s+\(([0-9]+) steps,\s+([0-9.]+) it/s\)",
        log_text,
    )
    tile_ranges = re.search(
        r"tile splats: avg ([0-9.]+) -> ([0-9.]+), median ([0-9.]+) -> ([0-9.]+), max ([0-9.]+) -> ([0-9.]+)",
        log_text,
    )
    replay = re.search(
        r"persplat replay estimate: active_pairs ([0-9.]+)M/sample, diagonal_steps ([0-9.]+)M/sample, tightened_skip ([0-9.]+)M/sample",
        log_text,
    )
    atomics = re.search(
        r"pixel atomic estimate: warp_groups ([0-9.]+)M/sample, tile_merge_floor ([0-9.]+)M/sample, max_reduction ([0-9.]+)%",
        log_text,
    )
    saturated_pixels = re.search(r"saturated pixels: ([0-9.]+)/sample", log_text)

    training_seconds = parse_duration(training.group(1).strip()) if training else None
    training_steps = int(training.group(2)) if training else None
    metrics: dict[str, object] = {
        "iter_mean_ms": float(benchmark.group(1)) if benchmark else (float(wall_only_benchmark.group(1)) if wall_only_benchmark else None),
        "iter_median_ms": float(benchmark.group(2)) if benchmark else None,
        "training_seconds": training_seconds,
        "training_steps": training_steps,
        "training_ips": float(training.group(3)) if training else None,
        "forced_syncs": parse_int(r"forced syncs:\s+([0-9]+)", log_text),
        "forced_sync_reasons": parse_forced_sync_reasons(log_text),
        "cpu_prepare_p95_ms": parse_phase_stat(log_text, "prepare", "p95"),
        "cpu_prepare_max_ms": parse_phase_stat(log_text, "prepare", "max"),
        "cpu_full_iteration_p95_ms": parse_phase_stat(log_text, "full_iteration", "p95"),
        "cpu_full_iteration_max_ms": parse_phase_stat(log_text, "full_iteration", "max"),
        "cpu_schedulers_p95_ms": parse_phase_stat(log_text, "schedulers", "p95"),
        "cpu_schedulers_max_ms": parse_phase_stat(log_text, "schedulers", "max"),
        "cpu_after_train_p95_ms": parse_phase_stat(log_text, "after_train", "p95"),
        "cpu_after_train_max_ms": parse_phase_stat(log_text, "after_train", "max"),
        "cpu_commit_p95_ms": parse_phase_stat(log_text, "commit", "p95"),
        "cpu_commit_max_ms": parse_phase_stat(log_text, "commit", "max"),
        "cpu_pre_refine_drain_p95_ms": parse_phase_stat(log_text, "pre_refine_drain", "p95"),
        "cpu_pre_refine_drain_max_ms": parse_phase_stat(log_text, "pre_refine_drain", "max"),
        "refine_prepare_flags_mean_ms": parse_phase_stat(log_text, "refine_prepare_flags", "mean"),
        "refine_prepare_flags_max_ms": parse_phase_stat(log_text, "refine_prepare_flags", "max"),
        "densify_count_readback_mean_ms": parse_phase_stat(log_text, "densify_count_readback", "mean"),
        "densify_count_readback_max_ms": parse_phase_stat(log_text, "densify_count_readback", "max"),
        "flag_sync_readback_mean_ms": parse_phase_stat(log_text, "flag_sync_readback", "mean"),
        "flag_sync_readback_max_ms": parse_phase_stat(log_text, "flag_sync_readback", "max"),
        "gpu_stage_total_median_ms": parse_float(r"^\s*TOTAL \(sum medians\)\s+([0-9.]+)ms", log_text),
        "loss_fwd_bwd_median_ms": parse_float(r"^\s*loss_fwd_bwd\s+median=([0-9.]+)ms", log_text),
        "rast_bwd_median_ms": parse_float(r"^\s*rast_bwd\s+median=([0-9.]+)ms", log_text),
        "rast_bwd_mean_ms": parse_float(r"^\s*rast_bwd\s+median=[0-9.]+ms\s+mean=([0-9.]+)ms", log_text),
        "proj_sh_bwd_adam_median_ms": parse_float(r"^\s*proj_sh_bwd_adam\s+median=([0-9.]+)ms", log_text),
        "psnr": parse_float(r"train PSNR:\s+([0-9.]+)", log_text),
        "ssim": parse_float(r"train SSIM:\s+([0-9.]+)", log_text),
        "l1": parse_float(r"train L1:\s+([0-9.]+)", log_text),
        "splats": parse_int(r"Progress:\s+100\.0%.*?\s+([0-9]+)\s+gaussians", log_text),
        "tile_avg_before": float(tile_ranges.group(1)) if tile_ranges else None,
        "tile_avg_after": float(tile_ranges.group(2)) if tile_ranges else None,
        "tile_median_before": float(tile_ranges.group(3)) if tile_ranges else None,
        "tile_median_after": float(tile_ranges.group(4)) if tile_ranges else None,
        "pixel_atomic_groups_m": float(atomics.group(1)) if atomics else None,
        "tile_merge_floor_m": float(atomics.group(2)) if atomics else None,
        "tile_merge_max_reduction": float(atomics.group(3)) if atomics else None,
        "replay_active_m": float(replay.group(1)) if replay else None,
        "replay_diagonal_m": float(replay.group(2)) if replay else None,
        "replay_skip_m": float(replay.group(3)) if replay else None,
        "saturated_pixels": float(saturated_pixels.group(1)) if saturated_pixels else None,
    }
    metrics["wall_mean_ms"] = (
        training_seconds * 1000.0 / training_steps
        if isinstance(training_seconds, float) and isinstance(training_steps, int) and training_steps > 0
        else None
    )
    return metrics


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_delta(value: object, baseline: object, digits: int = 1, percent: bool = True) -> str:
    if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
        return "-"
    if percent:
        if baseline == 0:
            return "-"
        delta = 100.0 * (float(value) - float(baseline)) / abs(float(baseline))
        return f"{delta:+.{digits}f}%"
    delta = float(value) - float(baseline)
    return f"{delta:+.{digits}f}"


def numeric_delta(value: object, baseline: object) -> float | None:
    if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
        return None
    return float(value) - float(baseline)


def relative_delta(value: object, baseline: object) -> float | None:
    if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
        return None
    if baseline == 0:
        return None
    return (float(value) - float(baseline)) / abs(float(baseline))


def run_mode(
    args: argparse.Namespace,
    mode: str,
    project_sh_specialization: bool,
    loss_specialization: bool,
    raster_specialization: bool,
    half_sorted_buffers: bool,
    warp_merge: bool,
    intersection_key_bits: str,
    refine_flag_mode: str,
    output_dir: Path,
    extra_args: list[str],
) -> dict[str, object]:
    run_name = mode
    if project_sh_specialization:
        run_name += "-project-sh-spec"
    if loss_specialization:
        run_name += "-loss-spec"
    if raster_specialization:
        run_name += "-rb-spec"
    if half_sorted_buffers:
        run_name += "-half"
    if warp_merge:
        run_name += "-warp-merge"
    if intersection_key_bits != "default":
        run_name += f"-key-{intersection_key_bits}"
    if refine_flag_mode == "gpu":
        run_name += "-gpu-refine"
    mode_dir = output_dir / run_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    log_path = mode_dir / "run.log"
    output_path = mode_dir / "out.ply"
    quality_args = []
    if args.quality_metrics and not has_final_quality_arg(extra_args):
        quality_args.append("--final-quality")

    env = os.environ.copy()
    env["BENCHMARK"] = "1"
    env["MSPLAT_BENCHMARK_TIMING_MODE"] = args.timing_mode
    env["MSPLAT_BENCHMARK_DRAIN_INTERVAL"] = str(args.drain_interval)
    env["MSPLAT_OVERFLOW_POLL_INTERVAL"] = str(args.overflow_poll_interval)
    if args.pre_refine_drain:
        env["MSPLAT_BENCHMARK_PRE_REFINE_DRAIN"] = "1"
    else:
        env.pop("MSPLAT_BENCHMARK_PRE_REFINE_DRAIN", None)
    if args.profile_stages:
        env["PROFILE_STAGES"] = "1"
        env["MSPLAT_PRINT_ROOFLINE"] = "1"
        if args.stage_report_every:
            env["PROFILE_STAGES_REPORT_EVERY"] = str(args.stage_report_every)
    else:
        env.pop("PROFILE_STAGES", None)
        env.pop("PROFILE_STAGES_REPORT_EVERY", None)
        if args.roofline_dimensions:
            env["MSPLAT_PRINT_ROOFLINE"] = "1"
        else:
            env.pop("MSPLAT_PRINT_ROOFLINE", None)
    if args.debug:
        env["MSPLAT_BACKWARD_DEBUG"] = "1"
        env["MSPLAT_BACKWARD_DEBUG_INTERVAL"] = str(args.debug_interval or args.iters)
    else:
        env.pop("MSPLAT_BACKWARD_DEBUG", None)
        env.pop("MSPLAT_BACKWARD_DEBUG_INTERVAL", None)
    if project_sh_specialization:
        env["MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION"] = "1"
    else:
        env.pop("MSPLAT_ENABLE_PROJECT_SH_SPECIALIZATION", None)
    if loss_specialization:
        env["MSPLAT_ENABLE_LOSS_SPECIALIZATION"] = "1"
    else:
        env.pop("MSPLAT_ENABLE_LOSS_SPECIALIZATION", None)
    if raster_specialization:
        env["MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION"] = "1"
    else:
        env.pop("MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION", None)
    if half_sorted_buffers:
        env["MSPLAT_HALF_SORTED_BUFFERS"] = "1"
    else:
        env.pop("MSPLAT_HALF_SORTED_BUFFERS", None)
    if warp_merge:
        env["MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE"] = "1"
    else:
        env.pop("MSPLAT_ENABLE_RASTER_BACKWARD_WARP_MERGE", None)
    if intersection_key_bits == "default":
        env.pop("MSPLAT_INTERSECTION_KEY_BITS", None)
    else:
        env["MSPLAT_INTERSECTION_KEY_BITS"] = intersection_key_bits
    if refine_flag_mode == "gpu":
        env["MSPLAT_REFINE_FLAG_MODE"] = "gpu"
    else:
        env.pop("MSPLAT_REFINE_FLAG_MODE", None)

    cmd = [
        str(args.binary),
        str(args.dataset),
        "--output",
        str(output_path),
        "--total-train-iters",
        str(args.iters),
        "--backward-rasterizer",
        mode,
        *quality_args,
        *extra_args,
    ]
    labels = []
    if project_sh_specialization:
        labels.append("project/SH specialization")
    if loss_specialization:
        labels.append("loss specialization")
    if raster_specialization:
        labels.append("raster-backward specialization")
    if half_sorted_buffers:
        labels.append("half sorted buffers")
    if warp_merge:
        labels.append("warp-merge atomics")
    if intersection_key_bits != "default":
        labels.append(f"{intersection_key_bits}-bit intersection keys" if intersection_key_bits != "auto" else "auto intersection keys")
    if refine_flag_mode == "gpu":
        labels.append("GPU refine flags")
    suffix = f" + {' + '.join(labels)}" if labels else ""
    print(f"\n=== Running {mode}{suffix} ===", flush=True)
    process = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=1,
    )
    output_chunks: list[str] = []
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(1)
        if chunk == "":
            break
        output_chunks.append(chunk)
        print(chunk, end="", flush=True)
    returncode = process.wait()
    log_text = "".join(output_chunks)
    log_path.write_text(log_text, encoding="utf-8")
    if returncode != 0:
        raise SystemExit(returncode)

    metrics = parse_metrics(log_text)
    metrics["mode"] = run_name
    metrics["log"] = log_path
    return metrics


def project_sh_specialization_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def loss_specialization_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def raster_specialization_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def half_sorted_buffer_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def warp_merge_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def intersection_key_bit_variants(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def refine_flag_mode_variants(value: str) -> list[str]:
    if value == "both":
        return ["brush", "gpu"]
    return [value]


def has_final_quality_arg(args: list[str]) -> bool:
    return any(arg in {"--final-quality", "--no-final-quality"} for arg in args)


def print_table(results: list[dict[str, object]], output_dir: Path) -> None:
    print(f"Logs: {output_dir}")
    print()
    print("| mode | train it/s | train s | wall mean ms | iter median ms | GPU stage total ms | loss median ms | rast_bwd median ms | proj/SH/Adam median ms | PSNR | SSIM | L1 | splats | tile avg | pixel atomic M | merge ceiling % | replay active M | replay diag M | replay skip M | sat px |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        tile_avg = "-"
        if row["tile_avg_before"] is not None and row["tile_avg_after"] is not None:
            tile_avg = f"{fmt(row['tile_avg_before'], 1)}->{fmt(row['tile_avg_after'], 1)}"
        print(
            "| {mode} | {ips} | {train_s} | {wall_mean_ms} | {iter_ms} | {gpu_total_ms} | {loss_ms} | {bwd_ms} | {proj_adam_ms} | {psnr} | {ssim} | {l1} | {splats} | {tile_avg} | {atomic_groups} | {merge_ceiling} | {replay} | {replay_diag} | {replay_skip} | {sat_px} |".format(
                mode=row["mode"],
                ips=fmt(row["training_ips"], 2),
                train_s=fmt(row["training_seconds"], 2),
                wall_mean_ms=fmt(row["wall_mean_ms"]),
                iter_ms=fmt(row["iter_median_ms"]),
                gpu_total_ms=fmt(row["gpu_stage_total_median_ms"]),
                loss_ms=fmt(row["loss_fwd_bwd_median_ms"]),
                bwd_ms=fmt(row["rast_bwd_median_ms"]),
                proj_adam_ms=fmt(row["proj_sh_bwd_adam_median_ms"]),
                psnr=fmt(row["psnr"], 2),
                ssim=fmt(row["ssim"], 4),
                l1=fmt(row["l1"], 5),
                splats=fmt(row["splats"], 0),
                tile_avg=tile_avg,
                atomic_groups=fmt(row["pixel_atomic_groups_m"], 1),
                merge_ceiling=fmt(row["tile_merge_max_reduction"], 1),
                replay=fmt(row["replay_active_m"], 1),
                replay_diag=fmt(row["replay_diagonal_m"], 1),
                replay_skip=fmt(row["replay_skip_m"], 1),
                sat_px=fmt(row["saturated_pixels"], 1),
            )
        )
    print_sync_diagnostics(results)
    print_baseline_deltas(results)
    print_quality_warnings(results)


def forced_sync_reason_count(row: dict[str, object], reason: str) -> int | None:
    reasons = row.get("forced_sync_reasons")
    if not isinstance(reasons, dict):
        return None
    count = reasons.get(reason)
    return int(count) if isinstance(count, int) else None


def print_sync_diagnostics(results: list[dict[str, object]]) -> None:
    if not any(
        row.get("forced_syncs")
        or row.get("flag_sync_readback_max_ms") is not None
        or row.get("densify_count_readback_max_ms") is not None
        or row.get("cpu_pre_refine_drain_max_ms") is not None
        for row in results
    ):
        return

    print()
    print("Sync diagnostics:")
    print("| mode | forced syncs | flag sync max ms | densify count max ms | pre-drain max ms | after_train max ms | full_iteration max ms | overflow checks | flag readbacks | densify readbacks | pre-drains |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            "| {mode} | {forced} | {flag_sync} | {densify_count} | {pre_drain} | {after_train} | {full_iter} | {overflow} | {flag_reads} | {densify_reads} | {pre_drains} |".format(
                mode=row["mode"],
                forced=fmt(row.get("forced_syncs"), 0),
                flag_sync=fmt(row.get("flag_sync_readback_max_ms")),
                densify_count=fmt(row.get("densify_count_readback_max_ms")),
                pre_drain=fmt(row.get("cpu_pre_refine_drain_max_ms")),
                after_train=fmt(row.get("cpu_after_train_max_ms")),
                full_iter=fmt(row.get("cpu_full_iteration_max_ms")),
                overflow=fmt(forced_sync_reason_count(row, "overflow-check"), 0),
                flag_reads=fmt(forced_sync_reason_count(row, "refine-flags-readback"), 0),
                densify_reads=fmt(forced_sync_reason_count(row, "densify-count-readback"), 0),
                pre_drains=fmt(forced_sync_reason_count(row, "pre-refine-drain"), 0),
            )
        )


def print_baseline_deltas(results: list[dict[str, object]]) -> None:
    if len(results) < 2:
        return
    baseline = results[0]
    print()
    print(f"Baseline deltas vs {baseline['mode']}:")
    print("| mode | train it/s | iter median | rast_bwd median | PSNR | SSIM | L1 | splats |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results[1:]:
        print(
            "| {mode} | {ips} | {iter_ms} | {bwd_ms} | {psnr} | {ssim} | {l1} | {splats} |".format(
                mode=row["mode"],
                ips=fmt_delta(row["training_ips"], baseline["training_ips"]),
                iter_ms=fmt_delta(row["iter_median_ms"], baseline["iter_median_ms"]),
                bwd_ms=fmt_delta(row["rast_bwd_median_ms"], baseline["rast_bwd_median_ms"]),
                psnr=fmt_delta(row["psnr"], baseline["psnr"], digits=2, percent=False),
                ssim=fmt_delta(row["ssim"], baseline["ssim"], digits=4, percent=False),
                l1=fmt_delta(row["l1"], baseline["l1"]),
                splats=fmt_delta(row["splats"], baseline["splats"]),
            )
        )


def quality_drift_reasons(row: dict[str, object], baseline: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    psnr_delta = numeric_delta(row["psnr"], baseline["psnr"])
    if psnr_delta is not None and abs(psnr_delta) >= 0.25:
        reasons.append(f"PSNR {psnr_delta:+.2f} dB")
    ssim_delta = numeric_delta(row["ssim"], baseline["ssim"])
    if ssim_delta is not None and abs(ssim_delta) >= 0.01:
        reasons.append(f"SSIM {ssim_delta:+.4f}")
    l1_delta = relative_delta(row["l1"], baseline["l1"])
    if l1_delta is not None and abs(l1_delta) >= 0.05:
        reasons.append(f"L1 {l1_delta * 100.0:+.1f}%")
    splat_delta = relative_delta(row["splats"], baseline["splats"])
    if splat_delta is not None and abs(splat_delta) >= 0.05:
        reasons.append(f"splats {splat_delta * 100.0:+.1f}%")
    return reasons


def print_quality_warnings(results: list[dict[str, object]]) -> None:
    if len(results) < 2:
        return
    baseline = results[0]
    warnings = quality_warning_rows(results)
    if not warnings:
        return

    print()
    print(f"Large quality/count drift vs {baseline['mode']}:")
    for warning in warnings:
        print(f"  {warning['mode']}: {', '.join(warning['reasons'])}")


def quality_warning_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(results) < 2:
        return []
    baseline = results[0]
    rows: list[dict[str, object]] = []
    for row in results[1:]:
        reasons = quality_drift_reasons(row, baseline)
        if reasons:
            rows.append({"mode": row["mode"], "reasons": reasons})
    return rows


def json_result(row: dict[str, object]) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in row.items()
    }


def write_summary(args: argparse.Namespace, results: list[dict[str, object]], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "binary": str(args.binary),
        "output_dir": str(output_dir),
        "iters": args.iters,
        "profile_stages": args.profile_stages,
        "profile_mode": "stage" if args.profile_stages else "production",
        "timing_mode": args.timing_mode,
        "drain_interval": args.drain_interval,
        "overflow_poll_interval": args.overflow_poll_interval,
        "pre_refine_drain": args.pre_refine_drain,
        "debug": args.debug,
        "quality_metrics": args.quality_metrics,
        "project_sh_specialization": args.project_sh_specialization,
        "loss_specialization": args.loss_specialization,
        "raster_backward_specialization": args.raster_backward_specialization,
        "half_sorted_buffers": args.half_sorted_buffers,
        "warp_merge": args.warp_merge,
        "intersection_key_bits": args.intersection_key_bits,
        "refine_flag_mode": args.refine_flag_mode,
        "msplat_args": args.msplat_args,
        "results": [json_result(row) for row in results],
        "quality_warnings": quality_warning_rows(results),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")
    return summary_path


def main() -> int:
    args = parse_args()
    extra_args = args.msplat_args
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="msplat_backward_ab."))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_mode(
            args,
            mode,
            project_sh_specialization,
            loss_specialization,
            raster_specialization,
            half_sorted_buffers,
            warp_merge,
            intersection_key_bits,
            refine_flag_mode,
            output_dir,
            extra_args,
        )
        for mode in args.modes
        for project_sh_specialization in project_sh_specialization_variants(args.project_sh_specialization)
        for loss_specialization in loss_specialization_variants(args.loss_specialization)
        for raster_specialization in raster_specialization_variants(args.raster_backward_specialization)
        for half_sorted_buffers in half_sorted_buffer_variants(args.half_sorted_buffers)
        for warp_merge in warp_merge_variants(args.warp_merge)
        for intersection_key_bits in intersection_key_bit_variants(args.intersection_key_bits)
        for refine_flag_mode in refine_flag_mode_variants(args.refine_flag_mode)
    ]
    print_table(results, output_dir)
    write_summary(args, results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
