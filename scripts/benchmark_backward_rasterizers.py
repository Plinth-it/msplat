#!/usr/bin/env python3
"""Run throughput and stage-profile A/Bs for backward rasterizer modes."""

from __future__ import annotations

import argparse
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
        default=True,
        help="Enable synchronized per-stage GPU profiling (default).",
    )
    parser.add_argument(
        "--no-profile-stages",
        dest="profile_stages",
        action="store_false",
        help="Disable per-stage profiling for production throughput measurements.",
    )
    parser.add_argument(
        "--stage-report-every",
        type=int,
        help="PROFILE_STAGES_REPORT_EVERY value when stage profiling is enabled.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=True,
        help="Enable backward raster debug counters (default).",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable backward raster debug counters for production throughput measurements.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["auto", "pixel", "persplat"],
        choices=["auto", "pixel", "perpixel", "chunked", "persplat", "brush"],
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
    return args


def clean_extra_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_duration(text: str) -> float | None:
    if match := re.fullmatch(r"([0-9.]+) s", text):
        return float(match.group(1))
    if match := re.fullmatch(r"([0-9]+)m ([0-9.]+)s", text):
        return int(match.group(1)) * 60.0 + float(match.group(2))
    return None


def parse_metrics(log_text: str) -> dict[str, object]:
    benchmark = re.search(
        r"=== Benchmark .*?mean:\s+([0-9.]+) ms/iter\s+median:\s+([0-9.]+) ms/iter",
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

    return {
        "iter_mean_ms": float(benchmark.group(1)) if benchmark else None,
        "iter_median_ms": float(benchmark.group(2)) if benchmark else None,
        "training_seconds": parse_duration(training.group(1).strip()) if training else None,
        "training_steps": int(training.group(2)) if training else None,
        "training_ips": float(training.group(3)) if training else None,
        "rast_bwd_median_ms": parse_float(r"^\s*rast_bwd\s+median=([0-9.]+)ms", log_text),
        "rast_bwd_mean_ms": parse_float(r"^\s*rast_bwd\s+median=[0-9.]+ms\s+mean=([0-9.]+)ms", log_text),
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


def run_mode(
    args: argparse.Namespace,
    mode: str,
    raster_specialization: bool,
    half_sorted_buffers: bool,
    intersection_key_bits: str,
    output_dir: Path,
    extra_args: list[str],
) -> dict[str, object]:
    run_name = mode
    if raster_specialization:
        run_name += "-rb-spec"
    if half_sorted_buffers:
        run_name += "-half"
    if intersection_key_bits != "default":
        run_name += f"-key-{intersection_key_bits}"
    mode_dir = output_dir / run_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    log_path = mode_dir / "run.log"
    output_path = mode_dir / "out.ply"
    quality_args = []
    if args.quality_metrics and not has_final_quality_arg(extra_args):
        quality_args.append("--final-quality")

    env = os.environ.copy()
    env["BENCHMARK"] = "1"
    if args.profile_stages:
        env["PROFILE_STAGES"] = "1"
        if args.stage_report_every:
            env["PROFILE_STAGES_REPORT_EVERY"] = str(args.stage_report_every)
    else:
        env.pop("PROFILE_STAGES", None)
        env.pop("PROFILE_STAGES_REPORT_EVERY", None)
    if args.debug:
        env["MSPLAT_BACKWARD_DEBUG"] = "1"
        env["MSPLAT_BACKWARD_DEBUG_INTERVAL"] = str(args.debug_interval or args.iters)
    else:
        env.pop("MSPLAT_BACKWARD_DEBUG", None)
        env.pop("MSPLAT_BACKWARD_DEBUG_INTERVAL", None)
    if raster_specialization:
        env["MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION"] = "1"
    else:
        env.pop("MSPLAT_ENABLE_RASTER_BACKWARD_SPECIALIZATION", None)
    if half_sorted_buffers:
        env["MSPLAT_HALF_SORTED_BUFFERS"] = "1"
    else:
        env.pop("MSPLAT_HALF_SORTED_BUFFERS", None)
    if intersection_key_bits == "default":
        env.pop("MSPLAT_INTERSECTION_KEY_BITS", None)
    else:
        env["MSPLAT_INTERSECTION_KEY_BITS"] = intersection_key_bits

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
    if raster_specialization:
        labels.append("raster-backward specialization")
    if half_sorted_buffers:
        labels.append("half sorted buffers")
    if intersection_key_bits != "default":
        labels.append(f"{intersection_key_bits}-bit intersection keys" if intersection_key_bits != "auto" else "auto intersection keys")
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


def raster_specialization_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def half_sorted_buffer_variants(value: str) -> list[bool]:
    if value == "both":
        return [False, True]
    return [value == "on"]


def intersection_key_bit_variants(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def has_final_quality_arg(args: list[str]) -> bool:
    return any(arg in {"--final-quality", "--no-final-quality"} for arg in args)


def print_table(results: list[dict[str, object]], output_dir: Path) -> None:
    print(f"Logs: {output_dir}")
    print()
    print("| mode | train it/s | train s | iter median ms | rast_bwd median ms | PSNR | SSIM | L1 | splats | tile avg | pixel atomic M | merge ceiling % | replay active M | replay diag M | replay skip M | sat px |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        tile_avg = "-"
        if row["tile_avg_before"] is not None and row["tile_avg_after"] is not None:
            tile_avg = f"{fmt(row['tile_avg_before'], 1)}->{fmt(row['tile_avg_after'], 1)}"
        print(
            "| {mode} | {ips} | {train_s} | {iter_ms} | {bwd_ms} | {psnr} | {ssim} | {l1} | {splats} | {tile_avg} | {atomic_groups} | {merge_ceiling} | {replay} | {replay_diag} | {replay_skip} | {sat_px} |".format(
                mode=row["mode"],
                ips=fmt(row["training_ips"], 2),
                train_s=fmt(row["training_seconds"], 2),
                iter_ms=fmt(row["iter_median_ms"]),
                bwd_ms=fmt(row["rast_bwd_median_ms"]),
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
    print_baseline_deltas(results)


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


def main() -> int:
    args = parse_args()
    extra_args = args.msplat_args
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="msplat_backward_ab."))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_mode(args, mode, raster_specialization, half_sorted_buffers, intersection_key_bits, output_dir, extra_args)
        for mode in args.modes
        for raster_specialization in raster_specialization_variants(args.raster_backward_specialization)
        for half_sorted_buffers in half_sorted_buffer_variants(args.half_sorted_buffers)
        for intersection_key_bits in intersection_key_bit_variants(args.intersection_key_bits)
    ]
    print_table(results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
