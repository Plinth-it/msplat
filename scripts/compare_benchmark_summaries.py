#!/usr/bin/env python3
"""Compare benchmark_backward_rasterizers.py summary.json artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact cross-run comparison for benchmark summary.json files."
    )
    parser.add_argument("summaries", nargs="+", type=Path, help="summary.json paths to compare")
    return parser.parse_args()


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(value: object, baseline: object) -> float | None:
    if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
        return None
    return float(value) - float(baseline)


def percent_delta(value: object, baseline: object) -> float | None:
    if not isinstance(value, (float, int)) or not isinstance(baseline, (float, int)):
        return None
    if baseline == 0:
        return None
    return 100.0 * (float(value) - float(baseline)) / abs(float(baseline))


def fmt_delta(value: object, baseline: object, digits: int = 1, percent: bool = True) -> str:
    value_delta = percent_delta(value, baseline) if percent else delta(value, baseline)
    if value_delta is None:
        return "-"
    suffix = "%" if percent else ""
    return f"{value_delta:+.{digits}f}{suffix}"


def run_label(summary_path: Path, summary: dict[str, object]) -> str:
    output_dir = summary.get("output_dir")
    if isinstance(output_dir, str) and output_dir:
        return Path(output_dir).name
    return summary_path.parent.name


def warning_map(summary: dict[str, object]) -> dict[str, str]:
    warnings: dict[str, str] = {}
    for warning in summary.get("quality_warnings", []):
        if not isinstance(warning, dict):
            continue
        mode = warning.get("mode")
        reasons = warning.get("reasons")
        if isinstance(mode, str) and isinstance(reasons, list):
            warnings[mode] = ", ".join(str(reason) for reason in reasons)
    return warnings


def print_summary(path: Path) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    results = summary.get("results", [])
    if not isinstance(results, list) or not results:
        raise SystemExit(f"{path}: no benchmark results found")
    baseline = results[0]
    if not isinstance(baseline, dict):
        raise SystemExit(f"{path}: invalid baseline result")

    warnings = warning_map(summary)
    label = run_label(path, summary)
    dataset = summary.get("dataset", "-")
    print(f"\n## {label}")
    print(f"dataset: {dataset}")
    print()
    print("| mode | train it/s | it/s delta | iter median delta | rast_bwd delta | PSNR delta | SSIM delta | L1 delta | splats delta | warnings |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode", "-"))
        print(
            "| {mode} | {ips} | {ips_delta} | {iter_delta} | {bwd_delta} | {psnr_delta} | {ssim_delta} | {l1_delta} | {splat_delta} | {warnings} |".format(
                mode=mode,
                ips=fmt(row.get("training_ips"), 2),
                ips_delta=fmt_delta(row.get("training_ips"), baseline.get("training_ips")),
                iter_delta=fmt_delta(row.get("iter_median_ms"), baseline.get("iter_median_ms")),
                bwd_delta=fmt_delta(row.get("rast_bwd_median_ms"), baseline.get("rast_bwd_median_ms")),
                psnr_delta=fmt_delta(row.get("psnr"), baseline.get("psnr"), digits=2, percent=False),
                ssim_delta=fmt_delta(row.get("ssim"), baseline.get("ssim"), digits=4, percent=False),
                l1_delta=fmt_delta(row.get("l1"), baseline.get("l1")),
                splat_delta=fmt_delta(row.get("splats"), baseline.get("splats")),
                warnings=warnings.get(mode, "-"),
            )
        )


def main() -> int:
    args = parse_args()
    for path in args.summaries:
        print_summary(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
