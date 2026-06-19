#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the single-ion shuttling milestone and write inspectable outputs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if TYPE_CHECKING:
    from mqt.yaqs.applications.shuttling import ShuttlingConfig, ShuttlingResult


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured shuttling scenarios."""
    args = _parse_args(argv)
    from mqt.yaqs.applications.shuttling import simulate_shuttling

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    cases = _build_cases(args)

    print(f"Writing shuttling outputs to {output_dir}")
    for name, config in cases.items():
        print(f"Running {name}: duration={config.duration:g}, dt={config.dt:g}")
        result = simulate_shuttling(config)
        _save_arrays(output_dir / f"{name}.npz", result)
        _save_static_plots(output_dir, name, result, dpi=args.dpi)
        if args.animate == "all" or (args.animate == "fast" and name == "fast_shuttle"):
            _save_animation(output_dir / f"{name}_animation.html", result, stride=args.animation_stride)
        summary[name] = _summarize(result)

    _save_summary(output_dir, summary)
    print("Done.")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no/slow/fast single-ion shuttling scenarios and save plots, arrays, and animations.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "shuttling")
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--grid-points", type=int, default=256)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--hbar", type=float, default=1.0)
    parser.add_argument("--q-initial", type=float, default=-1.5)
    parser.add_argument("--q-final", type=float, default=1.5)
    parser.add_argument("--no-duration", type=float, default=2.0)
    parser.add_argument("--slow-duration", type=float, default=8.0)
    parser.add_argument("--fast-duration", type=float, default=1.0)
    parser.add_argument("--trail-time", type=float, default=0.0)
    parser.add_argument("--no-dt", type=float, default=0.02)
    parser.add_argument("--slow-dt", type=float, default=0.02)
    parser.add_argument("--fast-dt", type=float, default=0.01)
    parser.add_argument("--population-levels", type=int, default=4)
    parser.add_argument("--animate", choices=("none", "fast", "all"), default="fast")
    parser.add_argument("--animation-stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _build_cases(args: argparse.Namespace) -> dict[str, ShuttlingConfig]:
    from mqt.yaqs.applications.shuttling import Ion, ShuttlingConfig, SpatialGrid

    grid = SpatialGrid(x_min=args.x_min, x_max=args.x_max, num_points=args.grid_points)
    ion = Ion(mass=args.mass)
    return {
        "no_shuttle": ShuttlingConfig(
            ion=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=0.0,
            q_final=0.0,
            duration=args.no_duration,
            trail_time=args.trail_time,
            dt=args.no_dt,
            population_levels=args.population_levels,
        ),
        "slow_shuttle": ShuttlingConfig(
            ion=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=args.q_initial,
            q_final=args.q_final,
            duration=args.slow_duration,
            trail_time=args.trail_time,
            dt=args.slow_dt,
            population_levels=args.population_levels,
        ),
        "fast_shuttle": ShuttlingConfig(
            ion=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=args.q_initial,
            q_final=args.q_final,
            duration=args.fast_duration,
            trail_time=args.trail_time,
            dt=args.fast_dt,
            population_levels=args.population_levels,
        ),
    }


def _save_arrays(path: Path, result: ShuttlingResult) -> None:
    import numpy as np

    arrays = {
        "positions": result.positions,
        "times": result.times,
        "trap_center": result.trap_center,
        "trap_velocity": result.trap_velocity,
        "position_expectation": result.position_expectation,
        "position_std": result.position_std,
        "momentum_expectation": result.momentum_expectation,
        "relative_momentum_expectation": result.relative_momentum_expectation,
        "momentum_std": result.momentum_std,
        "excitation_number": result.excitation_number,
        "density": result.density,
        "potential_energy": result.potential_energy,
        "norm": result.norm,
    }
    if result.populations is not None:
        arrays["populations"] = result.populations
    savez_compressed = cast("Any", np.savez_compressed)
    savez_compressed(path, **arrays)


def _save_static_plots(output_dir: Path, name: str, result: ShuttlingResult, *, dpi: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import (  # noqa: PLC0415
        plot_excitation_trace,
        plot_momentum_trace,
        plot_phase_space,
        plot_populations,
        plot_position_trace,
    )

    fig, _ax = plot_position_trace(result)
    fig.savefig(output_dir / f"{name}_position_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_excitation_trace(result)
    fig.savefig(output_dir / f"{name}_excitation_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_momentum_trace(result)
    fig.savefig(output_dir / f"{name}_momentum_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_phase_space(result)
    fig.savefig(output_dir / f"{name}_phase_space.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if result.populations is not None:
        fig, _ax = plot_populations(result)
        fig.savefig(output_dir / f"{name}_populations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _save_animation(path: Path, result: ShuttlingResult, *, stride: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import animate_shuttling  # noqa: PLC0415

    animation_result = _stride_result(result, stride)
    animation = animate_shuttling(animation_result)
    path.write_text(animation.to_jshtml(), encoding="utf-8")
    plt.close("all")


def _stride_result(result: ShuttlingResult, stride: int) -> ShuttlingResult:
    if stride <= 1:
        return result
    return replace(
        result,
        times=result.times[::stride],
        trap_center=result.trap_center[::stride],
        trap_velocity=result.trap_velocity[::stride],
        position_expectation=result.position_expectation[::stride],
        position_std=result.position_std[::stride],
        momentum_expectation=result.momentum_expectation[::stride],
        relative_momentum_expectation=result.relative_momentum_expectation[::stride],
        momentum_std=result.momentum_std[::stride],
        excitation_number=result.excitation_number[::stride],
        density=result.density[::stride],
        potential_energy=result.potential_energy[::stride],
        norm=result.norm[::stride],
        populations=None if result.populations is None else result.populations[::stride],
    )


def _summarize(result: ShuttlingResult) -> dict[str, float]:
    import numpy as np

    lag = result.position_expectation - result.trap_center
    return {
        "final_excitation": float(result.excitation_number[-1]),
        "max_excitation": float(np.max(result.excitation_number)),
        "final_position_expectation": float(result.position_expectation[-1]),
        "final_trap_center": float(result.trap_center[-1]),
        "initial_position_std": float(result.position_std[0]),
        "final_position_std": float(result.position_std[-1]),
        "max_position_std": float(np.max(result.position_std)),
        "final_momentum_expectation": float(result.momentum_expectation[-1]),
        "final_relative_momentum_expectation": float(result.relative_momentum_expectation[-1]),
        "initial_momentum_std": float(result.momentum_std[0]),
        "final_momentum_std": float(result.momentum_std[-1]),
        "max_abs_lag": float(np.max(np.abs(lag))),
        "final_norm": float(result.norm[-1]),
    }


def _save_summary(output_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    fieldnames = ["case", *next(iter(summary.values())).keys()]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in summary.items():
            writer.writerow({"case": name, **values})


if __name__ == "__main__":
    raise SystemExit(main())
