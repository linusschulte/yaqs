#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run two-ion shuttling with softened Coulomb repulsion and write outputs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if TYPE_CHECKING:
    from mqt.yaqs.applications.shuttling import TwoIonShuttlingConfig, TwoIonShuttlingResult


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured two-ion shuttling scenarios."""
    args = _parse_args(argv)
    from mqt.yaqs.applications.shuttling import simulate_two_ion_shuttling

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    cases = _build_cases(args)

    print(f"Writing two-ion shuttling outputs to {output_dir}")
    for name, config in cases.items():
        print(f"Running {name}: duration={config.duration:g}, dt={config.dt:g}, grid={config.grid.num_points}")
        result = simulate_two_ion_shuttling(config)
        _save_arrays(output_dir / f"{name}.npz", result)
        _save_static_plots(output_dir, name, result, dpi=args.dpi)
        if args.animate == "all" or (args.animate == "fast" and name == "fast_shuttle"):
            _save_animations(output_dir, name, result)
        summary[name] = _summarize(result)

    _save_summary(output_dir, summary)
    print("Done.")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no/slow/fast two-ion shuttling scenarios and save plots, arrays, and animations.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "shuttling_two_ion")
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=8.0)
    parser.add_argument("--grid-points", type=int, default=96)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--hbar", type=float, default=1.0)
    parser.add_argument("--coulomb-strength", type=float, default=1.0)
    parser.add_argument("--softening-length", type=float, default=None)
    parser.add_argument(
        "--initial-state",
        choices=("collective_ground_state", "localized_product_state", "localized_crystal_ground_state"),
        default="localized_crystal_ground_state",
    )
    parser.add_argument("--initial-x1", type=float, default=None)
    parser.add_argument("--initial-x2", type=float, default=None)
    parser.add_argument("--q-initial", type=float, default=-1.5)
    parser.add_argument("--q-final", type=float, default=1.5)
    parser.add_argument("--no-duration", type=float, default=2.0)
    parser.add_argument("--slow-duration", type=float, default=8.0)
    parser.add_argument("--fast-duration", type=float, default=1.0)
    parser.add_argument("--trail-time", type=float, default=0.0)
    parser.add_argument("--no-dt", type=float, default=0.04)
    parser.add_argument("--slow-dt", type=float, default=0.04)
    parser.add_argument("--fast-dt", type=float, default=0.02)
    parser.add_argument("--snapshot-stride", type=int, default=5)
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument(
        "--mode-population-levels",
        type=int,
        default=6,
        help="Number of instantaneous COM/stretch oscillator levels to project onto. Use 0 to disable.",
    )
    parser.add_argument(
        "--com-mode-population-levels",
        type=int,
        default=None,
        help="Optional COM-like oscillator level cutoff overriding --mode-population-levels.",
    )
    parser.add_argument(
        "--stretch-mode-population-levels",
        type=int,
        default=None,
        help="Optional stretch-like oscillator level cutoff overriding --mode-population-levels.",
    )
    parser.add_argument("--animate", choices=("none", "fast", "all"), default="none")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _build_cases(args: argparse.Namespace) -> dict[str, TwoIonShuttlingConfig]:
    from mqt.yaqs.applications.shuttling import CoulombInteraction, Ion, SpatialGrid, TwoIonShuttlingConfig

    grid = SpatialGrid(x_min=args.x_min, x_max=args.x_max, num_points=args.grid_points)
    ion = Ion(mass=args.mass)
    coulomb = CoulombInteraction(strength=args.coulomb_strength, softening_length=args.softening_length)
    snapshot_stride = None if args.no_snapshots else args.snapshot_stride
    return {
        "no_shuttle": TwoIonShuttlingConfig(
            ion1=ion,
            ion2=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=0.0,
            q_final=0.0,
            duration=args.no_duration,
            trail_time=args.trail_time,
            dt=args.no_dt,
            coulomb=coulomb,
            snapshot_stride=snapshot_stride,
            initial_state=args.initial_state,
            initial_x1=args.initial_x1,
            initial_x2=args.initial_x2,
            mode_population_levels=args.mode_population_levels,
            com_mode_population_levels=args.com_mode_population_levels,
            stretch_mode_population_levels=args.stretch_mode_population_levels,
        ),
        "slow_shuttle": TwoIonShuttlingConfig(
            ion1=ion,
            ion2=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=args.q_initial,
            q_final=args.q_final,
            duration=args.slow_duration,
            trail_time=args.trail_time,
            dt=args.slow_dt,
            coulomb=coulomb,
            snapshot_stride=snapshot_stride,
            initial_state=args.initial_state,
            initial_x1=args.initial_x1,
            initial_x2=args.initial_x2,
            mode_population_levels=args.mode_population_levels,
            com_mode_population_levels=args.com_mode_population_levels,
            stretch_mode_population_levels=args.stretch_mode_population_levels,
        ),
        "fast_shuttle": TwoIonShuttlingConfig(
            ion1=ion,
            ion2=ion,
            grid=grid,
            omega=args.omega,
            hbar=args.hbar,
            q_initial=args.q_initial,
            q_final=args.q_final,
            duration=args.fast_duration,
            trail_time=args.trail_time,
            dt=args.fast_dt,
            coulomb=coulomb,
            snapshot_stride=snapshot_stride,
            initial_state=args.initial_state,
            initial_x1=args.initial_x1,
            initial_x2=args.initial_x2,
            mode_population_levels=args.mode_population_levels,
            com_mode_population_levels=args.com_mode_population_levels,
            stretch_mode_population_levels=args.stretch_mode_population_levels,
        ),
    }


def _save_arrays(path: Path, result: TwoIonShuttlingResult) -> None:
    import numpy as np

    arrays = {
        "positions": result.positions,
        "times": result.times,
        "trap_center": result.trap_center,
        "trap_velocity": result.trap_velocity,
        "x1_expectation": result.x1_expectation,
        "x2_expectation": result.x2_expectation,
        "center_of_mass": result.center_of_mass,
        "mean_separation": result.mean_separation,
        "equilibrium_x1": result.equilibrium_x1,
        "equilibrium_x2": result.equilibrium_x2,
        "p1_expectation": result.p1_expectation,
        "p2_expectation": result.p2_expectation,
        "center_of_mass_momentum": result.center_of_mass_momentum,
        "relative_center_of_mass_momentum": result.relative_center_of_mass_momentum,
        "com_mode_frequency": np.array(result.com_mode_frequency, dtype=np.float64),
        "stretch_mode_frequency": np.array(result.stretch_mode_frequency, dtype=np.float64),
        "com_mode_coordinate": result.com_mode_coordinate,
        "stretch_mode_coordinate": result.stretch_mode_coordinate,
        "com_mode_momentum": result.com_mode_momentum,
        "stretch_mode_momentum": result.stretch_mode_momentum,
        "relative_com_mode_momentum": result.relative_com_mode_momentum,
        "relative_stretch_mode_momentum": result.relative_stretch_mode_momentum,
        "com_mode_excitation": result.com_mode_excitation,
        "stretch_mode_excitation": result.stretch_mode_excitation,
        "energy": result.energy,
        "excess_energy": result.excess_energy,
        "excess_quanta": result.excess_quanta,
        "norm": result.norm,
        "ground_energy_reference": np.array(result.ground_energy_reference, dtype=np.float64),
    }
    if result.snapshot_times is not None:
        arrays["snapshot_times"] = result.snapshot_times
    if result.pair_density is not None:
        arrays["pair_density"] = result.pair_density
    if result.com_mode_populations is not None:
        arrays["com_mode_populations"] = result.com_mode_populations
    if result.stretch_mode_populations is not None:
        arrays["stretch_mode_populations"] = result.stretch_mode_populations
    if result.joint_mode_populations is not None:
        arrays["joint_mode_populations"] = result.joint_mode_populations
    if result.mode_population_residual is not None:
        arrays["mode_population_residual"] = result.mode_population_residual
    if result.com_mode_population_excitation is not None:
        arrays["com_mode_population_excitation"] = result.com_mode_population_excitation
    if result.stretch_mode_population_excitation is not None:
        arrays["stretch_mode_population_excitation"] = result.stretch_mode_population_excitation
    if result.final_state is not None:
        arrays["final_state"] = result.final_state
    savez_compressed = cast("Any", np.savez_compressed)
    savez_compressed(path, **arrays)


def _save_static_plots(output_dir: Path, name: str, result: TwoIonShuttlingResult, *, dpi: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import (  # noqa: PLC0415
        plot_center_of_mass_trace,
        plot_excess_energy_trace,
        plot_joint_mode_populations,
        plot_mean_separation_trace,
        plot_mode_excitation_trace,
        plot_mode_population_consistency,
        plot_mode_populations,
        plot_normal_mode_phase_spaces,
        plot_pair_density_snapshot,
        plot_two_ion_marginal_trajectories,
        plot_two_ion_phase_space,
    )

    fig, _ax = plot_center_of_mass_trace(result)
    fig.savefig(output_dir / f"{name}_center_of_mass_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_mean_separation_trace(result)
    fig.savefig(output_dir / f"{name}_separation_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_excess_energy_trace(result)
    fig.savefig(output_dir / f"{name}_excess_energy_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_mode_excitation_trace(result)
    fig.savefig(output_dir / f"{name}_mode_excitation_trace.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if result.com_mode_populations is not None:
        fig, _axes = plot_mode_populations(result)
        fig.savefig(output_dir / f"{name}_mode_populations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if result.joint_mode_populations is not None:
        fig, _axes = plot_joint_mode_populations(result)
        fig.savefig(output_dir / f"{name}_joint_mode_populations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if result.com_mode_population_excitation is not None:
        fig, _axes = plot_mode_population_consistency(result)
        fig.savefig(output_dir / f"{name}_mode_population_consistency.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    fig, _ax = plot_two_ion_phase_space(result)
    fig.savefig(output_dir / f"{name}_center_of_mass_phase_space.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _axes = plot_normal_mode_phase_spaces(result)
    fig.savefig(output_dir / f"{name}_normal_mode_phase_spaces.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if result.pair_density is not None:
        fig, _axes = plot_two_ion_marginal_trajectories(result)
        fig.savefig(output_dir / f"{name}_x1_x2_density_trajectories.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if result.pair_density is not None:
        fig, _ax = plot_pair_density_snapshot(result)
        fig.savefig(output_dir / f"{name}_pair_density_final.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _save_animations(output_dir: Path, name: str, result: TwoIonShuttlingResult) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import (  # noqa: PLC0415
        animate_pair_density,
        animate_two_ion_marginals,
    )

    if result.pair_density is None:
        return

    pair_density_animation = animate_pair_density(result)
    (output_dir / f"{name}_pair_density_animation.html").write_text(
        pair_density_animation.to_jshtml(),
        encoding="utf-8",
    )

    marginal_animation = animate_two_ion_marginals(result)
    (output_dir / f"{name}_x1_x2_marginals_animation.html").write_text(
        marginal_animation.to_jshtml(), encoding="utf-8"
    )

    plt.close("all")


def _summarize(result: TwoIonShuttlingResult) -> dict[str, float]:
    import numpy as np

    com_lag = result.center_of_mass - result.trap_center
    summary = {
        "final_excess_quanta": float(result.excess_quanta[-1]),
        "max_excess_quanta": float(np.max(result.excess_quanta)),
        "final_center_of_mass": float(result.center_of_mass[-1]),
        "final_trap_center": float(result.trap_center[-1]),
        "final_mean_separation": float(result.mean_separation[-1]),
        "max_abs_center_of_mass_lag": float(np.max(np.abs(com_lag))),
        "final_com_mode_excitation": float(result.com_mode_excitation[-1]),
        "final_stretch_mode_excitation": float(result.stretch_mode_excitation[-1]),
        "max_com_mode_excitation": float(np.max(result.com_mode_excitation)),
        "max_stretch_mode_excitation": float(np.max(result.stretch_mode_excitation)),
        "final_center_of_mass_momentum": float(result.center_of_mass_momentum[-1]),
        "final_relative_center_of_mass_momentum": float(result.relative_center_of_mass_momentum[-1]),
        "ground_energy_reference": float(result.ground_energy_reference),
        "final_norm": float(result.norm[-1]),
    }
    if result.com_mode_populations is not None and result.stretch_mode_populations is not None:
        summary["final_com_mode_ground_population"] = float(result.com_mode_populations[-1, 0])
        summary["final_stretch_mode_ground_population"] = float(result.stretch_mode_populations[-1, 0])
    if result.mode_population_residual is not None:
        summary["final_mode_population_residual"] = float(result.mode_population_residual[-1])
        summary["max_mode_population_residual"] = float(np.max(result.mode_population_residual))
    if result.com_mode_population_excitation is not None and result.stretch_mode_population_excitation is not None:
        final_com_difference = result.com_mode_excitation[-1] - result.com_mode_population_excitation[-1]
        final_stretch_difference = result.stretch_mode_excitation[-1] - result.stretch_mode_population_excitation[-1]
        summary["final_com_mode_population_excitation"] = float(result.com_mode_population_excitation[-1])
        summary["final_stretch_mode_population_excitation"] = float(result.stretch_mode_population_excitation[-1])
        summary["final_com_mode_population_difference"] = float(final_com_difference)
        summary["final_stretch_mode_population_difference"] = float(final_stretch_difference)
        summary["max_abs_com_mode_population_difference"] = float(
            np.max(np.abs(result.com_mode_excitation - result.com_mode_population_excitation))
        )
        summary["max_abs_stretch_mode_population_difference"] = float(
            np.max(np.abs(result.stretch_mode_excitation - result.stretch_mode_population_excitation))
        )
    return summary


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
