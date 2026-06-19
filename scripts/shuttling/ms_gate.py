#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run standalone analog MS-gate scenarios and write inspectable outputs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if TYPE_CHECKING:
    from mqt.yaqs.applications.shuttling import CollectiveMotionalState, MSGateConfig, MSGateResult


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured MS-gate scenarios."""
    args = _parse_args(argv)
    from mqt.yaqs.applications.shuttling import simulate_ms_gate

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    cases = _build_cases(args)
    results: dict[str, MSGateResult] = {}
    summary: dict[str, dict[str, float]] = {}

    print(f"Writing MS-gate outputs to {output_dir}")
    for index, (name, motional_state) in enumerate(cases.items()):
        case_config = config
        if index > 0:
            case_config = replace(
                config,
                drive_amplitude=results["cold"].calibrated_drive_amplitude,
                calibrate=False,
            )
        print(f"Running {name}: model={case_config.sideband_model}, dt={case_config.dt:g}")
        result = simulate_ms_gate(case_config, motional_state)
        results[name] = result
        _save_arrays(output_dir / f"{name}.npz", result)
        _save_static_plots(output_dir, name, result, dpi=args.dpi)
        summary[name] = _summarize(result)

    if args.include_linear_baseline:
        linear_config = replace(
            config,
            sideband_model="linear",
            drive_amplitude=None,
            calibrate=True,
        )
        for name, motional_state in cases.items():
            case_config = linear_config
            if name != "cold":
                case_config = replace(
                    linear_config,
                    drive_amplitude=results["linear_cold"].calibrated_drive_amplitude,
                    calibrate=False,
                )
            result = simulate_ms_gate(case_config, motional_state)
            results[f"linear_{name}"] = result
            _save_arrays(output_dir / f"linear_{name}.npz", result)
            summary[f"linear_{name}"] = _summarize(result)

    _save_comparison_plot(output_dir, results, dpi=args.dpi)
    imperfect = _save_loop_comparison(output_dir, config, cases["cold"], results["cold"], args, dpi=args.dpi)
    _save_arrays(output_dir / "imperfect_loop.npz", imperfect)
    summary["imperfect_loop"] = _summarize(imperfect)
    _save_summary(output_dir, summary)
    print("Done.")
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cold and motional-excited analog MS-gate scenarios.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "ms_gate")
    parser.add_argument("--com-cutoff", type=int, default=12)
    parser.add_argument("--stretch-cutoff", type=int, default=12)
    parser.add_argument("--duration", type=float, default=2.0 * 3.141592653589793)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--target-angle", type=float, default=0.5 * 3.141592653589793)
    parser.add_argument("--com-detuning", type=float, default=-1.0)
    parser.add_argument("--stretch-detuning", type=float, default=-2.0)
    parser.add_argument("--com-eta", type=float, default=0.10)
    parser.add_argument("--stretch-eta", type=float, default=0.08)
    parser.add_argument("--com-excited-level", type=int, default=5)
    parser.add_argument("--stretch-excited-level", type=int, default=8)
    parser.add_argument("--mixed-hot-weight", type=float, default=0.5)
    parser.add_argument("--sideband-model", choices=("finite_lamb_dicke", "linear"), default="finite_lamb_dicke")
    parser.add_argument("--calibration-scan-points", type=int, default=13)
    parser.add_argument("--calibration-upper", type=float, default=8.0)
    parser.add_argument("--include-linear-baseline", action="store_true")
    parser.add_argument(
        "--imperfect-duration-scale",
        type=float,
        default=0.85,
        help="Duration scale for the non-closed loop comparison using the calibrated cold drive.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> MSGateConfig:
    from mqt.yaqs.applications.shuttling import MSGateConfig, MotionalModeSpec

    modes = (
        MotionalModeSpec(
            name="COM",
            frequency=1.0,
            detuning=args.com_detuning,
            cutoff=args.com_cutoff,
            lamb_dicke=(args.com_eta, args.com_eta),
        ),
        MotionalModeSpec(
            name="stretch",
            frequency=3.0**0.5,
            detuning=args.stretch_detuning,
            cutoff=args.stretch_cutoff,
            lamb_dicke=(args.stretch_eta, -args.stretch_eta),
        ),
    )
    return MSGateConfig(
        modes=modes,
        duration=args.duration,
        dt=args.dt,
        target_angle=args.target_angle,
        sideband_model=args.sideband_model,
        calibration_scan_points=args.calibration_scan_points,
        calibration_amplitude_bounds=(0.0, args.calibration_upper),
    )


def _build_cases(args: argparse.Namespace) -> dict[str, CollectiveMotionalState]:
    import numpy as np
    from mqt.yaqs.applications.shuttling import (
        collective_diagonal_mixture,
        collective_fock_state,
        collective_ground_state,
    )

    cold = collective_ground_state(args.com_cutoff, args.stretch_cutoff)
    com_excited = collective_fock_state(args.com_cutoff, args.stretch_cutoff, args.com_excited_level, 0)
    stretch_excited = collective_fock_state(args.com_cutoff, args.stretch_cutoff, 0, args.stretch_excited_level)
    populations = np.zeros((args.com_cutoff, args.stretch_cutoff), dtype=np.float64)
    hot_weight = min(max(args.mixed_hot_weight, 0.0), 1.0)
    populations[0, 0] = 1.0 - hot_weight
    populations[args.com_excited_level, 0] += 0.5 * hot_weight
    populations[0, args.stretch_excited_level] += 0.5 * hot_weight
    return {
        "cold": cold,
        "com_excited": com_excited,
        "stretch_excited": stretch_excited,
        "mixed": collective_diagonal_mixture(populations),
    }


def _save_arrays(path: Path, result: MSGateResult) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        times=result.times,
        calibrated_drive_amplitude=np.array(result.calibrated_drive_amplitude, dtype=np.float64),
        calibration_fidelity=np.array(result.calibration_fidelity, dtype=np.float64),
        spin_density=result.spin_density,
        spin_populations=result.spin_populations,
        target_fidelity=result.target_fidelity,
        mode_expectations=result.mode_expectations,
        joint_mode_populations=result.joint_mode_populations,
        state_norm=result.state_norm,
    )


def _save_static_plots(output_dir: Path, name: str, result: MSGateResult, *, dpi: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import (  # noqa: PLC0415
        plot_ms_fidelity,
        plot_ms_mode_expectations,
        plot_ms_phase_space_loops,
        plot_ms_spin_populations,
    )

    fig, _ax = plot_ms_fidelity(result)
    fig.savefig(output_dir / f"{name}_fidelity.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_ms_spin_populations(result)
    fig.savefig(output_dir / f"{name}_spin_populations.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _ax = plot_ms_mode_expectations(result)
    fig.savefig(output_dir / f"{name}_mode_expectations.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, _axes = plot_ms_phase_space_loops(result)
    fig.savefig(output_dir / f"{name}_phase_space_loops.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_comparison_plot(output_dir: Path, results: dict[str, MSGateResult], *, dpi: int) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    from mqt.yaqs.applications.shuttling.visualization import plot_ms_final_fidelities  # noqa: PLC0415

    fig, _ax = plot_ms_final_fidelities(results)
    fig.savefig(output_dir / "final_fidelity_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_loop_comparison(
    output_dir: Path,
    config: MSGateConfig,
    cold_state: CollectiveMotionalState,
    cold_result: MSGateResult,
    args: argparse.Namespace,
    *,
    dpi: int,
) -> MSGateResult:
    from mqt.yaqs.applications.shuttling import simulate_ms_gate  # noqa: PLC0415
    from mqt.yaqs.applications.shuttling.visualization import plot_ms_phase_space_loops  # noqa: PLC0415

    plt = importlib.import_module("matplotlib.pyplot")
    imperfect_config = replace(
        config,
        duration=args.imperfect_duration_scale * config.duration,
        drive_amplitude=cold_result.calibrated_drive_amplitude,
        calibrate=False,
    )
    imperfect = simulate_ms_gate(imperfect_config, cold_state)
    fig, _axes = plot_ms_phase_space_loops({"closed": cold_result, "short gate": imperfect})
    fig.savefig(output_dir / "phase_space_loop_closed_vs_imperfect.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return imperfect


def _summarize(result: MSGateResult) -> dict[str, float]:
    import numpy as np

    return {
        "calibrated_drive_amplitude": float(result.calibrated_drive_amplitude),
        "calibration_fidelity": float(result.calibration_fidelity),
        "final_target_fidelity": float(result.target_fidelity[-1]),
        "min_target_fidelity": float(np.min(result.target_fidelity)),
        "final_com_occupation": float(result.mode_expectations[-1, 0]),
        "final_stretch_occupation": float(result.mode_expectations[-1, 1]),
        "final_state_norm": float(result.state_norm[-1]),
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
