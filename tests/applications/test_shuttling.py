# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for single-ion shuttling applications."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pytest

from mqt.yaqs.applications.shuttling import (
    MSGateConfig,
    CoulombInteraction,
    Ion,
    ShuttlingConfig,
    SpatialGrid,
    TwoIonShuttlingConfig,
    collective_diagonal_mixture,
    collective_fock_state,
    collective_ground_state,
    default_ms_modes,
    ms_phase_space_displacements,
    project_two_ion_state_to_collective_modes,
    quintic_trajectory,
    simulate_ms_gate,
    simulate_shuttling,
    simulate_two_ion_shuttling,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _test_grid() -> SpatialGrid:
    return SpatialGrid(x_min=-8.0, x_max=8.0, num_points=160)


def test_quintic_trajectory_endpoints_are_stationary() -> None:
    """The default trajectory reaches both endpoints with zero velocity."""
    duration = 3.0
    trajectory = quintic_trajectory(-1.0, 2.0, duration)
    eps = 1e-5

    assert np.isclose(trajectory(0.0), -1.0)
    assert np.isclose(trajectory(duration), 2.0)
    assert np.isclose((trajectory(eps) - trajectory(0.0)) / eps, 0.0, atol=1e-8)
    assert np.isclose((trajectory(duration) - trajectory(duration - eps)) / eps, 0.0, atol=1e-8)


def test_no_shuttle_remains_in_ground_state() -> None:
    """A stationary well should keep the wave packet centered and unexcited."""
    config = ShuttlingConfig(
        ion=Ion(mass=1.0),
        grid=_test_grid(),
        omega=1.0,
        q_initial=0.5,
        q_final=0.5,
        duration=1.0,
        dt=0.05,
        population_levels=3,
    )

    result = simulate_shuttling(config)

    assert np.allclose(result.norm, 1.0, atol=1e-8)
    assert np.allclose(result.position_expectation, config.q_initial, atol=2e-3)
    assert np.max(result.excitation_number) < 2e-2
    assert result.populations is not None
    assert result.populations[0, 0] > 0.999


def test_slow_shuttle_is_less_excited_than_fast_shuttle() -> None:
    """For the same displacement, slower transport should reduce final excitation."""
    ion = Ion(mass=1.0)
    grid = _test_grid()
    fast = simulate_shuttling(
        ShuttlingConfig(ion=ion, grid=grid, omega=1.0, q_initial=-1.0, q_final=1.0, dt=0.04, duration=0.5)
    )
    slow = simulate_shuttling(
        ShuttlingConfig(ion=ion, grid=grid, omega=1.0, q_initial=-1.0, q_final=1.0, dt=0.04, duration=5.0)
    )

    assert fast.excitation_number[-1] > 1e-2
    assert slow.excitation_number[-1] < fast.excitation_number[-1]


def test_result_shapes_and_values_are_finite() -> None:
    """Simulation output arrays should have consistent shapes and finite entries."""
    config = ShuttlingConfig(
        grid=SpatialGrid(x_min=-7.0, x_max=7.0, num_points=96),
        q_initial=-0.5,
        q_final=0.5,
        duration=0.2,
        dt=0.05,
        population_levels=4,
    )

    result = simulate_shuttling(config)
    expected_shape = (result.times.size, result.positions.size)

    assert result.density.shape == expected_shape
    assert result.potential_energy.shape == expected_shape
    assert result.trap_center.shape == result.times.shape
    assert result.trap_velocity.shape == result.times.shape
    assert result.position_expectation.shape == result.times.shape
    assert result.position_std.shape == result.times.shape
    assert result.momentum_expectation.shape == result.times.shape
    assert result.relative_momentum_expectation.shape == result.times.shape
    assert result.momentum_std.shape == result.times.shape
    assert result.excitation_number.shape == result.times.shape
    assert result.norm.shape == result.times.shape
    assert result.populations is not None
    assert result.populations.shape == (result.times.size, config.population_levels)
    assert np.all(np.isfinite(result.density))
    assert np.all(np.isfinite(result.potential_energy))
    assert np.all(np.isfinite(result.trap_velocity))
    assert np.all(np.isfinite(result.position_std))
    assert np.all(np.isfinite(result.momentum_expectation))
    assert np.all(np.isfinite(result.relative_momentum_expectation))
    assert np.all(np.isfinite(result.momentum_std))
    assert np.all(np.isfinite(result.excitation_number))
    assert np.all(np.isfinite(result.populations))


def test_trail_time_extends_simulation_after_trajectory_stops() -> None:
    """Trail time should keep evolving after the trap center has reached q_final."""
    config = ShuttlingConfig(
        grid=SpatialGrid(x_min=-7.0, x_max=7.0, num_points=96),
        q_initial=-0.5,
        q_final=0.5,
        duration=0.2,
        trail_time=0.3,
        dt=0.1,
    )

    result = simulate_shuttling(config)

    assert np.isclose(result.times[-1], config.duration + config.trail_time)
    assert np.allclose(result.trap_center[result.times >= config.duration], config.q_final)


def test_shuttling_import_does_not_import_matplotlib() -> None:
    """The core shuttling package should not require Matplotlib at import time."""
    imported_before = {module for module in sys.modules if module.startswith("matplotlib")}
    importlib.import_module("mqt.yaqs.applications.shuttling")
    imported_after = {module for module in sys.modules if module.startswith("matplotlib")}
    assert imported_after == imported_before


def test_runner_builds_no_slow_fast_cases(tmp_path: Path) -> None:
    """The console runner should expose the three standard sanity scenarios."""
    runner = _load_runner()
    args = runner._parse_args(  # noqa: SLF001
        ["--output-dir", str(tmp_path), "--grid-points", "48", "--animate", "none"]
    )

    cases = runner._build_cases(args)  # noqa: SLF001

    assert list(cases) == ["no_shuttle", "slow_shuttle", "fast_shuttle"]
    assert args.output_dir == tmp_path
    assert cases["no_shuttle"].q_initial == 0.0
    assert cases["no_shuttle"].q_final == 0.0
    assert cases["slow_shuttle"].duration > cases["fast_shuttle"].duration
    assert cases["fast_shuttle"].trail_time == 0.0
    assert cases["fast_shuttle"].grid.num_points == 48


def test_two_ion_no_shuttle_remains_in_ground_state() -> None:
    """A stationary two-ion well should preserve the initialized ground state."""
    config = TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-5.0, x_max=5.0, num_points=32),
        q_initial=0.0,
        q_final=0.0,
        duration=0.2,
        dt=0.1,
        snapshot_stride=1,
    )

    result = simulate_two_ion_shuttling(config)

    assert np.allclose(result.norm, 1.0, atol=1e-8)
    assert np.allclose(result.center_of_mass, 0.0, atol=5e-3)
    assert np.max(result.excess_quanta) < 1e-5


def test_two_ion_slow_shuttle_is_less_excited_than_fast_shuttle() -> None:
    """The two-ion exact-grid model should reproduce the slow-vs-fast trend."""
    grid = SpatialGrid(x_min=-5.0, x_max=5.0, num_points=32)
    fast = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(grid=grid, q_initial=-1.0, q_final=1.0, duration=0.4, dt=0.1, snapshot_stride=None)
    )
    slow = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(grid=grid, q_initial=-1.0, q_final=1.0, duration=3.0, dt=0.1, snapshot_stride=None)
    )

    assert fast.excess_quanta[-1] > slow.excess_quanta[-1]
    assert np.max(np.abs(fast.center_of_mass - fast.trap_center)) > np.max(
        np.abs(slow.center_of_mass - slow.trap_center)
    )


def test_two_ion_coulomb_increases_mean_separation() -> None:
    """Repulsive Coulomb interaction should increase the two-ion separation."""
    grid = SpatialGrid(x_min=-5.0, x_max=5.0, num_points=32)
    no_coulomb = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(
            grid=grid,
            q_initial=0.0,
            q_final=0.0,
            duration=0.0,
            coulomb=CoulombInteraction(strength=0.0),
            snapshot_stride=None,
        )
    )
    repulsive = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(
            grid=grid,
            q_initial=0.0,
            q_final=0.0,
            duration=0.0,
            coulomb=CoulombInteraction(strength=1.0),
            snapshot_stride=None,
        )
    )

    assert repulsive.mean_separation[0] > no_coulomb.mean_separation[0]


def test_two_ion_decoupled_center_of_mass_matches_single_ion() -> None:
    """With no Coulomb interaction, the two-ion COM follows the single-ion trajectory."""
    grid = SpatialGrid(x_min=-5.0, x_max=5.0, num_points=32)
    single = simulate_shuttling(
        ShuttlingConfig(grid=grid, q_initial=-0.8, q_final=0.8, duration=0.4, dt=0.1, population_levels=0)
    )
    two_ion = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(
            grid=grid,
            q_initial=-0.8,
            q_final=0.8,
            duration=0.4,
            dt=0.1,
            coulomb=CoulombInteraction(strength=0.0),
            snapshot_stride=None,
        )
    )

    assert np.allclose(two_ion.center_of_mass, single.position_expectation, atol=8e-2)


def test_two_ion_result_shapes_and_values_are_finite() -> None:
    """Two-ion simulation output arrays should have consistent shapes and finite entries."""
    config = TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-5.0, x_max=5.0, num_points=28),
        q_initial=-0.5,
        q_final=0.5,
        duration=0.2,
        dt=0.1,
        snapshot_stride=1,
        mode_population_levels=3,
        com_mode_population_levels=4,
    )

    result = simulate_two_ion_shuttling(config)

    assert result.x1_expectation.shape == result.times.shape
    assert result.x2_expectation.shape == result.times.shape
    assert result.center_of_mass.shape == result.times.shape
    assert result.mean_separation.shape == result.times.shape
    assert result.equilibrium_x1.shape == result.times.shape
    assert result.equilibrium_x2.shape == result.times.shape
    assert result.p1_expectation.shape == result.times.shape
    assert result.p2_expectation.shape == result.times.shape
    assert result.com_mode_coordinate.shape == result.times.shape
    assert result.stretch_mode_coordinate.shape == result.times.shape
    assert result.com_mode_momentum.shape == result.times.shape
    assert result.stretch_mode_momentum.shape == result.times.shape
    assert result.relative_com_mode_momentum.shape == result.times.shape
    assert result.relative_stretch_mode_momentum.shape == result.times.shape
    assert result.com_mode_excitation.shape == result.times.shape
    assert result.stretch_mode_excitation.shape == result.times.shape
    assert result.excess_quanta.shape == result.times.shape
    assert result.snapshot_times is not None
    assert result.pair_density is not None
    assert result.com_mode_populations is not None
    assert result.stretch_mode_populations is not None
    assert result.joint_mode_populations is not None
    assert result.mode_population_residual is not None
    assert result.com_mode_population_excitation is not None
    assert result.stretch_mode_population_excitation is not None
    assert result.pair_density.shape == (result.snapshot_times.size, config.grid.num_points, config.grid.num_points)
    assert result.com_mode_populations.shape == (result.times.size, 4)
    assert result.stretch_mode_populations.shape == (result.times.size, config.mode_population_levels)
    assert result.joint_mode_populations.shape == (
        result.times.size,
        4,
        config.mode_population_levels,
    )
    assert result.mode_population_residual.shape == result.times.shape
    assert result.com_mode_population_excitation.shape == result.times.shape
    assert result.stretch_mode_population_excitation.shape == result.times.shape
    assert np.all(np.isfinite(result.center_of_mass))
    assert np.all(np.isfinite(result.mean_separation))
    assert np.all(np.isfinite(result.com_mode_excitation))
    assert np.all(np.isfinite(result.stretch_mode_excitation))
    assert np.all(np.isfinite(result.excess_quanta))
    assert np.all(np.isfinite(result.pair_density))
    assert np.all(np.isfinite(result.com_mode_populations))
    assert np.all(np.isfinite(result.stretch_mode_populations))
    assert np.all(np.isfinite(result.joint_mode_populations))
    assert np.all(np.isfinite(result.mode_population_residual))
    assert np.all(np.isfinite(result.com_mode_population_excitation))
    assert np.all(np.isfinite(result.stretch_mode_population_excitation))
    assert np.all(result.mode_population_residual >= 0.0)


def test_two_ion_runner_builds_no_slow_fast_cases(tmp_path: Path) -> None:
    """The two-ion console runner should expose the standard sanity scenarios."""
    runner = _load_runner("two_ion_shuttling.py")
    args = runner._parse_args(  # noqa: SLF001
        ["--output-dir", str(tmp_path), "--grid-points", "32", "--animate", "none"]
    )

    cases = runner._build_cases(args)  # noqa: SLF001

    assert list(cases) == ["no_shuttle", "slow_shuttle", "fast_shuttle"]
    assert args.output_dir == tmp_path
    assert cases["no_shuttle"].q_initial == 0.0
    assert cases["no_shuttle"].q_final == 0.0
    assert cases["slow_shuttle"].duration > cases["fast_shuttle"].duration
    assert cases["fast_shuttle"].grid.num_points == 32
    assert cases["fast_shuttle"].coulomb.strength == 1.0
    assert cases["fast_shuttle"].initial_state == "localized_crystal_ground_state"
    assert cases["fast_shuttle"].mode_population_levels == 6
    assert cases["fast_shuttle"].resolved_mode_population_levels == (6, 6)

    args = runner._parse_args(  # noqa: SLF001
        [
            "--output-dir",
            str(tmp_path),
            "--grid-points",
            "32",
            "--com-mode-population-levels",
            "8",
            "--stretch-mode-population-levels",
            "3",
        ]
    )
    cases = runner._build_cases(args)  # noqa: SLF001
    assert cases["fast_shuttle"].resolved_mode_population_levels == (8, 3)


def test_two_ion_localized_product_state_breaks_marginal_symmetry() -> None:
    """A localized left-right product state should not force identical x1 and x2 marginals."""
    config = TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-5.0, x_max=5.0, num_points=32),
        q_initial=0.0,
        q_final=0.0,
        duration=0.0,
        dt=0.1,
        snapshot_stride=1,
        initial_state="localized_product_state",
        initial_x1=-1.0,
        initial_x2=1.0,
    )

    result = simulate_two_ion_shuttling(config)

    assert result.pair_density is not None
    dx = result.positions[1] - result.positions[0]
    rho1 = np.sum(result.pair_density[0], axis=1) * dx
    rho2 = np.sum(result.pair_density[0], axis=0) * dx

    assert result.x1_expectation[0] < result.x2_expectation[0]
    assert not np.allclose(rho1, rho2)


def test_two_ion_localized_crystal_state_is_correlated_and_lower_energy_than_product() -> None:
    """The crystal initializer should add normal-mode correlations around ordered centers."""
    grid = SpatialGrid(x_min=-6.0, x_max=6.0, num_points=36)
    coulomb = CoulombInteraction(strength=8.0, softening_length=0.1)
    product = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(
            grid=grid,
            omega=0.5,
            q_initial=0.0,
            q_final=0.0,
            duration=0.0,
            snapshot_stride=1,
            coulomb=coulomb,
            initial_state="localized_product_state",
            initial_x1=-2.0,
            initial_x2=2.0,
        )
    )
    crystal = simulate_two_ion_shuttling(
        TwoIonShuttlingConfig(
            grid=grid,
            omega=0.5,
            q_initial=0.0,
            q_final=0.0,
            duration=0.0,
            snapshot_stride=1,
            coulomb=coulomb,
            initial_state="localized_crystal_ground_state",
            initial_x1=-2.0,
            initial_x2=2.0,
            mode_population_levels=3,
        )
    )

    assert product.pair_density is not None
    assert crystal.pair_density is not None
    dx = crystal.positions[1] - crystal.positions[0]
    x1_grid, x2_grid = np.meshgrid(crystal.positions, crystal.positions, indexing="ij")
    crystal_covariance = float(
        np.sum(
            (x1_grid - crystal.x1_expectation[0])
            * (x2_grid - crystal.x2_expectation[0])
            * crystal.pair_density[0]
        )
        * dx**2
    )
    product_covariance = float(
        np.sum(
            (x1_grid - product.x1_expectation[0])
            * (x2_grid - product.x2_expectation[0])
            * product.pair_density[0]
        )
        * dx**2
    )

    assert crystal.x1_expectation[0] < crystal.x2_expectation[0]
    assert abs(crystal_covariance) > abs(product_covariance) + 1e-2
    assert crystal.excess_quanta[0] < product.excess_quanta[0]
    assert crystal.stretch_mode_excitation[0] < product.stretch_mode_excitation[0]
    assert crystal.com_mode_excitation[0] < 1e-2
    assert crystal.stretch_mode_excitation[0] < 1e-2
    assert crystal.com_mode_populations is not None
    assert crystal.stretch_mode_populations is not None
    assert crystal.com_mode_population_excitation is not None
    assert crystal.stretch_mode_population_excitation is not None
    assert crystal.com_mode_populations[0, 0] > 0.9
    assert crystal.stretch_mode_populations[0, 0] > 0.9
    assert crystal.com_mode_population_excitation[0] < 1e-3
    assert crystal.stretch_mode_population_excitation[0] < 1e-3


def test_two_ion_visualization_helpers_derive_marginals_and_write_outputs(tmp_path: Path) -> None:
    """Two-ion visualization helpers should expose marginal snapshots and runner artifacts."""
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None

    from mqt.yaqs.applications.shuttling.visualization import (
        _two_ion_collective_density,
        _two_ion_marginal_densities,
        animate_two_ion_marginals,
        plot_joint_mode_populations,
        plot_mode_excitation_trace,
        plot_mode_population_consistency,
        plot_mode_populations,
        plot_normal_mode_phase_spaces,
        plot_two_ion_marginal_trajectories,
    )

    config = TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-5.0, x_max=5.0, num_points=24),
        q_initial=-0.5,
        q_final=0.5,
        duration=0.2,
        dt=0.1,
        snapshot_stride=1,
        initial_state="localized_product_state",
        mode_population_levels=3,
    )
    result = simulate_two_ion_shuttling(config)
    assert result.snapshot_times is not None
    assert result.pair_density is not None

    x1_density, x2_density = _two_ion_marginal_densities(result, result.pair_density)
    collective_density = _two_ion_collective_density(result, result.pair_density)
    dx = result.positions[1] - result.positions[0]

    assert x1_density.shape == (result.snapshot_times.size, result.positions.size)
    assert x2_density.shape == (result.snapshot_times.size, result.positions.size)
    assert collective_density.shape == (result.snapshot_times.size, result.positions.size)
    assert np.allclose(np.sum(x1_density, axis=1) * dx, 1.0, atol=1e-6)
    assert np.allclose(np.sum(x2_density, axis=1) * dx, 1.0, atol=1e-6)
    assert np.allclose(np.sum(collective_density, axis=1) * dx, 1.0, atol=5e-3)

    fig, _axes = plot_two_ion_marginal_trajectories(result)
    fig.canvas.draw()
    fig, _ax = plot_mode_excitation_trace(result)
    fig.canvas.draw()
    fig, _axes = plot_mode_populations(result)
    fig.canvas.draw()
    fig, _axes = plot_joint_mode_populations(result)
    fig.canvas.draw()
    fig, _axes = plot_mode_population_consistency(result)
    fig.canvas.draw()
    fig, _axes = plot_normal_mode_phase_spaces(result)
    fig.canvas.draw()
    animation = animate_two_ion_marginals(result)
    assert len(list(animation.new_frame_seq())) == result.snapshot_times.size

    runner = _load_runner("two_ion_shuttling.py")
    runner._save_static_plots(tmp_path, "case", result, dpi=72)  # noqa: SLF001
    runner._save_animations(tmp_path, "case", result)  # noqa: SLF001

    assert (tmp_path / "case_x1_x2_density_trajectories.png").is_file()
    assert (tmp_path / "case_mode_populations.png").is_file()
    assert (tmp_path / "case_joint_mode_populations.png").is_file()
    assert (tmp_path / "case_mode_population_consistency.png").is_file()
    assert (tmp_path / "case_x1_x2_marginals_animation.html").is_file()


def _test_ms_config(*, sideband_model: Literal["finite_lamb_dicke", "linear"] = "finite_lamb_dicke") -> MSGateConfig:
    return MSGateConfig(
        modes=default_ms_modes(com_cutoff=8, stretch_cutoff=8),
        dt=0.08,
        sideband_model=sideband_model,
        calibration_scan_points=9,
        calibration_amplitude_bounds=(0.0, 8.0),
    )


def test_collective_motional_state_constructors_normalize_and_validate() -> None:
    """Collective motional state helpers should produce normalized Fock-basis states."""
    ground = collective_ground_state(4, 3)
    fock = collective_fock_state(4, 3, 2, 1)
    populations = np.zeros((4, 3), dtype=np.float64)
    populations[0, 0] = 2.0
    populations[2, 1] = 1.0
    mixture = collective_diagonal_mixture(populations)

    assert ground.cutoffs == (4, 3)
    assert ground.amplitudes is not None
    assert np.isclose(np.vdot(ground.amplitudes, ground.amplitudes).real, 1.0)
    assert fock.amplitudes is not None
    assert fock.amplitudes[2, 1] == 1.0
    assert mixture.populations is not None
    assert np.isclose(np.sum(mixture.populations), 1.0)

    with pytest.raises(ValueError, match="outside cutoff"):
        collective_fock_state(4, 3, 4, 0)


def test_ms_gate_cold_ground_state_calibrates_to_high_fidelity() -> None:
    """The finite-LD MS gate should be calibrated to the cold motional ground state."""
    result = simulate_ms_gate(_test_ms_config(), collective_ground_state(8, 8))

    assert result.target_fidelity[-1] > 0.999
    assert np.allclose(result.state_norm, 1.0, atol=1e-8)
    assert result.spin_density.shape == (result.times.size, 4, 4)
    assert result.spin_populations.shape == (result.times.size, 4)
    assert result.mode_expectations.shape == (result.times.size, 2)
    assert result.joint_mode_populations.shape == (result.times.size, 8, 8)
    assert np.all(np.isfinite(result.target_fidelity))
    assert np.all(np.isfinite(result.mode_expectations))


def test_ms_gate_finite_lamb_dicke_motion_reduces_fidelity() -> None:
    """Finite-LD sideband matrix elements should make hot motional states less ideal."""
    config = _test_ms_config()
    cold = simulate_ms_gate(config, collective_ground_state(8, 8))
    calibrated = replace(config, drive_amplitude=cold.calibrated_drive_amplitude, calibrate=False)
    com_hot = simulate_ms_gate(calibrated, collective_fock_state(8, 8, 4, 0))
    stretch_hot = simulate_ms_gate(calibrated, collective_fock_state(8, 8, 0, 5))
    populations = np.zeros((8, 8), dtype=np.float64)
    populations[0, 0] = 0.5
    populations[4, 0] = 0.25
    populations[0, 5] = 0.25
    mixed = simulate_ms_gate(calibrated, collective_diagonal_mixture(populations))

    assert com_hot.target_fidelity[-1] < cold.target_fidelity[-1] - 0.05
    assert stretch_hot.target_fidelity[-1] < cold.target_fidelity[-1] - 0.01
    assert mixed.target_fidelity[-1] < cold.target_fidelity[-1] - 0.02


def test_ms_gate_linear_closed_loop_is_motion_insensitive_for_low_levels() -> None:
    """The ideal linear closed-loop MS model should be nearly insensitive to low Fock levels."""
    config = _test_ms_config(sideband_model="linear")
    cold = simulate_ms_gate(config, collective_ground_state(8, 8))
    calibrated = replace(config, drive_amplitude=cold.calibrated_drive_amplitude, calibrate=False)
    com_one = simulate_ms_gate(calibrated, collective_fock_state(8, 8, 1, 0))
    stretch_one = simulate_ms_gate(calibrated, collective_fock_state(8, 8, 0, 1))

    assert cold.target_fidelity[-1] > 0.999
    assert abs(com_one.target_fidelity[-1] - cold.target_fidelity[-1]) < 3e-3
    assert abs(stretch_one.target_fidelity[-1] - cold.target_fidelity[-1]) < 3e-3


def test_ms_phase_space_loop_closes_and_short_gate_does_not() -> None:
    """The intended MS phase-space loop should close at matched detuning periods."""
    config = _test_ms_config(sideband_model="linear")
    closed = simulate_ms_gate(config, collective_ground_state(8, 8))
    short_config = replace(
        config,
        duration=0.85 * config.duration,
        drive_amplitude=closed.calibrated_drive_amplitude,
        calibrate=False,
    )
    short = simulate_ms_gate(short_config, collective_ground_state(8, 8))

    closed_loop = ms_phase_space_displacements(closed)
    short_loop = ms_phase_space_displacements(short)
    active_closed = np.max(np.abs(closed_loop.displacements), axis=0) > 1e-8

    assert closed_loop.branches == ((1, 1), (1, -1), (-1, 1), (-1, -1))
    assert closed_loop.displacements.shape == (closed.times.size, 2, 4)
    assert np.max(np.abs(closed_loop.displacements[-1][active_closed])) < 2e-3
    assert np.max(np.abs(short_loop.displacements[-1])) > 0.1


def test_two_ion_final_state_projects_to_collective_modes() -> None:
    """Stored two-ion grid states should be projectable into gate-ready collective modes."""
    config = TwoIonShuttlingConfig(
        grid=SpatialGrid(x_min=-5.0, x_max=5.0, num_points=28),
        q_initial=0.0,
        q_final=0.0,
        duration=0.0,
        snapshot_stride=None,
        initial_state="localized_crystal_ground_state",
        mode_population_levels=3,
        store_final_state=True,
    )

    result = simulate_two_ion_shuttling(config)
    assert result.final_state is not None
    projection = project_two_ion_state_to_collective_modes(result.final_state, config, levels=(3, 3))

    assert projection.amplitudes.shape == (3, 3)
    assert projection.populations.shape == (3, 3)
    assert projection.frequencies.shape == (2,)
    assert projection.residual >= 0.0
    assert np.all(np.isfinite(projection.amplitudes))
    assert np.all(np.isfinite(projection.populations))


def test_ms_gate_runner_writes_default_outputs(tmp_path: Path) -> None:
    """The MS-gate console runner should write arrays, summaries, and plots."""
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None

    runner = _load_runner("ms_gate.py")
    exit_code = runner.main(  # noqa: SLF001
        [
            "--output-dir",
            str(tmp_path),
            "--com-cutoff",
            "5",
            "--stretch-cutoff",
            "5",
            "--com-excited-level",
            "2",
            "--stretch-excited-level",
            "2",
            "--dt",
            "0.2",
            "--calibration-scan-points",
            "5",
            "--dpi",
            "72",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cold.npz").is_file()
    assert (tmp_path / "com_excited.npz").is_file()
    assert (tmp_path / "stretch_excited.npz").is_file()
    assert (tmp_path / "mixed.npz").is_file()
    assert (tmp_path / "imperfect_loop.npz").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "cold_fidelity.png").is_file()
    assert (tmp_path / "cold_phase_space_loops.png").is_file()
    assert (tmp_path / "final_fidelity_comparison.png").is_file()
    assert (tmp_path / "phase_space_loop_closed_vs_imperfect.png").is_file()


def _load_runner(filename: str = "single_ion_shuttling.py") -> ModuleType:
    path = REPO_ROOT / "scripts" / "shuttling" / filename
    spec = importlib.util.spec_from_file_location(f"{filename}_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
