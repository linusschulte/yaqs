# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Finite-difference simulation of a single ion in a moving harmonic well."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.special import eval_hermite

from numpy.typing import NDArray

from ._numerics import (
    expectation,
    kinetic_operator_1d,
    momentum_operator_1d,
    normalize_grid_state,
    operator_std,
    sample_times,
    trap_velocity,
)
from .model import (
    MovingHarmonicPotential,
    Potential1D,
    ShuttlingConfig,
    ShuttlingResult,
    Trajectory,
    quintic_trajectory,
)


def simulate_shuttling(config: ShuttlingConfig | None = None) -> ShuttlingResult:
    """Simulate coherent single-ion shuttling in a one-dimensional moving potential.

    Args:
        config: Simulation configuration. Defaults to :class:`ShuttlingConfig`.

    Returns:
        Time traces and grid-resolved diagnostics for the shuttling schedule.
    """
    cfg = ShuttlingConfig() if config is None else config
    trajectory = _resolve_trajectory(cfg)
    potential = cfg.potential or MovingHarmonicPotential(cfg.ion, cfg.omega, trajectory)

    positions = cfg.grid.positions
    dx = cfg.grid.dx
    times = sample_times(cfg.duration + cfg.trail_time, cfg.dt)
    kinetic = kinetic_operator_1d(num_points=cfg.grid.num_points, dx=dx, mass=cfg.ion.mass, hbar=cfg.hbar)
    momentum = momentum_operator_1d(num_points=cfg.grid.num_points, dx=dx, hbar=cfg.hbar)

    psi = _harmonic_ground_state(positions, cfg.q_initial, cfg.ion.mass, cfg.omega, cfg.hbar)
    density = np.empty((times.size, positions.size), dtype=np.float64)
    potential_energy = np.empty_like(density)
    trap_center = np.empty(times.size, dtype=np.float64)
    position_expectation = np.empty(times.size, dtype=np.float64)
    position_std = np.empty(times.size, dtype=np.float64)
    momentum_expectation = np.empty(times.size, dtype=np.float64)
    momentum_std = np.empty(times.size, dtype=np.float64)
    excitation_number = np.empty(times.size, dtype=np.float64)
    norm = np.empty(times.size, dtype=np.float64)
    populations = (
        np.empty((times.size, cfg.population_levels), dtype=np.float64) if cfg.population_levels > 0 else None
    )

    _record_step(
        psi,
        positions,
        times[0],
        trajectory,
        potential,
        kinetic,
        momentum,
        cfg,
        density,
        potential_energy,
        trap_center,
        position_expectation,
        position_std,
        momentum_expectation,
        momentum_std,
        excitation_number,
        norm,
        populations,
        step_index=0,
    )

    for step_index, (t_start, t_end) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
        step_dt = t_end - t_start
        midpoint = t_start + 0.5 * step_dt
        hamiltonian = kinetic + scipy.sparse.diags(potential(positions, midpoint), format="csr")
        psi = scipy.sparse.linalg.expm_multiply((-1j * step_dt / cfg.hbar) * hamiltonian, psi)
        _record_step(
            psi,
            positions,
            t_end,
            trajectory,
            potential,
            kinetic,
            momentum,
            cfg,
            density,
            potential_energy,
            trap_center,
            position_expectation,
            position_std,
            momentum_expectation,
            momentum_std,
            excitation_number,
            norm,
            populations,
            step_index=step_index,
        )

    trap_velocity_values = trap_velocity(times, trap_center)
    relative_momentum_expectation = momentum_expectation - cfg.ion.mass * trap_velocity_values

    return ShuttlingResult(
        config=cfg,
        positions=positions,
        times=times,
        trap_center=trap_center,
        trap_velocity=trap_velocity_values,
        position_expectation=position_expectation,
        position_std=position_std,
        momentum_expectation=momentum_expectation,
        relative_momentum_expectation=relative_momentum_expectation,
        momentum_std=momentum_std,
        excitation_number=excitation_number,
        density=density,
        potential_energy=potential_energy,
        norm=norm,
        populations=populations,
    )


def _resolve_trajectory(config: ShuttlingConfig) -> Trajectory:
    if config.trajectory is not None:
        return config.trajectory
    return quintic_trajectory(config.q_initial, config.q_final, config.duration)


def _harmonic_ground_state(
    positions: NDArray[np.float64],
    center: float,
    mass: float,
    omega: float,
    hbar: float,
) -> NDArray[np.complex128]:
    oscillator_length = math.sqrt(hbar / (mass * omega))
    scaled = (positions - center) / oscillator_length
    psi = np.exp(-0.5 * scaled**2) / (np.pi**0.25 * math.sqrt(oscillator_length))
    return normalize_grid_state(psi.astype(np.complex128), volume_element=positions[1] - positions[0])


def _record_step(
    psi: NDArray[np.complex128],
    positions: NDArray[np.float64],
    time: float,
    trajectory: Trajectory,
    potential: Potential1D,
    kinetic: scipy.sparse.csr_matrix,
    momentum: scipy.sparse.csr_matrix,
    config: ShuttlingConfig,
    density: NDArray[np.float64],
    potential_energy: NDArray[np.float64],
    trap_center: NDArray[np.float64],
    position_expectation: NDArray[np.float64],
    position_std: NDArray[np.float64],
    momentum_expectation: NDArray[np.float64],
    momentum_std: NDArray[np.float64],
    excitation_number: NDArray[np.float64],
    norm: NDArray[np.float64],
    populations: NDArray[np.float64] | None,
    *,
    step_index: int,
) -> None:
    dx = positions[1] - positions[0]
    prob_density = np.abs(psi) ** 2
    current_norm = float(np.sum(prob_density) * dx)
    center = trajectory(time)
    potential_values = potential(positions, time)
    reference_potential = 0.5 * config.ion.mass * config.omega**2 * (positions - center) ** 2
    h_osc = kinetic + scipy.sparse.diags(reference_potential, format="csr")
    energy = expectation(psi, h_osc, volume_element=dx).real

    density[step_index] = prob_density
    potential_energy[step_index] = potential_values
    trap_center[step_index] = center
    x_mean = float(np.sum(positions * prob_density) * dx)
    x2_mean = float(np.sum(positions**2 * prob_density) * dx)
    p_mean = expectation(psi, momentum, volume_element=dx).real
    position_expectation[step_index] = x_mean
    position_std[step_index] = math.sqrt(max(0.0, x2_mean - x_mean**2))
    momentum_expectation[step_index] = p_mean
    momentum_std[step_index] = operator_std(psi, momentum, p_mean, volume_element=dx)
    excitation_number[step_index] = max(0.0, energy / (config.hbar * config.omega) - 0.5)
    norm[step_index] = current_norm

    if populations is not None:
        populations[step_index] = _oscillator_populations(
            psi,
            positions,
            center,
            config.ion.mass,
            config.omega,
            config.hbar,
            populations.shape[1],
        )


def _oscillator_populations(
    psi: NDArray[np.complex128],
    positions: NDArray[np.float64],
    center: float,
    mass: float,
    omega: float,
    hbar: float,
    levels: int,
) -> NDArray[np.float64]:
    dx = positions[1] - positions[0]
    populations = np.empty(levels, dtype=np.float64)
    for level in range(levels):
        basis_state = _harmonic_oscillator_state(positions, center, mass, omega, hbar, level)
        amplitude = np.vdot(basis_state, psi) * dx
        populations[level] = float(np.abs(amplitude) ** 2)
    return populations


def _harmonic_oscillator_state(
    positions: NDArray[np.float64],
    center: float,
    mass: float,
    omega: float,
    hbar: float,
    level: int,
) -> NDArray[np.complex128]:
    oscillator_length = math.sqrt(hbar / (mass * omega))
    scaled = (positions - center) / oscillator_length
    normalization = 1.0 / math.sqrt((2**level) * math.factorial(level) * math.sqrt(np.pi) * oscillator_length)
    state = normalization * eval_hermite(level, scaled) * np.exp(-0.5 * scaled**2)
    return normalize_grid_state(state.astype(np.complex128), volume_element=positions[1] - positions[0])
