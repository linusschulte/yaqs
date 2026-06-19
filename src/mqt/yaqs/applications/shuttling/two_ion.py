# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact two-coordinate grid simulation for two shuttled ions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg

from numpy.typing import NDArray

from ._numerics import (
    expectation,
    kinetic_operator_1d,
    momentum_operator_1d,
    normalize_grid_state,
    sample_times,
)
from .model import TwoIonShuttlingConfig, TwoIonShuttlingResult, Trajectory, quintic_trajectory


def simulate_two_ion_shuttling(config: TwoIonShuttlingConfig | None = None) -> TwoIonShuttlingResult:
    """Simulate two ions in a moving harmonic well with softened Coulomb repulsion.

    Args:
        config: Two-ion simulation configuration. Defaults to :class:`TwoIonShuttlingConfig`.

    Returns:
        Time traces and optional pair-density snapshots for the two-ion trajectory.
    """
    cfg = TwoIonShuttlingConfig() if config is None else config
    trajectory = _resolve_trajectory(cfg)

    positions = cfg.grid.positions
    dx = cfg.grid.dx
    area_element = dx**2
    times = sample_times(cfg.duration + cfg.trail_time, cfg.dt)
    operators = _build_two_ion_operators(cfg, dx)
    normal_modes = _build_normal_mode_frame(cfg, positions, operators)
    coulomb_values = _coulomb_values(cfg, positions)
    static_part = operators.kinetic + scipy.sparse.diags(coulomb_values.ravel(), format="csr")

    ground_energy_reference, psi = _initialize_state(
        cfg,
        positions,
        static_part,
        initial_centers=(normal_modes.reference_x1, normal_modes.reference_x2),
        area_element=area_element,
    )

    n_steps = times.size
    trap_center = np.empty(n_steps, dtype=np.float64)
    x1_expectation = np.empty(n_steps, dtype=np.float64)
    x2_expectation = np.empty(n_steps, dtype=np.float64)
    center_of_mass = np.empty(n_steps, dtype=np.float64)
    mean_separation = np.empty(n_steps, dtype=np.float64)
    equilibrium_x1 = np.empty(n_steps, dtype=np.float64)
    equilibrium_x2 = np.empty(n_steps, dtype=np.float64)
    p1_expectation = np.empty(n_steps, dtype=np.float64)
    p2_expectation = np.empty(n_steps, dtype=np.float64)
    center_of_mass_momentum = np.empty(n_steps, dtype=np.float64)
    mode_coordinate = np.empty((2, n_steps), dtype=np.float64)
    mode_coordinate_squared = np.empty((2, n_steps), dtype=np.float64)
    mode_momentum = np.empty((2, n_steps), dtype=np.float64)
    mode_momentum_squared = np.empty((2, n_steps), dtype=np.float64)
    energy = np.empty(n_steps, dtype=np.float64)
    excess_energy = np.empty(n_steps, dtype=np.float64)
    excess_quanta = np.empty(n_steps, dtype=np.float64)
    norm = np.empty(n_steps, dtype=np.float64)
    com_population_levels, stretch_population_levels = cfg.resolved_mode_population_levels
    com_mode_populations = (
        np.empty((n_steps, com_population_levels), dtype=np.float64)
        if com_population_levels > 0 and stretch_population_levels > 0
        else None
    )
    stretch_mode_populations = (
        np.empty((n_steps, stretch_population_levels), dtype=np.float64)
        if com_population_levels > 0 and stretch_population_levels > 0
        else None
    )
    joint_mode_populations = (
        np.empty((n_steps, com_population_levels, stretch_population_levels), dtype=np.float64)
        if com_population_levels > 0 and stretch_population_levels > 0
        else None
    )
    mode_population_residual = (
        np.empty(n_steps, dtype=np.float64)
        if com_population_levels > 0 and stretch_population_levels > 0
        else None
    )

    snapshot_indices = _snapshot_indices(n_steps, cfg.snapshot_stride)
    pair_density = (
        np.empty((snapshot_indices.size, positions.size, positions.size), dtype=np.float64)
        if snapshot_indices is not None
        else None
    )
    snapshot_cursor = 0

    snapshot_cursor = _record_step(
        psi,
        positions,
        times[0],
        trajectory,
        static_part,
        operators,
        normal_modes,
        cfg,
        ground_energy_reference,
        trap_center,
        x1_expectation,
        x2_expectation,
        center_of_mass,
        mean_separation,
        equilibrium_x1,
        equilibrium_x2,
        p1_expectation,
        p2_expectation,
        center_of_mass_momentum,
        mode_coordinate,
        mode_coordinate_squared,
        mode_momentum,
        mode_momentum_squared,
        energy,
        excess_energy,
        excess_quanta,
        norm,
        com_mode_populations,
        stretch_mode_populations,
        joint_mode_populations,
        mode_population_residual,
        pair_density,
        snapshot_indices,
        snapshot_cursor,
        step_index=0,
    )

    for step_index, (t_start, t_end) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
        step_dt = t_end - t_start
        midpoint = t_start + 0.5 * step_dt
        center = trajectory(midpoint)
        hamiltonian = static_part + scipy.sparse.diags(_trap_values(cfg, positions, center).ravel(), format="csr")
        psi = scipy.sparse.linalg.expm_multiply((-1j * step_dt / cfg.hbar) * hamiltonian, psi)
        snapshot_cursor = _record_step(
            psi,
            positions,
            t_end,
            trajectory,
            static_part,
            operators,
            normal_modes,
            cfg,
            ground_energy_reference,
            trap_center,
            x1_expectation,
            x2_expectation,
            center_of_mass,
            mean_separation,
            equilibrium_x1,
            equilibrium_x2,
            p1_expectation,
            p2_expectation,
            center_of_mass_momentum,
            mode_coordinate,
            mode_coordinate_squared,
            mode_momentum,
            mode_momentum_squared,
            energy,
            excess_energy,
            excess_quanta,
            norm,
            com_mode_populations,
            stretch_mode_populations,
            joint_mode_populations,
            mode_population_residual,
            pair_density,
            snapshot_indices,
            snapshot_cursor,
            step_index=step_index,
        )

    trap_velocity_values = _trajectory_velocity_values(trajectory, times, dt=cfg.dt)
    total_mass = cfg.ion1.mass + cfg.ion2.mass
    relative_center_of_mass_momentum = center_of_mass_momentum - total_mass * trap_velocity_values
    equilibrium_mode_momentum = normal_modes.equilibrium_velocity_coefficients[:, None] * trap_velocity_values[None, :]
    relative_mode_momentum = mode_momentum - equilibrium_mode_momentum
    mode_excitation = _mode_excitations(
        mode_coordinate_squared,
        mode_momentum,
        mode_momentum_squared,
        equilibrium_mode_momentum,
        norm,
        normal_modes.frequencies,
        hbar=cfg.hbar,
    )
    population_excitation = (
        _population_mode_excitations(joint_mode_populations)
        if joint_mode_populations is not None
        else None
    )
    snapshot_times = None if snapshot_indices is None else times[snapshot_indices]

    return TwoIonShuttlingResult(
        config=cfg,
        positions=positions,
        times=times,
        trap_center=trap_center,
        trap_velocity=trap_velocity_values,
        x1_expectation=x1_expectation,
        x2_expectation=x2_expectation,
        center_of_mass=center_of_mass,
        mean_separation=mean_separation,
        equilibrium_x1=equilibrium_x1,
        equilibrium_x2=equilibrium_x2,
        p1_expectation=p1_expectation,
        p2_expectation=p2_expectation,
        center_of_mass_momentum=center_of_mass_momentum,
        relative_center_of_mass_momentum=relative_center_of_mass_momentum,
        com_mode_frequency=float(normal_modes.frequencies[0]),
        stretch_mode_frequency=float(normal_modes.frequencies[1]),
        com_mode_coordinate=mode_coordinate[0],
        stretch_mode_coordinate=mode_coordinate[1],
        com_mode_momentum=mode_momentum[0],
        stretch_mode_momentum=mode_momentum[1],
        relative_com_mode_momentum=relative_mode_momentum[0],
        relative_stretch_mode_momentum=relative_mode_momentum[1],
        com_mode_excitation=mode_excitation[0],
        stretch_mode_excitation=mode_excitation[1],
        energy=energy,
        ground_energy_reference=ground_energy_reference,
        excess_energy=excess_energy,
        excess_quanta=excess_quanta,
        norm=norm,
        snapshot_times=snapshot_times,
        pair_density=pair_density,
        com_mode_populations=com_mode_populations,
        stretch_mode_populations=stretch_mode_populations,
        joint_mode_populations=joint_mode_populations,
        mode_population_residual=mode_population_residual,
        com_mode_population_excitation=None if population_excitation is None else population_excitation[0],
        stretch_mode_population_excitation=None if population_excitation is None else population_excitation[1],
        final_state=psi.copy() if cfg.store_final_state else None,
    )


@dataclass(frozen=True)
class CollectiveModeProjection:
    """Projection of a two-ion grid state into a local COM/stretch oscillator basis."""

    amplitudes: NDArray[np.complex128]
    populations: NDArray[np.float64]
    residual: float
    frequencies: NDArray[np.float64]
    reference_positions: tuple[float, float]


def project_two_ion_state_to_collective_modes(
    psi: NDArray[np.complex128],
    config: TwoIonShuttlingConfig,
    *,
    levels: tuple[int, int],
    trap_center: float | None = None,
    trap_velocity_value: float = 0.0,
) -> CollectiveModeProjection:
    """Project a two-ion grid wavefunction onto the local COM/stretch Fock basis.

    Args:
        psi: Flattened ``psi(x1, x2)`` grid wavefunction.
        config: Two-ion shuttling configuration defining the grid and local mode frame.
        levels: Number of COM-like and stretch-like oscillator levels to retain.
        trap_center: Trap center for the projection. Defaults to ``config.q_final``.
        trap_velocity_value: Moving-frame trap velocity used to remove the coherent boost.

    Returns:
        Truncated complex amplitudes, populations, residual probability, and mode metadata.
    """
    if levels[0] <= 0 or levels[1] <= 0:
        msg = f"levels must be positive for both modes, got {levels!r}."
        raise ValueError(msg)
    expected_size = config.grid.num_points**2
    if psi.shape != (expected_size,):
        msg = f"psi must have shape ({expected_size},), got {psi.shape}."
        raise ValueError(msg)

    positions = config.grid.positions
    operators = _build_two_ion_operators(config, config.grid.dx)
    normal_modes = _build_normal_mode_frame(config, positions, operators)
    center = config.q_final if trap_center is None else trap_center
    center_shift = center - config.q_initial
    equilibrium_x1 = normal_modes.reference_x1 + center_shift
    equilibrium_x2 = normal_modes.reference_x2 + center_shift
    area_element = config.grid.dx**2
    amplitudes = _normal_mode_amplitudes(
        psi,
        operators,
        normal_modes,
        equilibrium_x1,
        equilibrium_x2,
        trap_velocity_value,
        config,
        levels=levels,
        area_element=area_element,
    )
    populations = np.abs(amplitudes) ** 2
    norm = float(np.vdot(psi, psi).real * area_element)
    residual = max(0.0, norm - float(np.sum(populations)))
    return CollectiveModeProjection(
        amplitudes=amplitudes,
        populations=populations,
        residual=residual,
        frequencies=normal_modes.frequencies.copy(),
        reference_positions=(equilibrium_x1, equilibrium_x2),
    )


@dataclass(frozen=True)
class _TwoIonOperators:
    """Sparse operators and flattened coordinate grids for two-ion evolution."""

    kinetic: scipy.sparse.csr_matrix
    p1: scipy.sparse.csr_matrix
    p2: scipy.sparse.csr_matrix
    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    separation: NDArray[np.float64]


@dataclass(frozen=True)
class _NormalModeFrame:
    """Local quadratic normal-mode frame around the ordered crystal equilibrium."""

    reference_x1: float
    reference_x2: float
    frequencies: NDArray[np.float64]
    coordinate_coefficients: NDArray[np.float64]
    equilibrium_velocity_coefficients: NDArray[np.float64]
    momentum_operators: tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]
    momentum_squared_operators: tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]


def _resolve_trajectory(config: TwoIonShuttlingConfig) -> Trajectory:
    if config.trajectory is not None:
        return config.trajectory
    return quintic_trajectory(config.q_initial, config.q_final, config.duration)


def _trajectory_velocity_values(
    trajectory: Trajectory,
    times: NDArray[np.float64],
    *,
    dt: float,
) -> NDArray[np.float64]:
    return np.asarray([_trajectory_velocity(trajectory, time, dt=dt) for time in times])


def _trajectory_velocity(
    trajectory: Trajectory,
    time: float,
    *,
    dt: float,
) -> float:
    epsilon = max(1e-6, 1e-4 * dt)
    return float((trajectory(time + epsilon) - trajectory(time - epsilon)) / (2.0 * epsilon))


def _build_two_ion_operators(config: TwoIonShuttlingConfig, dx: float) -> _TwoIonOperators:
    n_points = config.grid.num_points
    identity = scipy.sparse.identity(n_points, format="csr", dtype=np.complex128)
    k1 = kinetic_operator_1d(num_points=n_points, dx=dx, mass=config.ion1.mass, hbar=config.hbar)
    k2 = kinetic_operator_1d(num_points=n_points, dx=dx, mass=config.ion2.mass, hbar=config.hbar)
    p = momentum_operator_1d(num_points=n_points, dx=dx, hbar=config.hbar)
    kinetic = scipy.sparse.kron(k1, identity, format="csr") + scipy.sparse.kron(identity, k2, format="csr")
    p1 = scipy.sparse.kron(p, identity, format="csr")
    p2 = scipy.sparse.kron(identity, p, format="csr")
    positions = config.grid.positions
    x1_grid, x2_grid = np.meshgrid(positions, positions, indexing="ij")
    return _TwoIonOperators(
        kinetic=cast("scipy.sparse.csr_matrix", kinetic),
        p1=cast("scipy.sparse.csr_matrix", p1),
        p2=cast("scipy.sparse.csr_matrix", p2),
        x1=x1_grid.ravel(),
        x2=x2_grid.ravel(),
        separation=np.abs(x2_grid - x1_grid).ravel(),
    )


def _build_normal_mode_frame(
    config: TwoIonShuttlingConfig,
    positions: NDArray[np.float64],
    operators: _TwoIonOperators,
) -> _NormalModeFrame:
    x1_center, x2_center = _resolve_localized_initial_centers(config, positions)
    hessian = _potential_hessian_at_centers(config, x1_center, x2_center)
    masses = np.array([config.ion1.mass, config.ion2.mass], dtype=np.float64)
    mass_sqrt = np.diag(np.sqrt(masses))
    mass_inv_sqrt = np.diag(1.0 / np.sqrt(masses))
    dynamical_matrix = mass_inv_sqrt @ hessian @ mass_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)
    if np.any(eigenvalues <= 0.0):
        msg = f"normal-mode expansion requires stable frequencies, got {eigenvalues!r}."
        raise ValueError(msg)

    coordinate_coefficients = eigenvectors.T @ mass_sqrt
    momentum_coefficients = eigenvectors.T @ mass_inv_sqrt
    p_mode_0 = cast(
        "scipy.sparse.csr_matrix",
        momentum_coefficients[0, 0] * operators.p1 + momentum_coefficients[0, 1] * operators.p2,
    )
    p_mode_1 = cast(
        "scipy.sparse.csr_matrix",
        momentum_coefficients[1, 0] * operators.p1 + momentum_coefficients[1, 1] * operators.p2,
    )
    p_mode_squared_0 = cast("scipy.sparse.csr_matrix", p_mode_0 @ p_mode_0)
    p_mode_squared_1 = cast("scipy.sparse.csr_matrix", p_mode_1 @ p_mode_1)
    return _NormalModeFrame(
        reference_x1=x1_center,
        reference_x2=x2_center,
        frequencies=np.sqrt(eigenvalues),
        coordinate_coefficients=coordinate_coefficients,
        equilibrium_velocity_coefficients=np.sum(coordinate_coefficients, axis=1),
        momentum_operators=(p_mode_0, p_mode_1),
        momentum_squared_operators=(p_mode_squared_0, p_mode_squared_1),
    )


def _coulomb_values(config: TwoIonShuttlingConfig, positions: NDArray[np.float64]) -> NDArray[np.float64]:
    x1_grid, x2_grid = np.meshgrid(positions, positions, indexing="ij")
    softening = config.coulomb.softening_length
    if softening is None:
        softening = config.grid.dx
    return config.coulomb.strength / np.sqrt((x1_grid - x2_grid) ** 2 + softening**2)


def _trap_values(config: TwoIonShuttlingConfig, positions: NDArray[np.float64], center: float) -> NDArray[np.float64]:
    x1_grid, x2_grid = np.meshgrid(positions, positions, indexing="ij")
    v1 = 0.5 * config.ion1.mass * config.omega**2 * (x1_grid - center) ** 2
    v2 = 0.5 * config.ion2.mass * config.omega**2 * (x2_grid - center) ** 2
    return v1 + v2


def _initialize_state(
    config: TwoIonShuttlingConfig,
    positions: NDArray[np.float64],
    static_part: scipy.sparse.csr_matrix,
    *,
    initial_centers: tuple[float, float],
    area_element: float,
) -> tuple[float, NDArray[np.complex128]]:
    h_initial = static_part + scipy.sparse.diags(
        _trap_values(config, positions, config.q_initial).ravel(),
        format="csr",
    )
    ground_energy_reference, ground_state = _ground_state(h_initial, area_element=area_element)
    if config.initial_state == "collective_ground_state":
        return ground_energy_reference, ground_state

    x1_center, x2_center = initial_centers
    if config.initial_state == "localized_crystal_ground_state":
        return ground_energy_reference, _localized_crystal_ground_state(config, positions, x1_center, x2_center)

    psi1 = _harmonic_ground_state(positions, x1_center, config.ion1.mass, config.omega, config.hbar)
    psi2 = _harmonic_ground_state(positions, x2_center, config.ion2.mass, config.omega, config.hbar)
    psi = np.outer(psi1, psi2).ravel()
    psi = normalize_grid_state(psi.astype(np.complex128), volume_element=area_element)
    return ground_energy_reference, psi


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


def _localized_crystal_ground_state(
    config: TwoIonShuttlingConfig,
    positions: NDArray[np.float64],
    x1_center: float,
    x2_center: float,
) -> NDArray[np.complex128]:
    """Approximate the ordered crystal ground state from local normal modes."""
    hessian = _potential_hessian_at_centers(config, x1_center, x2_center)
    masses = np.array([config.ion1.mass, config.ion2.mass], dtype=np.float64)
    mass_sqrt = np.diag(np.sqrt(masses))
    mass_inv_sqrt = np.diag(1.0 / np.sqrt(masses))
    dynamical_matrix = mass_inv_sqrt @ hessian @ mass_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(dynamical_matrix)
    if np.any(eigenvalues <= 0.0):
        msg = (
            "localized_crystal_ground_state requires a stable quadratic expansion; "
            f"got normal-mode frequency squares {eigenvalues!r}."
        )
        raise ValueError(msg)

    mode_frequencies = np.sqrt(eigenvalues)
    width_matrix = mass_sqrt @ eigenvectors @ np.diag(mode_frequencies) @ eigenvectors.T @ mass_sqrt
    x1_grid, x2_grid = np.meshgrid(positions, positions, indexing="ij")
    dx1 = x1_grid - x1_center
    dx2 = x2_grid - x2_center
    exponent = (
        width_matrix[0, 0] * dx1**2
        + 2.0 * width_matrix[0, 1] * dx1 * dx2
        + width_matrix[1, 1] * dx2**2
    ) / (2.0 * config.hbar)
    psi = np.exp(-exponent).ravel()
    return normalize_grid_state(psi.astype(np.complex128), volume_element=(positions[1] - positions[0]) ** 2)


def _potential_hessian_at_centers(
    config: TwoIonShuttlingConfig,
    x1_center: float,
    x2_center: float,
) -> NDArray[np.float64]:
    softening = config.coulomb.softening_length if config.coulomb.softening_length is not None else config.grid.dx
    separation = x1_center - x2_center
    denominator = (separation**2 + softening**2) ** 2.5
    coulomb_curvature = (
        config.coulomb.strength * (2.0 * separation**2 - softening**2) / denominator
        if denominator > 0.0
        else 0.0
    )
    return np.array(
        [
            [config.ion1.mass * config.omega**2 + coulomb_curvature, -coulomb_curvature],
            [-coulomb_curvature, config.ion2.mass * config.omega**2 + coulomb_curvature],
        ],
        dtype=np.float64,
    )


def _mode_excitations(
    mode_coordinate_squared: NDArray[np.float64],
    mode_momentum: NDArray[np.float64],
    mode_momentum_squared: NDArray[np.float64],
    equilibrium_mode_momentum: NDArray[np.float64],
    norm: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    *,
    hbar: float,
) -> NDArray[np.float64]:
    relative_momentum_squared = (
        mode_momentum_squared - 2.0 * equilibrium_mode_momentum * mode_momentum + equilibrium_mode_momentum**2 * norm
    )
    oscillator_energy = 0.5 * (relative_momentum_squared + frequencies[:, None] ** 2 * mode_coordinate_squared)
    excitation = oscillator_energy / (hbar * frequencies[:, None]) - 0.5 * norm
    return np.maximum(excitation, 0.0)


def _population_mode_excitations(
    joint_mode_populations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    com_levels = np.arange(joint_mode_populations.shape[1], dtype=np.float64)
    stretch_levels = np.arange(joint_mode_populations.shape[2], dtype=np.float64)
    com_excitation = np.sum(joint_mode_populations * com_levels[None, :, None], axis=(1, 2))
    stretch_excitation = np.sum(joint_mode_populations * stretch_levels[None, None, :], axis=(1, 2))
    return com_excitation, stretch_excitation


def _normal_mode_populations(
    psi: NDArray[np.complex128],
    operators: _TwoIonOperators,
    normal_modes: _NormalModeFrame,
    equilibrium_x1: float,
    equilibrium_x2: float,
    trap_velocity_value: float,
    config: TwoIonShuttlingConfig,
    *,
    levels: tuple[int, int],
    area_element: float,
) -> NDArray[np.float64]:
    amplitudes = _normal_mode_amplitudes(
        psi,
        operators,
        normal_modes,
        equilibrium_x1,
        equilibrium_x2,
        trap_velocity_value,
        config,
        levels=levels,
        area_element=area_element,
    )
    return np.abs(amplitudes) ** 2


def _normal_mode_amplitudes(
    psi: NDArray[np.complex128],
    operators: _TwoIonOperators,
    normal_modes: _NormalModeFrame,
    equilibrium_x1: float,
    equilibrium_x2: float,
    trap_velocity_value: float,
    config: TwoIonShuttlingConfig,
    *,
    levels: tuple[int, int],
    area_element: float,
) -> NDArray[np.complex128]:
    displacements = np.vstack((operators.x1 - equilibrium_x1, operators.x2 - equilibrium_x2))
    coordinates = normal_modes.coordinate_coefficients @ displacements
    equilibrium_momentum = normal_modes.equilibrium_velocity_coefficients * trap_velocity_value
    phase = np.exp(
        1j
        * (
            equilibrium_momentum[0] * coordinates[0]
            + equilibrium_momentum[1] * coordinates[1]
        )
        / config.hbar
    )
    basis_0 = _harmonic_oscillator_mode_basis(
        coordinates[0],
        normal_modes.frequencies[0],
        config.hbar,
        levels[0],
    )
    basis_1 = _harmonic_oscillator_mode_basis(
        coordinates[1],
        normal_modes.frequencies[1],
        config.hbar,
        levels[1],
    )
    weighted_psi = np.conjugate(phase) * psi
    overlaps = (np.conjugate(basis_0) * weighted_psi[None, :]) @ np.conjugate(basis_1).T
    overlaps *= area_element
    norm_matrix = np.sqrt((np.abs(basis_0) ** 2) @ (np.abs(basis_1) ** 2).T * area_element)
    return np.divide(overlaps, norm_matrix, out=np.zeros_like(overlaps), where=norm_matrix > 0.0)


def _harmonic_oscillator_mode_basis(
    coordinates: NDArray[np.float64],
    frequency: float,
    hbar: float,
    levels: int,
) -> NDArray[np.complex128]:
    oscillator_length = math.sqrt(hbar / frequency)
    scaled = coordinates / oscillator_length
    basis = np.empty((levels, coordinates.size), dtype=np.complex128)
    if levels == 0:
        return basis

    basis[0] = np.pi ** (-0.25) * np.exp(-0.5 * scaled**2) / math.sqrt(oscillator_length)
    if levels == 1:
        return basis

    basis[1] = math.sqrt(2.0) * scaled * basis[0]
    for level in range(1, levels - 1):
        basis[level + 1] = (
            math.sqrt(2.0 / (level + 1)) * scaled * basis[level]
            - math.sqrt(level / (level + 1)) * basis[level - 1]
        )
    return basis


def _resolve_localized_initial_centers(
    config: TwoIonShuttlingConfig, positions: NDArray[np.float64]
) -> tuple[float, float]:
    if config.initial_x1 is not None and config.initial_x2 is not None:
        return float(config.initial_x1), float(config.initial_x2)

    softening = config.coulomb.softening_length if config.coulomb.softening_length is not None else config.grid.dx
    lower_bound = positions[0]
    upper_bound = positions[-1]
    initial_offset = max(config.grid.dx, softening)
    initial_guess = np.array(
        [max(lower_bound, config.q_initial - initial_offset), min(upper_bound, config.q_initial + initial_offset)],
        dtype=np.float64,
    )

    def objective(coordinates: NDArray[np.float64]) -> float:
        x1, x2 = coordinates
        trap = 0.5 * config.ion1.mass * config.omega**2 * (x1 - config.q_initial) ** 2
        trap += 0.5 * config.ion2.mass * config.omega**2 * (x2 - config.q_initial) ** 2
        coulomb = config.coulomb.strength / math.sqrt((x1 - x2) ** 2 + softening**2)
        ordering_penalty = (
            0.0
            if x1 <= x2
            else (x1 - x2) ** 2 * max(config.ion1.mass, config.ion2.mass) * config.omega**2
        )
        return trap + coulomb + ordering_penalty

    result = scipy.optimize.minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=[(lower_bound, upper_bound), (lower_bound, upper_bound)],
    )
    centers = result.x if result.success and np.all(np.isfinite(result.x)) else initial_guess
    x1_center, x2_center = float(centers[0]), float(centers[1])
    return (x1_center, x2_center) if x1_center <= x2_center else (x2_center, x1_center)


def _ground_state(hamiltonian: scipy.sparse.spmatrix, *, area_element: float) -> tuple[float, NDArray[np.complex128]]:
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(hamiltonian, k=1, which="SA")
    psi = normalize_grid_state(eigenvectors[:, 0].astype(np.complex128), volume_element=area_element)
    return float(eigenvalues[0]), psi


def _snapshot_indices(n_steps: int, snapshot_stride: int | None) -> NDArray[np.int64] | None:
    if snapshot_stride is None:
        return None
    indices = np.arange(0, n_steps, snapshot_stride, dtype=np.int64)
    if indices[-1] != n_steps - 1:
        indices = np.append(indices, n_steps - 1)
    return indices


def _record_step(
    psi: NDArray[np.complex128],
    positions: NDArray[np.float64],
    time: float,
    trajectory: Trajectory,
    static_part: scipy.sparse.csr_matrix,
    operators: _TwoIonOperators,
    normal_modes: _NormalModeFrame,
    config: TwoIonShuttlingConfig,
    ground_energy: float,
    trap_center: NDArray[np.float64],
    x1_expectation: NDArray[np.float64],
    x2_expectation: NDArray[np.float64],
    center_of_mass: NDArray[np.float64],
    mean_separation: NDArray[np.float64],
    equilibrium_x1: NDArray[np.float64],
    equilibrium_x2: NDArray[np.float64],
    p1_expectation: NDArray[np.float64],
    p2_expectation: NDArray[np.float64],
    center_of_mass_momentum: NDArray[np.float64],
    mode_coordinate: NDArray[np.float64],
    mode_coordinate_squared: NDArray[np.float64],
    mode_momentum: NDArray[np.float64],
    mode_momentum_squared: NDArray[np.float64],
    energy: NDArray[np.float64],
    excess_energy: NDArray[np.float64],
    excess_quanta: NDArray[np.float64],
    norm: NDArray[np.float64],
    com_mode_populations: NDArray[np.float64] | None,
    stretch_mode_populations: NDArray[np.float64] | None,
    joint_mode_populations: NDArray[np.float64] | None,
    mode_population_residual: NDArray[np.float64] | None,
    pair_density: NDArray[np.float64] | None,
    snapshot_indices: NDArray[np.int64] | None,
    snapshot_cursor: int,
    *,
    step_index: int,
) -> int:
    dx = positions[1] - positions[0]
    area_element = dx**2
    density = np.abs(psi) ** 2
    center = trajectory(time)
    trap = _trap_values(config, positions, center).ravel()
    hamiltonian = static_part + scipy.sparse.diags(trap, format="csr")
    current_energy = expectation(psi, hamiltonian, volume_element=area_element).real
    x1_mean = float(np.sum(operators.x1 * density) * area_element)
    x2_mean = float(np.sum(operators.x2 * density) * area_element)
    total_mass = config.ion1.mass + config.ion2.mass
    p1_mean = expectation(psi, operators.p1, volume_element=area_element).real
    p2_mean = expectation(psi, operators.p2, volume_element=area_element).real
    center_shift = center - config.q_initial
    eq_x1 = normal_modes.reference_x1 + center_shift
    eq_x2 = normal_modes.reference_x2 + center_shift

    trap_center[step_index] = center
    x1_expectation[step_index] = x1_mean
    x2_expectation[step_index] = x2_mean
    center_of_mass[step_index] = (config.ion1.mass * x1_mean + config.ion2.mass * x2_mean) / total_mass
    mean_separation[step_index] = float(np.sum(operators.separation * density) * area_element)
    equilibrium_x1[step_index] = eq_x1
    equilibrium_x2[step_index] = eq_x2
    p1_expectation[step_index] = p1_mean
    p2_expectation[step_index] = p2_mean
    center_of_mass_momentum[step_index] = p1_mean + p2_mean
    displacements = np.vstack((operators.x1 - eq_x1, operators.x2 - eq_x2))
    for mode_index in range(2):
        coordinate_values = normal_modes.coordinate_coefficients[mode_index] @ displacements
        mode_coordinate[mode_index, step_index] = float(np.sum(coordinate_values * density) * area_element)
        mode_coordinate_squared[mode_index, step_index] = float(
            np.sum(coordinate_values**2 * density) * area_element
        )
        mode_momentum[mode_index, step_index] = expectation(
            psi,
            normal_modes.momentum_operators[mode_index],
            volume_element=area_element,
        ).real
        mode_momentum_squared[mode_index, step_index] = expectation(
            psi,
            normal_modes.momentum_squared_operators[mode_index],
            volume_element=area_element,
        ).real
    energy[step_index] = current_energy
    excess_energy[step_index] = max(0.0, current_energy - ground_energy)
    excess_quanta[step_index] = excess_energy[step_index] / (config.hbar * config.omega)
    norm[step_index] = float(np.sum(density) * area_element)
    if (
        com_mode_populations is not None
        and stretch_mode_populations is not None
        and joint_mode_populations is not None
        and mode_population_residual is not None
    ):
        qdot = _trajectory_velocity(trajectory, time, dt=config.dt)
        joint_populations = _normal_mode_populations(
            psi,
            operators,
            normal_modes,
            eq_x1,
            eq_x2,
            qdot,
            config,
            levels=joint_mode_populations.shape[1:],
            area_element=area_element,
        )
        joint_mode_populations[step_index] = joint_populations
        com_mode_populations[step_index] = np.sum(joint_populations, axis=1)
        stretch_mode_populations[step_index] = np.sum(joint_populations, axis=0)
        mode_population_residual[step_index] = max(0.0, norm[step_index] - float(np.sum(joint_populations)))

    if pair_density is not None and snapshot_indices is not None and step_index == snapshot_indices[snapshot_cursor]:
        pair_density[snapshot_cursor] = density.reshape(positions.size, positions.size)
        snapshot_cursor += 1
    return snapshot_cursor
