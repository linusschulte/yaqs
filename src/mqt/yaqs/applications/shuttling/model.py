# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Data structures for one-dimensional single-ion shuttling simulations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from numpy.typing import NDArray

Trajectory = Callable[[float], float]
TwoIonInitialState = Literal[
    "collective_ground_state",
    "localized_product_state",
    "localized_crystal_ground_state",
]


class Potential1D(Protocol):
    """Time-dependent one-dimensional potential energy surface."""

    def __call__(self, positions: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        """Evaluate the potential on a spatial grid.

        Args:
            positions: Grid coordinates.
            time: Simulation time.

        Returns:
            Potential energy at each grid coordinate.
        """
        ...


@dataclass(frozen=True)
class Ion:
    """Physical parameters of a single trapped ion."""

    mass: float = 1.0

    def __post_init__(self) -> None:
        """Validate ion parameters."""
        if self.mass <= 0.0 or not np.isfinite(self.mass):
            msg = f"mass must be a finite positive value, got {self.mass!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class SpatialGrid:
    """Uniform one-dimensional position grid."""

    x_min: float = -8.0
    x_max: float = 8.0
    num_points: int = 256

    def __post_init__(self) -> None:
        """Validate grid parameters."""
        if self.num_points < 3:
            msg = f"num_points must be at least 3, got {self.num_points}."
            raise ValueError(msg)
        if not np.isfinite(self.x_min) or not np.isfinite(self.x_max) or self.x_min >= self.x_max:
            msg = f"x_min must be finite and smaller than x_max, got {self.x_min!r}, {self.x_max!r}."
            raise ValueError(msg)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Grid coordinates."""
        return np.linspace(self.x_min, self.x_max, self.num_points)

    @property
    def dx(self) -> float:
        """Uniform grid spacing."""
        return (self.x_max - self.x_min) / (self.num_points - 1)


@dataclass(frozen=True)
class MovingHarmonicPotential:
    """Moving harmonic potential for a single ion."""

    ion: Ion
    omega: float
    trajectory: Trajectory

    def __post_init__(self) -> None:
        """Validate potential parameters."""
        if self.omega <= 0.0 or not np.isfinite(self.omega):
            msg = f"omega must be a finite positive value, got {self.omega!r}."
            raise ValueError(msg)

    def __call__(self, positions: NDArray[np.float64], time: float) -> NDArray[np.float64]:
        """Evaluate the moving harmonic potential."""
        center = self.trajectory(time)
        return 0.5 * self.ion.mass * self.omega**2 * (positions - center) ** 2


@dataclass(frozen=True)
class ShuttlingConfig:
    """Numerical and physical configuration for a single-ion shuttling simulation."""

    ion: Ion = field(default_factory=Ion)
    grid: SpatialGrid = field(default_factory=SpatialGrid)
    omega: float = 1.0
    q_initial: float = -2.0
    q_final: float = 2.0
    duration: float = 8.0
    trail_time: float = 0.0
    dt: float = 0.02
    hbar: float = 1.0
    trajectory: Trajectory | None = None
    potential: Potential1D | None = None
    population_levels: int = 0

    def __post_init__(self) -> None:
        """Validate shuttling configuration."""
        for name, value in {
            "omega": self.omega,
            "duration": self.duration,
            "trail_time": self.trail_time,
            "dt": self.dt,
            "hbar": self.hbar,
            "q_initial": self.q_initial,
            "q_final": self.q_final,
        }.items():
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value!r}."
                raise ValueError(msg)
        if self.omega <= 0.0:
            msg = f"omega must be positive, got {self.omega!r}."
            raise ValueError(msg)
        if self.duration < 0.0:
            msg = f"duration must be non-negative, got {self.duration!r}."
            raise ValueError(msg)
        if self.trail_time < 0.0:
            msg = f"trail_time must be non-negative, got {self.trail_time!r}."
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = f"dt must be positive, got {self.dt!r}."
            raise ValueError(msg)
        if self.hbar <= 0.0:
            msg = f"hbar must be positive, got {self.hbar!r}."
            raise ValueError(msg)
        if self.population_levels < 0:
            msg = f"population_levels must be non-negative, got {self.population_levels}."
            raise ValueError(msg)


@dataclass(frozen=True)
class ShuttlingResult:
    """Recorded output from a single-ion shuttling simulation."""

    config: ShuttlingConfig
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    trap_center: NDArray[np.float64]
    trap_velocity: NDArray[np.float64]
    position_expectation: NDArray[np.float64]
    position_std: NDArray[np.float64]
    momentum_expectation: NDArray[np.float64]
    relative_momentum_expectation: NDArray[np.float64]
    momentum_std: NDArray[np.float64]
    excitation_number: NDArray[np.float64]
    density: NDArray[np.float64]
    potential_energy: NDArray[np.float64]
    norm: NDArray[np.float64]
    populations: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class CoulombInteraction:
    """Softened one-dimensional Coulomb interaction."""

    strength: float = 1.0
    softening_length: float | None = None

    def __post_init__(self) -> None:
        """Validate Coulomb parameters."""
        if not np.isfinite(self.strength):
            msg = f"strength must be finite, got {self.strength!r}."
            raise ValueError(msg)
        if self.softening_length is not None and (
            self.softening_length <= 0.0 or not np.isfinite(self.softening_length)
        ):
            msg = f"softening_length must be finite and positive, got {self.softening_length!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class TwoIonShuttlingConfig:
    """Configuration for two ions in one moving 1D harmonic well."""

    ion1: Ion = field(default_factory=Ion)
    ion2: Ion = field(default_factory=Ion)
    grid: SpatialGrid = field(default_factory=lambda: SpatialGrid(x_min=-8.0, x_max=8.0, num_points=96))
    omega: float = 1.0
    q_initial: float = -2.0
    q_final: float = 2.0
    duration: float = 8.0
    trail_time: float = 0.0
    dt: float = 0.02
    hbar: float = 1.0
    trajectory: Trajectory | None = None
    coulomb: CoulombInteraction = field(default_factory=CoulombInteraction)
    snapshot_stride: int | None = 5
    initial_state: TwoIonInitialState = "collective_ground_state"
    initial_x1: float | None = None
    initial_x2: float | None = None
    mode_population_levels: int = 0
    com_mode_population_levels: int | None = None
    stretch_mode_population_levels: int | None = None
    store_final_state: bool = False

    def __post_init__(self) -> None:
        """Validate two-ion shuttling configuration."""
        for name, value in {
            "omega": self.omega,
            "duration": self.duration,
            "trail_time": self.trail_time,
            "dt": self.dt,
            "hbar": self.hbar,
            "q_initial": self.q_initial,
            "q_final": self.q_final,
        }.items():
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value!r}."
                raise ValueError(msg)
        if self.omega <= 0.0:
            msg = f"omega must be positive, got {self.omega!r}."
            raise ValueError(msg)
        if self.duration < 0.0:
            msg = f"duration must be non-negative, got {self.duration!r}."
            raise ValueError(msg)
        if self.trail_time < 0.0:
            msg = f"trail_time must be non-negative, got {self.trail_time!r}."
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = f"dt must be positive, got {self.dt!r}."
            raise ValueError(msg)
        if self.hbar <= 0.0:
            msg = f"hbar must be positive, got {self.hbar!r}."
            raise ValueError(msg)
        if self.snapshot_stride is not None and self.snapshot_stride <= 0:
            msg = f"snapshot_stride must be positive or None, got {self.snapshot_stride!r}."
            raise ValueError(msg)
        if self.mode_population_levels < 0:
            msg = f"mode_population_levels must be non-negative, got {self.mode_population_levels}."
            raise ValueError(msg)
        for name, value in {
            "com_mode_population_levels": self.com_mode_population_levels,
            "stretch_mode_population_levels": self.stretch_mode_population_levels,
        }.items():
            if value is not None and value < 0:
                msg = f"{name} must be non-negative when provided, got {value}."
                raise ValueError(msg)
        com_levels, stretch_levels = self.resolved_mode_population_levels
        if (com_levels == 0) != (stretch_levels == 0):
            msg = (
                "normal-mode populations require both COM-like and stretch-like cutoffs to be positive, "
                "or both to be zero."
            )
            raise ValueError(msg)
        if self.initial_state not in {
            "collective_ground_state",
            "localized_product_state",
            "localized_crystal_ground_state",
        }:
            msg = f"initial_state must be a supported mode, got {self.initial_state!r}."
            raise ValueError(msg)
        for name, value in {"initial_x1": self.initial_x1, "initial_x2": self.initial_x2}.items():
            if value is not None and not np.isfinite(value):
                msg = f"{name} must be finite when provided, got {value!r}."
                raise ValueError(msg)
        if (self.initial_x1 is None) != (self.initial_x2 is None):
            msg = "initial_x1 and initial_x2 must either both be set or both be omitted."
            raise ValueError(msg)
        if self.initial_state == "collective_ground_state" and self.initial_x1 is not None:
            msg = (
                "initial_x1 and initial_x2 can only be used with "
                "initial_state='localized_product_state' or initial_state='localized_crystal_ground_state'."
            )
            raise ValueError(msg)

    @property
    def resolved_mode_population_levels(self) -> tuple[int, int]:
        """Return the effective COM-like and stretch-like population cutoffs."""
        com_levels = (
            self.com_mode_population_levels
            if self.com_mode_population_levels is not None
            else self.mode_population_levels
        )
        stretch_levels = (
            self.stretch_mode_population_levels
            if self.stretch_mode_population_levels is not None
            else self.mode_population_levels
        )
        return com_levels, stretch_levels


@dataclass(frozen=True)
class TwoIonShuttlingResult:
    """Recorded output from a two-ion shuttling simulation."""

    config: TwoIonShuttlingConfig
    positions: NDArray[np.float64]
    times: NDArray[np.float64]
    trap_center: NDArray[np.float64]
    trap_velocity: NDArray[np.float64]
    x1_expectation: NDArray[np.float64]
    x2_expectation: NDArray[np.float64]
    center_of_mass: NDArray[np.float64]
    mean_separation: NDArray[np.float64]
    equilibrium_x1: NDArray[np.float64]
    equilibrium_x2: NDArray[np.float64]
    p1_expectation: NDArray[np.float64]
    p2_expectation: NDArray[np.float64]
    center_of_mass_momentum: NDArray[np.float64]
    relative_center_of_mass_momentum: NDArray[np.float64]
    com_mode_frequency: float
    stretch_mode_frequency: float
    com_mode_coordinate: NDArray[np.float64]
    stretch_mode_coordinate: NDArray[np.float64]
    com_mode_momentum: NDArray[np.float64]
    stretch_mode_momentum: NDArray[np.float64]
    relative_com_mode_momentum: NDArray[np.float64]
    relative_stretch_mode_momentum: NDArray[np.float64]
    com_mode_excitation: NDArray[np.float64]
    stretch_mode_excitation: NDArray[np.float64]
    energy: NDArray[np.float64]
    ground_energy_reference: float
    excess_energy: NDArray[np.float64]
    excess_quanta: NDArray[np.float64]
    norm: NDArray[np.float64]
    snapshot_times: NDArray[np.float64] | None = None
    pair_density: NDArray[np.float64] | None = None
    com_mode_populations: NDArray[np.float64] | None = None
    stretch_mode_populations: NDArray[np.float64] | None = None
    joint_mode_populations: NDArray[np.float64] | None = None
    mode_population_residual: NDArray[np.float64] | None = None
    com_mode_population_excitation: NDArray[np.float64] | None = None
    stretch_mode_population_excitation: NDArray[np.float64] | None = None
    final_state: NDArray[np.complex128] | None = None


def quintic_trajectory(q_initial: float, q_final: float, duration: float) -> Trajectory:
    """Build a smooth quintic trap-center trajectory.

    Args:
        q_initial: Initial trap center.
        q_final: Final trap center.
        duration: Total shuttling duration.

    Returns:
        Callable mapping time to trap center.
    """
    if duration < 0.0 or not np.isfinite(duration):
        msg = f"duration must be finite and non-negative, got {duration!r}."
        raise ValueError(msg)
    if not np.isfinite(q_initial) or not np.isfinite(q_final):
        msg = f"trajectory endpoints must be finite, got {q_initial!r}, {q_final!r}."
        raise ValueError(msg)

    def trajectory(time: float) -> float:
        if duration == 0.0:
            return q_final
        s = np.clip(time / duration, 0.0, 1.0)
        blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        return float(q_initial + (q_final - q_initial) * blend)

    return trajectory
