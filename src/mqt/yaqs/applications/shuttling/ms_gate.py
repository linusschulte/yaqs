# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analog two-qubit Mølmer-Sørensen gate in a collective motional Fock basis."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.special

from numpy.typing import NDArray

from mqt.yaqs.core.libraries.gate_library import Rxx

from ._numerics import sample_times

SidebandModel = Literal["finite_lamb_dicke", "linear"]


@dataclass(frozen=True)
class MotionalModeSpec:
    """One collective motional mode coupled to the MS force."""

    name: str
    frequency: float
    detuning: float
    cutoff: int
    lamb_dicke: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate mode parameters."""
        if not self.name:
            msg = "mode name must be non-empty."
            raise ValueError(msg)
        for name, value in {"frequency": self.frequency, "detuning": self.detuning}.items():
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value!r}."
                raise ValueError(msg)
        if self.frequency <= 0.0:
            msg = f"frequency must be positive, got {self.frequency!r}."
            raise ValueError(msg)
        if self.detuning == 0.0:
            msg = "detuning must be non-zero."
            raise ValueError(msg)
        if self.cutoff < 2:
            msg = f"cutoff must be at least 2, got {self.cutoff}."
            raise ValueError(msg)
        if len(self.lamb_dicke) != 2 or not all(np.isfinite(value) for value in self.lamb_dicke):
            msg = f"lamb_dicke must contain two finite ion couplings, got {self.lamb_dicke!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class CollectiveMotionalState:
    """Pure or diagonal mixed motional state in the COM/stretch Fock basis."""

    amplitudes: NDArray[np.complex128] | None = None
    populations: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the motional state."""
        if (self.amplitudes is None) == (self.populations is None):
            msg = "provide exactly one of amplitudes or populations."
            raise ValueError(msg)
        if self.amplitudes is not None:
            amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
            if amplitudes.ndim != 2:
                msg = f"amplitudes must be a 2D array, got shape {amplitudes.shape}."
                raise ValueError(msg)
            norm = math.sqrt(float(np.vdot(amplitudes, amplitudes).real))
            if norm == 0.0 or not np.isfinite(norm):
                msg = "amplitudes must have finite non-zero norm."
                raise ValueError(msg)
            object.__setattr__(self, "amplitudes", amplitudes / norm)
        if self.populations is not None:
            populations = np.asarray(self.populations, dtype=np.float64)
            if populations.ndim != 2:
                msg = f"populations must be a 2D array, got shape {populations.shape}."
                raise ValueError(msg)
            if np.any(populations < 0.0) or not np.all(np.isfinite(populations)):
                msg = "populations must be finite and non-negative."
                raise ValueError(msg)
            total = float(np.sum(populations))
            if total == 0.0:
                msg = "populations must contain non-zero weight."
                raise ValueError(msg)
            object.__setattr__(self, "populations", populations / total)

    @property
    def cutoffs(self) -> tuple[int, int]:
        """Return COM and stretch cutoffs."""
        data = self.amplitudes if self.amplitudes is not None else self.populations
        if data is None:  # pragma: no cover - guarded by validation.
            msg = "motional state is not initialized."
            raise ValueError(msg)
        return int(data.shape[0]), int(data.shape[1])

    @property
    def is_pure(self) -> bool:
        """Whether this state is represented by amplitudes."""
        return self.amplitudes is not None

    def components(self, *, threshold: float = 1e-14) -> Iterator[tuple[float, NDArray[np.complex128]]]:
        """Yield pure-state components and weights for propagation."""
        if self.amplitudes is not None:
            yield 1.0, self.amplitudes
            return
        if self.populations is None:  # pragma: no cover - guarded by validation.
            return
        for com_level, stretch_level in np.argwhere(self.populations > threshold):
            amplitudes = np.zeros_like(self.populations, dtype=np.complex128)
            amplitudes[com_level, stretch_level] = 1.0
            yield float(self.populations[com_level, stretch_level]), amplitudes


@dataclass(frozen=True)
class MSGateConfig:
    """Configuration for an analog two-qubit MS gate simulation."""

    modes: tuple[MotionalModeSpec, MotionalModeSpec] = field(default_factory=lambda: default_ms_modes())
    duration: float = 2.0 * math.pi
    dt: float = 0.05
    hbar: float = 1.0
    target_angle: float = math.pi / 2.0
    spin_phase: float = 0.0
    initial_spin_state: str = "00"
    sideband_model: SidebandModel = "finite_lamb_dicke"
    drive_amplitude: float | None = None
    calibrate: bool = True
    ion_drive_amplitudes: tuple[float, float] = (1.0, 1.0)
    drive_envelope: Callable[[float], float] | None = None
    calibration_scan_points: int = 13
    calibration_amplitude_bounds: tuple[float, float] = (0.0, 8.0)
    calibration_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        """Validate MS-gate configuration."""
        if len(self.modes) != 2:
            msg = f"exactly two motional modes are required, got {len(self.modes)}."
            raise ValueError(msg)
        for name, value in {
            "duration": self.duration,
            "dt": self.dt,
            "hbar": self.hbar,
            "target_angle": self.target_angle,
            "spin_phase": self.spin_phase,
        }.items():
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value!r}."
                raise ValueError(msg)
        if self.duration <= 0.0:
            msg = f"duration must be positive, got {self.duration!r}."
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = f"dt must be positive, got {self.dt!r}."
            raise ValueError(msg)
        if self.hbar <= 0.0:
            msg = f"hbar must be positive, got {self.hbar!r}."
            raise ValueError(msg)
        if self.sideband_model not in {"finite_lamb_dicke", "linear"}:
            msg = f"unsupported sideband_model {self.sideband_model!r}."
            raise ValueError(msg)
        if self.drive_amplitude is not None and (
            self.drive_amplitude < 0.0 or not np.isfinite(self.drive_amplitude)
        ):
            msg = f"drive_amplitude must be finite and non-negative, got {self.drive_amplitude!r}."
            raise ValueError(msg)
        if len(self.ion_drive_amplitudes) != 2 or not all(np.isfinite(value) for value in self.ion_drive_amplitudes):
            msg = f"ion_drive_amplitudes must contain two finite values, got {self.ion_drive_amplitudes!r}."
            raise ValueError(msg)
        if self.calibration_scan_points < 3:
            msg = f"calibration_scan_points must be at least 3, got {self.calibration_scan_points}."
            raise ValueError(msg)
        lower, upper = self.calibration_amplitude_bounds
        if lower < 0.0 or upper <= lower or not np.isfinite(lower) or not np.isfinite(upper):
            msg = f"invalid calibration_amplitude_bounds {self.calibration_amplitude_bounds!r}."
            raise ValueError(msg)
        if self.calibration_tolerance <= 0.0 or not np.isfinite(self.calibration_tolerance):
            msg = f"calibration_tolerance must be positive, got {self.calibration_tolerance!r}."
            raise ValueError(msg)


@dataclass(frozen=True)
class MSGateResult:
    """Recorded output from an analog MS-gate simulation."""

    config: MSGateConfig
    motional_state: CollectiveMotionalState
    times: NDArray[np.float64]
    calibrated_drive_amplitude: float
    calibration_fidelity: float
    target_spin_state: NDArray[np.complex128]
    spin_density: NDArray[np.complex128]
    spin_populations: NDArray[np.float64]
    target_fidelity: NDArray[np.float64]
    mode_expectations: NDArray[np.float64]
    joint_mode_populations: NDArray[np.float64]
    state_norm: NDArray[np.float64]
    final_state: NDArray[np.complex128] | None = None


@dataclass(frozen=True)
class MSPhaseSpaceLoop:
    """Branch-resolved ideal MS displacement loops in motional phase space."""

    branches: tuple[tuple[int, int], ...]
    displacements: NDArray[np.complex128]


@dataclass(frozen=True)
class _MSGateOperators:
    """Prebuilt sparse operators for MS propagation."""

    couplings: tuple[tuple[float, scipy.sparse.csr_matrix], ...]
    shape: tuple[int, int]


def default_ms_modes(*, com_cutoff: int = 12, stretch_cutoff: int = 12) -> tuple[MotionalModeSpec, MotionalModeSpec]:
    """Return dimensionless default COM/stretch mode specifications."""
    return (
        MotionalModeSpec(
            name="COM",
            frequency=1.0,
            detuning=-1.0,
            cutoff=com_cutoff,
            lamb_dicke=(0.10, 0.10),
        ),
        MotionalModeSpec(
            name="stretch",
            frequency=math.sqrt(3.0),
            detuning=-2.0,
            cutoff=stretch_cutoff,
            lamb_dicke=(0.08, -0.08),
        ),
    )


def collective_ground_state(com_cutoff: int, stretch_cutoff: int) -> CollectiveMotionalState:
    """Construct ``|0_COM, 0_stretch>``."""
    return collective_fock_state(com_cutoff, stretch_cutoff, 0, 0)


def collective_fock_state(
    com_cutoff: int,
    stretch_cutoff: int,
    com_level: int,
    stretch_level: int,
) -> CollectiveMotionalState:
    """Construct one pure collective Fock state."""
    if not 0 <= com_level < com_cutoff:
        msg = f"com_level {com_level} is outside cutoff {com_cutoff}."
        raise ValueError(msg)
    if not 0 <= stretch_level < stretch_cutoff:
        msg = f"stretch_level {stretch_level} is outside cutoff {stretch_cutoff}."
        raise ValueError(msg)
    amplitudes = np.zeros((com_cutoff, stretch_cutoff), dtype=np.complex128)
    amplitudes[com_level, stretch_level] = 1.0
    return CollectiveMotionalState(amplitudes=amplitudes)


def collective_diagonal_mixture(populations: NDArray[np.float64]) -> CollectiveMotionalState:
    """Construct a diagonal mixed motional state from joint Fock populations."""
    return CollectiveMotionalState(populations=populations)


def collective_thermal_state(
    com_cutoff: int,
    stretch_cutoff: int,
    *,
    mean_com: float,
    mean_stretch: float,
) -> CollectiveMotionalState:
    """Construct a truncated product thermal state over the two collective modes."""
    if mean_com < 0.0 or mean_stretch < 0.0:
        msg = f"thermal occupations must be non-negative, got {mean_com!r}, {mean_stretch!r}."
        raise ValueError(msg)
    com_distribution = _thermal_distribution(com_cutoff, mean_com)
    stretch_distribution = _thermal_distribution(stretch_cutoff, mean_stretch)
    return CollectiveMotionalState(populations=np.outer(com_distribution, stretch_distribution))


def collective_state_from_two_ion_projection(amplitudes: NDArray[np.complex128]) -> CollectiveMotionalState:
    """Convert projected COM/stretch amplitudes into a gate-ready pure motional state."""
    return CollectiveMotionalState(amplitudes=amplitudes)


def simulate_ms_gate(
    config: MSGateConfig | None = None,
    motional_state: CollectiveMotionalState | None = None,
) -> MSGateResult:
    """Simulate a two-qubit MS gate coupled to COM/stretch motional modes.

    Args:
        config: MS-gate configuration. Defaults to :class:`MSGateConfig`.
        motional_state: Initial collective motional state. Defaults to the cold ground state.

    Returns:
        Time-resolved spin fidelities, spin populations, and mode diagnostics.
    """
    cfg = MSGateConfig() if config is None else config
    state = motional_state if motional_state is not None else collective_ground_state(*_mode_cutoffs(cfg))
    _validate_state_matches_config(state, cfg)
    target_spin_state = _target_spin_state(cfg)
    operators = _build_ms_gate_operators(cfg)
    calibrated_amplitude = _resolve_drive_amplitude(cfg, operators, target_spin_state)
    calibration_fidelity = _final_fidelity(
        cfg,
        collective_ground_state(*_mode_cutoffs(cfg)),
        operators,
        target_spin_state,
        calibrated_amplitude,
    )
    return _propagate_ms_gate(
        cfg,
        state,
        operators,
        target_spin_state,
        calibrated_amplitude,
        calibration_fidelity=calibration_fidelity,
        record=True,
    )


def ms_phase_space_displacements(result: MSGateResult) -> MSPhaseSpaceLoop:
    """Compute ideal spin-branch MS phase-space displacements.

    The returned loop is the linear spin-dependent displacement implied by
    the configured force. In the finite-Lamb-Dicke model this is still the
    intended loop geometry, while the actual sideband matrix elements become
    motional-level dependent.

    Args:
        result: MS-gate simulation result.

    Returns:
        Spin-branch labels and complex displacements with shape
        ``(n_times, n_modes, n_branches)``.
    """
    branches = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    displacements = np.zeros((result.times.size, len(result.config.modes), len(branches)), dtype=np.complex128)
    for step_index, (t_start, t_end) in enumerate(zip(result.times[:-1], result.times[1:], strict=True), start=1):
        step_dt = t_end - t_start
        midpoint = t_start + 0.5 * step_dt
        envelope = 1.0 if result.config.drive_envelope is None else float(result.config.drive_envelope(midpoint))
        displacements[step_index] = displacements[step_index - 1]
        for mode_index, mode in enumerate(result.config.modes):
            phase = np.exp(1j * mode.detuning * midpoint)
            for branch_index, branch in enumerate(branches):
                branch_coupling = sum(
                    result.config.ion_drive_amplitudes[ion_index] * mode.lamb_dicke[ion_index] * branch[ion_index]
                    for ion_index in range(2)
                )
                displacements[step_index, mode_index, branch_index] += (
                    -1j * result.calibrated_drive_amplitude * envelope * branch_coupling * phase * step_dt
                )
    return MSPhaseSpaceLoop(branches=branches, displacements=displacements)


def _resolve_drive_amplitude(
    config: MSGateConfig,
    operators: _MSGateOperators,
    target_spin_state: NDArray[np.complex128],
) -> float:
    if config.drive_amplitude is not None and not config.calibrate:
        return config.drive_amplitude
    if config.drive_amplitude is not None and config.calibrate:
        center = config.drive_amplitude
        width = max(0.25 * center, 0.1)
        lower = max(0.0, center - width)
        upper = center + width
    else:
        lower, upper = config.calibration_amplitude_bounds

    ground = collective_ground_state(*_mode_cutoffs(config))
    scan_points = np.linspace(lower, upper, config.calibration_scan_points)
    fidelities = np.array([
        _final_fidelity(config, ground, operators, target_spin_state, float(amplitude))
        for amplitude in scan_points
    ])
    best_index = int(np.argmax(fidelities))
    left_index = max(0, best_index - 1)
    right_index = min(scan_points.size - 1, best_index + 1)
    refine_lower = float(scan_points[left_index])
    refine_upper = float(scan_points[right_index])
    if refine_lower == refine_upper:
        return float(scan_points[best_index])

    def objective(amplitude: float) -> float:
        return -_final_fidelity(config, ground, operators, target_spin_state, amplitude)

    optimum = scipy.optimize.minimize_scalar(
        objective,
        bounds=(refine_lower, refine_upper),
        method="bounded",
        options={"xatol": config.calibration_tolerance},
    )
    if not optimum.success:
        return float(scan_points[best_index])
    return float(optimum.x)


def _final_fidelity(
    config: MSGateConfig,
    motional_state: CollectiveMotionalState,
    operators: _MSGateOperators,
    target_spin_state: NDArray[np.complex128],
    drive_amplitude: float,
) -> float:
    result = _propagate_ms_gate(
        config,
        motional_state,
        operators,
        target_spin_state,
        drive_amplitude,
        calibration_fidelity=float("nan"),
        record=False,
    )
    return float(result.target_fidelity[-1])


def _propagate_ms_gate(
    config: MSGateConfig,
    motional_state: CollectiveMotionalState,
    operators: _MSGateOperators,
    target_spin_state: NDArray[np.complex128],
    drive_amplitude: float,
    *,
    calibration_fidelity: float,
    record: bool,
) -> MSGateResult:
    times = sample_times(config.duration, config.dt)
    com_cutoff, stretch_cutoff = _mode_cutoffs(config)
    spin_density = np.zeros((times.size, 4, 4), dtype=np.complex128)
    spin_populations = np.zeros((times.size, 4), dtype=np.float64)
    target_fidelity = np.zeros(times.size, dtype=np.float64)
    mode_expectations = np.zeros((times.size, 2), dtype=np.float64)
    joint_mode_populations = np.zeros((times.size, com_cutoff, stretch_cutoff), dtype=np.float64)
    state_norm = np.zeros(times.size, dtype=np.float64)
    final_state: NDArray[np.complex128] | None = None
    initial_spin = _spin_state_vector(config.initial_spin_state)

    for weight, motional_amplitudes in motional_state.components():
        component_state = _tensor_initial_state(initial_spin, motional_amplitudes)
        if record:
            _accumulate_ms_step(
                component_state,
                weight,
                target_spin_state,
                spin_density,
                spin_populations,
                target_fidelity,
                mode_expectations,
                joint_mode_populations,
                state_norm,
                step_index=0,
            )
        for step_index, (t_start, t_end) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
            step_dt = t_end - t_start
            midpoint = t_start + 0.5 * step_dt
            hamiltonian = _ms_hamiltonian(config, operators, midpoint, drive_amplitude)
            component_state = scipy.sparse.linalg.expm_multiply(
                (-1j * step_dt / config.hbar) * hamiltonian,
                component_state,
            )
            if record:
                _accumulate_ms_step(
                    component_state,
                    weight,
                    target_spin_state,
                    spin_density,
                    spin_populations,
                    target_fidelity,
                    mode_expectations,
                    joint_mode_populations,
                    state_norm,
                    step_index=step_index,
                )
        if not record:
            _accumulate_ms_step(
                component_state,
                weight,
                target_spin_state,
                spin_density,
                spin_populations,
                target_fidelity,
                mode_expectations,
                joint_mode_populations,
                state_norm,
                step_index=0,
            )
        if motional_state.is_pure:
            final_state = component_state.copy()

    if not record:
        spin_density = spin_density[:1]
        spin_populations = spin_populations[:1]
        target_fidelity = target_fidelity[:1]
        mode_expectations = mode_expectations[:1]
        joint_mode_populations = joint_mode_populations[:1]
        state_norm = state_norm[:1]
        times = np.array([config.duration], dtype=np.float64)

    return MSGateResult(
        config=config,
        motional_state=motional_state,
        times=times,
        calibrated_drive_amplitude=drive_amplitude,
        calibration_fidelity=calibration_fidelity,
        target_spin_state=target_spin_state,
        spin_density=spin_density,
        spin_populations=spin_populations,
        target_fidelity=np.clip(target_fidelity, 0.0, 1.0),
        mode_expectations=mode_expectations,
        joint_mode_populations=joint_mode_populations,
        state_norm=state_norm,
        final_state=final_state if motional_state.is_pure else None,
    )


def _build_ms_gate_operators(config: MSGateConfig) -> _MSGateOperators:
    com_cutoff, stretch_cutoff = _mode_cutoffs(config)
    identity_com = scipy.sparse.identity(com_cutoff, format="csr", dtype=np.complex128)
    identity_stretch = scipy.sparse.identity(stretch_cutoff, format="csr", dtype=np.complex128)
    spin_operators = _spin_operators(config.spin_phase)
    couplings: list[tuple[float, scipy.sparse.csr_matrix]] = []

    for mode_index, mode in enumerate(config.modes):
        for ion_index, eta in enumerate(mode.lamb_dicke):
            if eta == 0.0 or config.ion_drive_amplitudes[ion_index] == 0.0:
                continue
            sideband = _sideband_lowering_operator(mode.cutoff, eta, config.sideband_model)
            if mode_index == 0:
                mode_operator = scipy.sparse.kron(sideband, identity_stretch, format="csr")
            else:
                mode_operator = scipy.sparse.kron(identity_com, sideband, format="csr")
            coupling = scipy.sparse.kron(spin_operators[ion_index], mode_operator, format="csr")
            coupling *= config.ion_drive_amplitudes[ion_index]
            couplings.append((mode.detuning, cast("scipy.sparse.csr_matrix", coupling)))
    if not couplings:
        msg = "at least one non-zero ion-mode coupling is required."
        raise ValueError(msg)
    return _MSGateOperators(couplings=tuple(couplings), shape=(com_cutoff, stretch_cutoff))


def _ms_hamiltonian(
    config: MSGateConfig,
    operators: _MSGateOperators,
    time: float,
    drive_amplitude: float,
) -> scipy.sparse.csr_matrix:
    envelope = 1.0 if config.drive_envelope is None else float(config.drive_envelope(time))
    scale = config.hbar * drive_amplitude * envelope
    hamiltonian: scipy.sparse.csr_matrix | None = None
    for detuning, lowering in operators.couplings:
        phase = np.exp(-1j * detuning * time)
        term = phase * lowering + np.conjugate(phase) * lowering.getH()
        hamiltonian = cast("scipy.sparse.csr_matrix", term if hamiltonian is None else hamiltonian + term)
    if hamiltonian is None:  # pragma: no cover - guarded by operator construction.
        msg = "MS Hamiltonian has no coupling terms."
        raise ValueError(msg)
    return cast("scipy.sparse.csr_matrix", scale * hamiltonian)


def _sideband_lowering_operator(cutoff: int, eta: float, model: SidebandModel) -> scipy.sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[complex] = []
    eta_abs_squared = eta * eta
    for level in range(1, cutoff):
        if model == "linear":
            matrix_element = eta * math.sqrt(level)
        else:
            laguerre = scipy.special.eval_genlaguerre(level - 1, 1, eta_abs_squared)
            matrix_element = eta * math.exp(-0.5 * eta_abs_squared) * laguerre / math.sqrt(level)
        rows.append(level - 1)
        cols.append(level)
        values.append(complex(matrix_element))
    return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(cutoff, cutoff), dtype=np.complex128)


def _spin_operators(spin_phase: float) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    sigma_phi = math.cos(spin_phase) * x + math.sin(spin_phase) * y
    return (
        scipy.sparse.csr_matrix(np.kron(sigma_phi, identity)),
        scipy.sparse.csr_matrix(np.kron(identity, sigma_phi)),
    )


def _spin_state_vector(label: str) -> NDArray[np.complex128]:
    basis = {
        "00": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        "01": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
        "10": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
        "11": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128),
    }
    try:
        return basis[label]
    except KeyError as exc:
        msg = f"initial_spin_state must be one of {tuple(basis)}, got {label!r}."
        raise ValueError(msg) from exc


def _target_spin_state(config: MSGateConfig) -> NDArray[np.complex128]:
    initial_spin = _spin_state_vector(config.initial_spin_state)
    return Rxx([config.target_angle]).matrix @ initial_spin


def _tensor_initial_state(
    spin_state: NDArray[np.complex128],
    motional_amplitudes: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    state = spin_state[:, None, None] * motional_amplitudes[None, :, :]
    return state.ravel()


def _accumulate_ms_step(
    state: NDArray[np.complex128],
    weight: float,
    target_spin_state: NDArray[np.complex128],
    spin_density: NDArray[np.complex128],
    spin_populations: NDArray[np.float64],
    target_fidelity: NDArray[np.float64],
    mode_expectations: NDArray[np.float64],
    joint_mode_populations: NDArray[np.float64],
    state_norm: NDArray[np.float64],
    *,
    step_index: int,
) -> None:
    spin_count = 4
    com_cutoff = joint_mode_populations.shape[1]
    stretch_cutoff = joint_mode_populations.shape[2]
    tensor = state.reshape(spin_count, com_cutoff, stretch_cutoff)
    flattened_motion = tensor.reshape(spin_count, com_cutoff * stretch_cutoff)
    rho_spin = flattened_motion @ flattened_motion.conjugate().T
    probability = np.abs(tensor) ** 2
    joint_populations = np.sum(probability, axis=0)
    com_levels = np.arange(com_cutoff, dtype=np.float64)
    stretch_levels = np.arange(stretch_cutoff, dtype=np.float64)

    spin_density[step_index] += weight * rho_spin
    spin_populations[step_index] += weight * np.real(np.diag(rho_spin))
    target_fidelity[step_index] += weight * float(np.vdot(target_spin_state, rho_spin @ target_spin_state).real)
    mode_expectations[step_index, 0] += weight * float(np.sum(joint_populations * com_levels[:, None]))
    mode_expectations[step_index, 1] += weight * float(np.sum(joint_populations * stretch_levels[None, :]))
    joint_mode_populations[step_index] += weight * joint_populations
    state_norm[step_index] += weight * float(np.vdot(state, state).real)


def _validate_state_matches_config(state: CollectiveMotionalState, config: MSGateConfig) -> None:
    expected_cutoffs = _mode_cutoffs(config)
    if state.cutoffs != expected_cutoffs:
        msg = f"motional state cutoffs {state.cutoffs} do not match config cutoffs {expected_cutoffs}."
        raise ValueError(msg)


def _mode_cutoffs(config: MSGateConfig) -> tuple[int, int]:
    return config.modes[0].cutoff, config.modes[1].cutoff


def _thermal_distribution(cutoff: int, mean_occupation: float) -> NDArray[np.float64]:
    if mean_occupation == 0.0:
        distribution = np.zeros(cutoff, dtype=np.float64)
        distribution[0] = 1.0
        return distribution
    ratio = mean_occupation / (1.0 + mean_occupation)
    levels = np.arange(cutoff, dtype=np.float64)
    distribution = (1.0 - ratio) * ratio**levels
    return distribution / np.sum(distribution)
