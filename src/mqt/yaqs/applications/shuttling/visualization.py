# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Plotting helpers for single-ion and two-ion shuttling results."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model import ShuttlingResult, TwoIonShuttlingResult
    from .ms_gate import MSGateResult

    FuncAnimation: TypeAlias = Any
    Axes: TypeAlias = Any
    Figure: TypeAlias = Any


def animate_shuttling(
    result: ShuttlingResult,
    *,
    interval: int = 40,
    density_scale: float | None = None,
) -> FuncAnimation:
    """Animate the moving potential and wave-packet density.

    Args:
        result: Simulation output.
        interval: Delay between frames in milliseconds.
        density_scale: Optional scale factor applied to the probability density overlay.

    Returns:
        Matplotlib animation object.
    """
    pyplot, animation = _require_matplotlib()
    fig, ax = pyplot.subplots()

    potential = result.potential_energy
    density = result.density
    if density_scale is None:
        potential_span = float(np.max(potential) - np.min(potential))
        density_max = float(np.max(density))
        density_scale = 0.35 * potential_span / density_max if density_max > 0.0 else 1.0

    (potential_line,) = ax.plot(result.positions, potential[0], label="V(x,t)")
    (density_line,) = ax.plot(result.positions, density[0] * density_scale, label="|psi(x,t)|^2")
    trap_line = ax.axvline(result.trap_center[0], color="tab:green", linestyle="--", label="q(t)")
    expectation_line = ax.axvline(result.position_expectation[0], color="tab:red", linestyle=":", label="<x(t)>")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_xlabel("x")
    ax.set_ylabel("energy / scaled density")
    ax.set_xlim(result.positions[0], result.positions[-1])
    y_max = float(np.max(potential + density * density_scale))
    ax.set_ylim(0.0, y_max * 1.05 if y_max > 0.0 else 1.0)
    ax.legend(loc="upper right")

    def update(frame: int) -> tuple[Any, ...]:
        potential_line.set_ydata(potential[frame])
        density_line.set_ydata(density[frame] * density_scale)
        trap_line.set_xdata([result.trap_center[frame], result.trap_center[frame]])
        expectation_line.set_xdata([result.position_expectation[frame], result.position_expectation[frame]])
        time_text.set_text(f"t = {result.times[frame]:.3g}")
        return potential_line, density_line, trap_line, expectation_line, time_text

    return animation.FuncAnimation(fig, update, frames=result.times.size, interval=interval, blit=True)


def plot_position_trace(result: ShuttlingResult) -> tuple[Figure, Axes]:
    """Plot trap center and position expectation value versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.trap_center, label="q(t)")
    ax.plot(result.times, result.position_expectation, label="<x(t)>")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend()
    return fig, ax


def plot_excitation_trace(result: ShuttlingResult) -> tuple[Figure, Axes]:
    """Plot instantaneous motional excitation number versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.excitation_number)
    ax.set_xlabel("time")
    ax.set_ylabel("<n(t)>")
    return fig, ax


def plot_momentum_trace(result: ShuttlingResult) -> tuple[Figure, Axes]:
    """Plot lab-frame and moving-frame momentum expectation values versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    lower = result.momentum_expectation - result.momentum_std
    upper = result.momentum_expectation + result.momentum_std
    ax.plot(result.times, result.momentum_expectation, label="<p(t)> lab")
    ax.plot(result.times, result.relative_momentum_expectation, "--", label="<p(t)> - m qdot(t)")
    ax.fill_between(result.times, lower, upper, alpha=0.2, label="+/- Delta p")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("momentum")
    ax.legend()
    return fig, ax


def plot_phase_space(result: ShuttlingResult) -> tuple[Figure, Axes]:
    """Plot the motional phase-space trajectory relative to the moving trap center."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    displacement = result.position_expectation - result.trap_center
    momentum = result.relative_momentum_expectation
    line = ax.scatter(displacement, momentum, c=result.times, s=12)
    ax.scatter(displacement[0], momentum[0], marker="o", color="tab:green", label="start")
    ax.scatter(displacement[-1], momentum[-1], marker="x", color="tab:red", label="end")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("<x(t)> - q(t)")
    ax.set_ylabel("<p(t)> - m qdot(t)")
    ax.legend()
    fig.colorbar(line, ax=ax, label="time")
    return fig, ax


def plot_populations(result: ShuttlingResult) -> tuple[Figure, Axes]:
    """Plot instantaneous oscillator populations versus time."""
    if result.populations is None:
        msg = "result does not contain populations; set ShuttlingConfig.population_levels > 0."
        raise ValueError(msg)
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    for level in range(result.populations.shape[1]):
        ax.plot(result.times, result.populations[:, level], label=f"P_{level}")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    _legend_first_n(ax)
    return fig, ax


def plot_ms_fidelity(result: MSGateResult) -> tuple[Figure, Axes]:
    """Plot MS-gate target-state fidelity versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.target_fidelity)
    ax.set_xlabel("time")
    ax.set_ylabel("target fidelity")
    ax.set_ylim(0.0, 1.02)
    return fig, ax


def plot_ms_spin_populations(result: MSGateResult) -> tuple[Figure, Axes]:
    """Plot two-qubit computational-basis populations during the MS gate."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    labels = ("00", "01", "10", "11")
    for index, label in enumerate(labels):
        ax.plot(result.times, result.spin_populations[:, index], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("spin population")
    ax.legend()
    return fig, ax


def plot_ms_mode_expectations(result: MSGateResult) -> tuple[Figure, Axes]:
    """Plot COM and stretch mode occupations during the MS gate."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.mode_expectations[:, 0], label=result.config.modes[0].name)
    ax.plot(result.times, result.mode_expectations[:, 1], label=result.config.modes[1].name)
    ax.set_xlabel("time")
    ax.set_ylabel("mode occupation")
    ax.legend()
    return fig, ax


def plot_ms_phase_space_loops(
    results: MSGateResult | dict[str, MSGateResult],
    *,
    branch_indices: tuple[int, int] | None = None,
) -> tuple[Figure, Any]:
    """Plot intended MS spin-branch displacement loops for COM and stretch modes.

    Args:
        results: One result or a named collection of results to overlay.
        branch_indices: Optional branch index per mode. If omitted, the largest
            active branch from the first result is used for each mode.

    Returns:
        Matplotlib figure and axes.
    """
    from .ms_gate import ms_phase_space_displacements

    result_map = cast("dict[str, MSGateResult]", results if isinstance(results, dict) else {"loop": results})
    first_result = next(iter(result_map.values()))
    first_loop = ms_phase_space_displacements(first_result)
    if branch_indices is None:
        branch_indices = cast(
            "tuple[int, int]",
            tuple(
                int(np.argmax(np.max(np.abs(first_loop.displacements[:, mode_index, :]), axis=0)))
                for mode_index in range(first_loop.displacements.shape[1])
            ),
        )
    selected_branch_indices = branch_indices

    pyplot, _animation = _require_matplotlib()
    fig, axes = pyplot.subplots(1, len(first_result.config.modes), figsize=(5 * len(first_result.config.modes), 4))
    if len(first_result.config.modes) == 1:
        axes = np.asarray([axes], dtype=object)

    for mode_index, ax in enumerate(axes):
        branch_index = selected_branch_indices[mode_index]
        branch_label = _ms_branch_label(first_loop.branches[branch_index])
        for label, result in result_map.items():
            loop = ms_phase_space_displacements(result)
            displacement = loop.displacements[:, mode_index, branch_index]
            ax.plot(displacement.real, displacement.imag, label=label)
            ax.scatter(displacement.real[0], displacement.imag[0], marker="o", color="tab:green", s=25)
            ax.scatter(displacement.real[-1], displacement.imag[-1], marker="x", color="tab:red", s=35)
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{first_result.config.modes[mode_index].name}, branch {branch_label}")
        ax.set_xlabel("Re alpha")
        ax.set_ylabel("Im alpha")
        ax.legend()
    return fig, axes


def plot_ms_final_fidelities(results: dict[str, MSGateResult]) -> tuple[Figure, Axes]:
    """Plot final target fidelity for several MS-gate cases."""
    pyplot, _animation = _require_matplotlib()
    labels = list(results)
    values = [results[label].target_fidelity[-1] for label in labels]
    fig, ax = pyplot.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("final target fidelity")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    return fig, ax


def _ms_branch_label(branch: tuple[int, int]) -> str:
    return "".join("+" if value > 0 else "-" for value in branch)


def plot_center_of_mass_trace(result: TwoIonShuttlingResult) -> tuple[Figure, Axes]:
    """Plot two-ion center of mass and trap center versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.trap_center, label="q(t)")
    ax.plot(result.times, result.center_of_mass, label="<R(t)>")
    ax.plot(result.times, result.x1_expectation, "--", alpha=0.6, label="<x1(t)>")
    ax.plot(result.times, result.x2_expectation, "--", alpha=0.6, label="<x2(t)>")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.legend()
    return fig, ax


def plot_mean_separation_trace(result: TwoIonShuttlingResult) -> tuple[Figure, Axes]:
    """Plot mean ion separation versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.mean_separation)
    ax.set_xlabel("time")
    ax.set_ylabel("<|x2 - x1|>")
    return fig, ax


def plot_excess_energy_trace(result: TwoIonShuttlingResult) -> tuple[Figure, Axes]:
    """Plot two-ion excess energy in oscillator quanta versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.excess_quanta)
    ax.set_xlabel("time")
    ax.set_ylabel("excess energy / (hbar omega)")
    return fig, ax


def plot_mode_excitation_trace(result: TwoIonShuttlingResult) -> tuple[Figure, Axes]:
    """Plot COM-like and stretch-like normal-mode excitations versus time."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    ax.plot(result.times, result.com_mode_excitation, label="COM-like mode")
    ax.plot(result.times, result.stretch_mode_excitation, label="stretch-like mode")
    ax.set_xlabel("time")
    ax.set_ylabel("moving-frame mode excitation")
    _legend_first_n(ax)
    return fig, ax


def plot_mode_populations(result: TwoIonShuttlingResult) -> tuple[Figure, Any]:
    """Plot truncated instantaneous normal-mode populations versus time.

    The plotted COM-like and stretch-like curves are marginal probabilities obtained
    from the stored joint normal-mode population table. If only a finite number of
    levels was requested, their sum can be smaller than one.
    """
    if result.com_mode_populations is None or result.stretch_mode_populations is None:
        msg = (
            "result does not contain normal-mode populations; "
            "set TwoIonShuttlingConfig.mode_population_levels > 0."
        )
        raise ValueError(msg)

    pyplot, _animation = _require_matplotlib()
    include_residual = result.mode_population_residual is not None
    column_count = 3 if include_residual else 2
    fig, axes = pyplot.subplots(1, column_count, figsize=(5 * column_count, 4), sharey=True)
    population_data = (
        ("COM-like mode", result.com_mode_populations),
        ("stretch-like mode", result.stretch_mode_populations),
    )
    for ax, (title, populations) in zip(axes[:2], population_data, strict=True):
        for level in range(populations.shape[1]):
            ax.plot(result.times, populations[:, level], label=f"P_{level}")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("population")
        _legend_first_n(ax)
    if include_residual:
        axes[2].plot(result.times, result.mode_population_residual, color="black", label="outside cutoff")
        axes[2].set_title("truncation residual")
        axes[2].set_xlabel("time")
        axes[2].set_ylabel("population")
        _legend_first_n(axes[2])
    return fig, axes


def plot_joint_mode_populations(
    result: TwoIonShuttlingResult,
    time_indices: Sequence[int] | None = None,
) -> tuple[Figure, Any]:
    """Plot joint COM-like/stretch-like normal-mode populations at selected times."""
    if result.joint_mode_populations is None:
        msg = (
            "result does not contain joint normal-mode populations; "
            "set TwoIonShuttlingConfig.mode_population_levels > 0."
        )
        raise ValueError(msg)

    indices = _selected_time_indices(result.times.size, time_indices)
    pyplot, _animation = _require_matplotlib()
    fig, axes = pyplot.subplots(1, indices.size, figsize=(4.3 * indices.size, 3.8), squeeze=False)
    axes_row = axes[0]
    images = []
    for ax, time_index in zip(axes_row, indices, strict=True):
        populations = result.joint_mode_populations[time_index]
        image = ax.imshow(
            populations.T,
            origin="lower",
            aspect="auto",
            extent=(-0.5, populations.shape[0] - 0.5, -0.5, populations.shape[1] - 0.5),
        )
        images.append(image)
        ax.set_title(f"t = {result.times[time_index]:.3g}")
        ax.set_xlabel("COM-like level")
        ax.set_ylabel("stretch-like level")
    fig.colorbar(images[-1], ax=axes_row, label="population")
    return fig, axes_row


def plot_mode_population_consistency(result: TwoIonShuttlingResult) -> tuple[Figure, Any]:
    """Compare moment-based mode excitations with population-derived occupations."""
    if result.com_mode_population_excitation is None or result.stretch_mode_population_excitation is None:
        msg = (
            "result does not contain population-derived mode excitations; "
            "set TwoIonShuttlingConfig.mode_population_levels > 0."
        )
        raise ValueError(msg)

    pyplot, _animation = _require_matplotlib()
    fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))
    mode_data = (
        (
            "COM-like mode",
            result.com_mode_excitation,
            result.com_mode_population_excitation,
        ),
        (
            "stretch-like mode",
            result.stretch_mode_excitation,
            result.stretch_mode_population_excitation,
        ),
    )
    for ax, (title, moment_excitation, population_excitation) in zip(axes[:2], mode_data, strict=True):
        ax.plot(result.times, moment_excitation, label="moment")
        ax.plot(result.times, population_excitation, "--", label="population")
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("excitation")
        _legend_first_n(ax)

    axes[2].plot(
        result.times,
        result.com_mode_excitation - result.com_mode_population_excitation,
        label="COM moment - population",
    )
    axes[2].plot(
        result.times,
        result.stretch_mode_excitation - result.stretch_mode_population_excitation,
        label="stretch moment - population",
    )
    axes[2].axhline(0.0, color="0.7", linewidth=0.8)
    axes[2].set_title("consistency difference")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("excitation")
    _legend_first_n(axes[2])
    return fig, axes


def plot_two_ion_phase_space(result: TwoIonShuttlingResult) -> tuple[Figure, Axes]:
    """Plot two-ion center-of-mass phase space in the moving-well frame."""
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    displacement = result.center_of_mass - result.trap_center
    momentum = result.relative_center_of_mass_momentum
    line = ax.scatter(displacement, momentum, c=result.times, s=12)
    ax.scatter(displacement[0], momentum[0], marker="o", color="tab:green", label="start")
    ax.scatter(displacement[-1], momentum[-1], marker="x", color="tab:red", label="end")
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.axvline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel("<R(t)> - q(t)")
    ax.set_ylabel("<P_COM(t)> - M qdot(t)")
    ax.legend()
    fig.colorbar(line, ax=ax, label="time")
    return fig, ax


def plot_normal_mode_phase_spaces(result: TwoIonShuttlingResult) -> tuple[Figure, Any]:
    """Plot COM-like and stretch-like normal-mode phase-space trajectories."""
    pyplot, _animation = _require_matplotlib()
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 4))

    mode_data = (
        (
            "COM-like mode",
            result.com_mode_coordinate,
            result.relative_com_mode_momentum,
            result.com_mode_frequency,
        ),
        (
            "stretch-like mode",
            result.stretch_mode_coordinate,
            result.relative_stretch_mode_momentum,
            result.stretch_mode_frequency,
        ),
    )
    for ax, (label, coordinate, momentum, frequency) in zip(axes, mode_data, strict=True):
        line = ax.scatter(coordinate, momentum, c=result.times, s=12)
        ax.scatter(coordinate[0], momentum[0], marker="o", color="tab:green", label="start")
        ax.scatter(coordinate[-1], momentum[-1], marker="x", color="tab:red", label="end")
        ax.axhline(0.0, color="0.7", linewidth=0.8)
        ax.axvline(0.0, color="0.7", linewidth=0.8)
        ax.set_title(f"{label}, omega = {frequency:.3g}")
        ax.set_xlabel("<Q>")
        ax.set_ylabel("<P> moving frame")
        ax.legend()
        fig.colorbar(line, ax=ax, label="time")
    return fig, axes


def plot_pair_density_snapshot(result: TwoIonShuttlingResult, snapshot_index: int = -1) -> tuple[Figure, Axes]:
    """Plot one pair-density snapshot over the two-ion coordinate grid."""
    if result.pair_density is None or result.snapshot_times is None:
        msg = "result does not contain pair-density snapshots; set snapshot_stride on TwoIonShuttlingConfig."
        raise ValueError(msg)
    pyplot, _animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    density = result.pair_density[snapshot_index]
    extent = (result.positions[0], result.positions[-1], result.positions[0], result.positions[-1])
    image = ax.imshow(density.T, origin="lower", extent=extent, aspect="auto")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"t = {result.snapshot_times[snapshot_index]:.3g}")
    fig.colorbar(image, ax=ax, label="|psi(x1,x2)|^2")
    return fig, ax


def animate_pair_density(result: TwoIonShuttlingResult, *, interval: int = 80) -> FuncAnimation:
    """Animate pair-density snapshots over the two-ion coordinate grid."""
    pair_density, snapshot_times = _require_two_ion_snapshots(result)
    pyplot, animation = _require_matplotlib()
    fig, ax = pyplot.subplots()
    extent = (result.positions[0], result.positions[-1], result.positions[0], result.positions[-1])
    image = ax.imshow(pair_density[0].T, origin="lower", extent=extent, aspect="auto")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white", va="top")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(image, ax=ax, label="|psi(x1,x2)|^2")

    def update(frame: int) -> tuple[Any, ...]:
        image.set_data(pair_density[frame].T)
        image.set_clim(0.0, float(np.max(pair_density[frame])))
        time_text.set_text(f"t = {snapshot_times[frame]:.3g}")
        return image, time_text

    return animation.FuncAnimation(fig, update, frames=pair_density.shape[0], interval=interval, blit=True)


def plot_two_ion_marginal_trajectories(result: TwoIonShuttlingResult) -> tuple[Figure, Any]:
    """Plot x1 and x2 marginal densities as position-versus-time heatmaps."""
    pair_density, snapshot_times = _require_two_ion_snapshots(result)
    x1_density, x2_density = _two_ion_marginal_densities(result, pair_density)
    snapshot_indices = _snapshot_time_indices(result, snapshot_times)

    pyplot, _animation = _require_matplotlib()
    fig, axes = pyplot.subplots(2, 1, sharex=True)
    extent = (snapshot_times[0], snapshot_times[-1], result.positions[0], result.positions[-1])

    x1_image = axes[0].imshow(x1_density.T, origin="lower", extent=extent, aspect="auto")
    axes[0].plot(snapshot_times, result.trap_center[snapshot_indices], color="white", linestyle="--", label="q(t)")
    axes[0].plot(snapshot_times, result.x1_expectation[snapshot_indices], color="tab:red", label="<x1(t)>")
    axes[0].set_ylabel("x1")
    axes[0].legend(loc="upper right")
    fig.colorbar(x1_image, ax=axes[0], label="rho1(x1, t)")

    x2_image = axes[1].imshow(x2_density.T, origin="lower", extent=extent, aspect="auto")
    axes[1].plot(snapshot_times, result.trap_center[snapshot_indices], color="white", linestyle="--", label="q(t)")
    axes[1].plot(snapshot_times, result.x2_expectation[snapshot_indices], color="tab:red", label="<x2(t)>")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("x2")
    axes[1].legend(loc="upper right")
    fig.colorbar(x2_image, ax=axes[1], label="rho2(x2, t)")

    return fig, axes


def animate_two_ion_marginals(
    result: TwoIonShuttlingResult,
    *,
    interval: int = 80,
    density_scale: float | None = None,
) -> FuncAnimation:
    """Animate labeled ion marginals with a derived collective COM density overlay."""
    pair_density, snapshot_times = _require_two_ion_snapshots(result)
    x1_density, x2_density = _two_ion_marginal_densities(result, pair_density)
    collective_density = _two_ion_collective_density(result, pair_density)
    snapshot_indices = _snapshot_time_indices(result, snapshot_times)
    trap_center = result.trap_center[snapshot_indices]
    x1_expectation = result.x1_expectation[snapshot_indices]
    x2_expectation = result.x2_expectation[snapshot_indices]
    collective_expectation = result.center_of_mass[snapshot_indices]
    collective_potential = _two_ion_collective_potential(result, trap_center)

    if density_scale is None:
        potential_span = float(np.max(collective_potential))
        density_max = float(max(np.max(x1_density), np.max(x2_density), np.max(collective_density)))
        density_scale = 0.35 * potential_span / density_max if density_max > 0.0 else 1.0

    pyplot, animation = _require_matplotlib()
    fig, ax = pyplot.subplots()

    (potential_line,) = ax.plot(result.positions, collective_potential[0], color="0.45", label="V_COM(R, t)")
    (density_1_line,) = ax.plot(result.positions, x1_density[0] * density_scale, color="tab:blue", label="rho1(x, t)")
    (density_2_line,) = ax.plot(result.positions, x2_density[0] * density_scale, color="tab:orange", label="rho2(x, t)")
    (collective_line,) = ax.plot(
        result.positions,
        collective_density[0] * density_scale,
        color="black",
        linestyle="--",
        label="rho_COM(R, t)",
    )
    trap_line = ax.axvline(trap_center[0], color="tab:green", linestyle="--", label="q(t)")
    expectation_1_line = ax.axvline(x1_expectation[0], color="tab:blue", linestyle=":", label="<x1(t)>")
    expectation_2_line = ax.axvline(x2_expectation[0], color="tab:orange", linestyle=":", label="<x2(t)>")
    collective_expectation_line = ax.axvline(
        collective_expectation[0], color="black", linestyle="-.", label="<R(t)>"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("energy / scaled density")
    y_max = float(
        np.max(
            collective_potential
            + np.maximum.reduce(
                (
                    x1_density * density_scale,
                    x2_density * density_scale,
                    collective_density * density_scale,
                )
            )
        )
    )
    ax.set_xlim(result.positions[0], result.positions[-1])
    ax.set_ylim(0.0, y_max * 1.05 if y_max > 0.0 else 1.0)
    ax.legend(loc="upper right")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(frame: int) -> tuple[Any, ...]:
        potential_line.set_ydata(collective_potential[frame])
        density_1_line.set_ydata(x1_density[frame] * density_scale)
        density_2_line.set_ydata(x2_density[frame] * density_scale)
        collective_line.set_ydata(collective_density[frame] * density_scale)
        trap_line.set_xdata([trap_center[frame], trap_center[frame]])
        expectation_1_line.set_xdata([x1_expectation[frame], x1_expectation[frame]])
        expectation_2_line.set_xdata([x2_expectation[frame], x2_expectation[frame]])
        collective_expectation_line.set_xdata([collective_expectation[frame], collective_expectation[frame]])
        time_text.set_text(f"t = {snapshot_times[frame]:.3g}")

        return (
            potential_line,
            density_1_line,
            density_2_line,
            collective_line,
            trap_line,
            expectation_1_line,
            expectation_2_line,
            collective_expectation_line,
            time_text,
        )

    return animation.FuncAnimation(fig, update, frames=snapshot_times.size, interval=interval, blit=True)


def _require_two_ion_snapshots(result: TwoIonShuttlingResult) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if result.pair_density is None or result.snapshot_times is None:
        msg = "result does not contain pair-density snapshots; set snapshot_stride on TwoIonShuttlingConfig."
        raise ValueError(msg)
    return result.pair_density, result.snapshot_times


def _two_ion_marginal_densities(
    result: TwoIonShuttlingResult, pair_density: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dx = result.positions[1] - result.positions[0]
    x1_density = np.sum(pair_density, axis=2) * dx
    x2_density = np.sum(pair_density, axis=1) * dx
    return x1_density, x2_density


def _two_ion_collective_density(
    result: TwoIonShuttlingResult, pair_density: NDArray[np.float64]
) -> NDArray[np.float64]:
    dx = result.positions[1] - result.positions[0]
    mass_1 = result.config.ion1.mass
    mass_2 = result.config.ion2.mass
    total_mass = mass_1 + mass_2
    x1_grid, x2_grid = np.meshgrid(result.positions, result.positions, indexing="ij")
    collective_positions = ((mass_1 * x1_grid + mass_2 * x2_grid) / total_mass).ravel()
    edges = _grid_bin_edges(result.positions)
    density = np.empty((pair_density.shape[0], result.positions.size), dtype=np.float64)
    area_element = dx**2

    for frame_index, frame_density in enumerate(pair_density):
        weights = frame_density.ravel() * area_element
        density[frame_index], _bins = np.histogram(collective_positions, bins=edges, weights=weights)
    return density / dx


def _snapshot_time_indices(result: TwoIonShuttlingResult, snapshot_times: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.searchsorted(result.times, snapshot_times).astype(np.int64)


def _selected_time_indices(total_steps: int, time_indices: Sequence[int] | None) -> NDArray[np.int64]:
    if time_indices is None:
        candidates = (0, total_steps // 2, total_steps - 1)
    else:
        candidates = tuple(time_indices)

    normalized: list[int] = []
    for index in candidates:
        positive_index = index + total_steps if index < 0 else index
        if positive_index < 0 or positive_index >= total_steps:
            msg = f"time index {index} is out of bounds for {total_steps} samples."
            raise IndexError(msg)
        if positive_index not in normalized:
            normalized.append(positive_index)
    if not normalized:
        msg = "at least one time index is required."
        raise ValueError(msg)
    return np.asarray(normalized, dtype=np.int64)


def _two_ion_collective_potential(
    result: TwoIonShuttlingResult,
    trap_center: NDArray[np.float64],
) -> NDArray[np.float64]:
    displacement = result.positions[None, :] - trap_center[:, None]
    total_mass = result.config.ion1.mass + result.config.ion2.mass
    return 0.5 * total_mass * result.config.omega**2 * displacement**2


def _grid_bin_edges(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    dx = positions[1] - positions[0]
    edges = np.empty(positions.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (positions[:-1] + positions[1:])
    edges[0] = positions[0] - 0.5 * dx
    edges[-1] = positions[-1] + 0.5 * dx
    return edges


def _legend_first_n(ax: Axes, max_entries: int = 7) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:max_entries], labels[:max_entries])


def _require_matplotlib() -> tuple[Any, Any]:
    try:
        animation = importlib.import_module("matplotlib.animation")
        pyplot = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        msg = "Matplotlib is required for shuttling visualization helpers."
        raise ImportError(msg) from exc
    return pyplot, animation
