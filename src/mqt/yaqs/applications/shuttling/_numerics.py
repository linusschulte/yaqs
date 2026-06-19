# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared finite-difference helpers for shuttling simulations."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import scipy.sparse

from numpy.typing import NDArray


def sample_times(duration: float, dt: float) -> NDArray[np.float64]:
    """Sample a closed interval with a final point exactly at ``duration``."""
    if duration == 0.0:
        return np.array([0.0], dtype=np.float64)
    step_count = int(np.ceil(duration / dt))
    times = np.linspace(0.0, duration, step_count + 1, dtype=np.float64)
    return cast("NDArray[np.float64]", times)


def trap_velocity(times: NDArray[np.float64], trap_center: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate trap-center velocity from sampled positions."""
    if times.size == 1:
        return np.zeros_like(times)
    edge_order = 2 if times.size > 2 else 1
    velocity = np.gradient(trap_center, times, edge_order=edge_order)
    return cast("NDArray[np.float64]", velocity)


def kinetic_operator_1d(
    *,
    num_points: int,
    dx: float,
    mass: float,
    hbar: float,
) -> scipy.sparse.csr_matrix:
    """Construct a 1D second-order finite-difference kinetic operator."""
    diagonal = -2.0 * np.ones(num_points, dtype=np.float64)
    off_diagonal = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = scipy.sparse.diags(
        [off_diagonal, diagonal, off_diagonal],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="csr",
    ) / dx**2
    return cast("scipy.sparse.csr_matrix", -(hbar**2 / (2.0 * mass)) * second_derivative)


def momentum_operator_1d(*, num_points: int, dx: float, hbar: float) -> scipy.sparse.csr_matrix:
    """Construct a 1D centered-difference momentum operator."""
    off_diagonal = np.ones(num_points - 1, dtype=np.complex128)
    first_derivative = scipy.sparse.diags(
        [-off_diagonal, off_diagonal],
        offsets=[-1, 1],
        shape=(num_points, num_points),
        format="csr",
    ) / (2.0 * dx)
    return cast("scipy.sparse.csr_matrix", -1j * hbar * first_derivative)


def normalize_grid_state(
    psi: NDArray[np.complex128],
    *,
    volume_element: float,
) -> NDArray[np.complex128]:
    """Normalize a vector with respect to a grid quadrature weight."""
    norm = math.sqrt(float(np.vdot(psi, psi).real * volume_element))
    if norm == 0.0:
        msg = "wavefunction has zero norm."
        raise ValueError(msg)
    return psi / norm


def expectation(
    psi: NDArray[np.complex128],
    operator: scipy.sparse.spmatrix,
    *,
    volume_element: float,
) -> complex:
    """Grid-weighted expectation value of a sparse operator."""
    sparse_operator = cast("Any", operator)
    return complex(np.vdot(psi, sparse_operator @ psi) * volume_element)


def operator_std(
    psi: NDArray[np.complex128],
    operator: scipy.sparse.spmatrix,
    mean: float,
    *,
    volume_element: float,
) -> float:
    """Grid-weighted standard deviation of a Hermitian operator."""
    sparse_operator = cast("Any", operator)
    op_psi = sparse_operator @ psi
    second_moment = float((np.vdot(op_psi, op_psi) * volume_element).real)
    return math.sqrt(max(0.0, second_moment - mean**2))
