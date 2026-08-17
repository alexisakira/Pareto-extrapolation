"""Sparse transition matrices with Pareto extrapolation."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix, csr_matrix, eye, vstack
from scipy.sparse.linalg import eigs, spsolve

from ._validation import expand_state_rows, prepare_markov_inputs, real_finite_array
from .zeta import get_zeta


def _locate(values: ArrayLike, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values_array = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values_array)):
        raise ValueError("destinations of the law of motion must be finite")
    n = grid.size
    index = np.searchsorted(grid, values_array, side="right") - 1
    below = values_array <= grid[0]
    above = values_array >= grid[-1]
    index = np.clip(index, 0, n - 2)
    theta = (values_array - grid[index]) / (grid[index + 1] - grid[index])
    theta[below] = 0.0
    theta[above] = 1.0
    return index, theta


def _stationary_distribution(q: csr_matrix) -> np.ndarray:
    size = q.shape[0]
    candidate: np.ndarray | None = None
    try:
        if size <= 3:
            eigenvalues, eigenvectors = np.linalg.eig(q.T.toarray())
            index = int(np.argmin(np.abs(eigenvalues - 1)))
            vector = eigenvectors[:, index]
        else:
            _, eigenvectors = eigs(q.T, k=1, which="LM")
            vector = eigenvectors[:, 0]
        vector = np.abs(np.real(vector))
        if np.all(np.isfinite(vector)) and vector.sum() > 0:
            candidate = vector / vector.sum()
            if np.linalg.norm(q.T @ candidate - candidate, ord=1) > 1e-8:
                candidate = None
    except Exception:  # ARPACK can fail on small, reducible, or ill-conditioned chains.
        candidate = None

    if candidate is None:
        equations = q.T - eye(size, format="csr")
        normalization = csr_matrix(np.ones((1, size)))
        system = vstack((equations[:-1, :], normalization), format="csr")
        rhs = np.zeros(size)
        rhs[-1] = 1.0
        candidate = np.asarray(spsolve(system, rhs), dtype=float)
        if not np.all(np.isfinite(candidate)):
            raise RuntimeError("failed to compute a finite stationary distribution")
        candidate = np.maximum(candidate, 0.0)
        if candidate.sum() <= 0:
            raise RuntimeError("failed to normalize the stationary distribution")
        candidate /= candidate.sum()

    residual = float(np.linalg.norm(q.T @ candidate - candidate, ord=1))
    if residual > 1e-8:
        raise RuntimeError(f"stationary-distribution residual is {residual:g}")
    return candidate


def get_q(
    ps: ArrayLike,
    pj: ArrayLike,
    v: ArrayLike,
    x0: float | None,
    x_grid: ArrayLike,
    gstjn: ArrayLike,
    g_stj: ArrayLike | None = None,
    zeta: float | None = None,
    *,
    compute_stationary: bool = True,
) -> tuple[csr_matrix, np.ndarray | None]:
    """Construct the sparse transition matrix and its stationary distribution.

    The joint-state order is ``(s,n)`` with the grid index varying fastest.
    Pass ``compute_stationary=False`` to skip the eigenvector calculation; the
    second returned value is then ``None``.
    """

    ps_array, pj_array, v_array = prepare_markov_inputs(ps, pj, v)
    s_count = ps_array.shape[0]
    j_count = pj_array.shape[1]

    grid = real_finite_array(x_grid, "x_grid")
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("x_grid must be a one-dimensional array with at least two values")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("x_grid must be strictly increasing")
    if grid[-1] <= 0:
        raise ValueError("the largest grid point must be positive")
    n_count = grid.size

    law = expand_state_rows(gstjn, "gstjn", s_count, n_count * j_count)
    has_reset = bool(np.any(v_array < 1))
    if has_reset:
        if x0 is None or not np.isscalar(x0) or not np.isreal(x0) or not np.isfinite(x0):
            raise ValueError("x0 must be a finite real scalar when V contains an entry below 1")
        if x0 < grid[0] or x0 >= grid[-1]:
            raise ValueError("x0 must satisfy min(x_grid) <= x0 < max(x_grid)")

    top_columns = n_count * np.arange(1, j_count + 1) - 1
    if g_stj is None:
        slopes = (law[:, top_columns] - law[:, top_columns - 1]) / (grid[-1] - grid[-2])
    else:
        slopes = expand_state_rows(g_stj, "Gstj", s_count, j_count)
    if np.any(slopes <= 0):
        raise ValueError("asymptotic slopes must be strictly positive")

    if zeta is None:
        zeta, _ = get_zeta(ps_array, pj_array, v_array, slopes)
    if not np.isscalar(zeta) or not np.isreal(zeta) or not np.isfinite(zeta):
        raise ValueError("zeta must be a finite real scalar")
    zeta = float(zeta)
    if zeta <= 1:
        raise ValueError("zeta must exceed 1 for a finite mean")

    spacing = grid[-1] - grid[-2]
    shortfall = (grid[-1] - law[:, top_columns]) / (slopes * spacing)
    extra_steps = int(np.max(np.maximum(np.ceil(shortfall), 0)))
    n_extra = extra_steps + 1
    if n_extra > 10_000_000:
        raise ValueError(
            f"the extrapolation requires {n_extra} points; use a larger upper bound or coarser grid"
        )

    tail_weights = np.ones(n_extra)
    if n_extra > 1:
        step = spacing / grid[-1]
        k = np.arange(n_extra - 1)
        tail_weights[:-1] = zeta * step * (1 + k * step) ** (-zeta - 1)
        tail_weights[-1] = (1 + (n_extra - 1) * step) ** (-zeta)
        tail_weights[-1] += (
            zeta * step * (1 + (n_extra - 1) * step) ** (-zeta - 1) / 2
        )
        tail_weights /= tail_weights.sum()

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    interior_rows = np.arange(n_count - 1)

    for s in range(s_count):
        for t in range(s_count):
            state_pair = s_count * s + t
            transition_weight = ps_array[s, t] * v_array[s, t]
            if transition_weight == 0:
                continue
            for j in range(j_count):
                weight = transition_weight * pj_array[state_pair, j]
                if weight == 0:
                    continue

                destinations = law[
                    state_pair, n_count * j + np.arange(n_count - 1)
                ]
                index, theta = _locate(destinations, grid)
                rows = s * n_count + interior_rows
                row_parts.append(np.concatenate((rows, rows)))
                column_parts.append(
                    np.concatenate((t * n_count + index, t * n_count + index + 1))
                )
                value_parts.append(np.concatenate(((1 - theta) * weight, theta * weight)))

                extrapolated = (
                    law[state_pair, n_count * (j + 1) - 1]
                    + slopes[state_pair, j] * spacing * np.arange(n_extra)
                )
                extra_index, extra_theta = _locate(extrapolated, grid)
                node_weights = np.bincount(
                    np.concatenate((extra_index, extra_index + 1)),
                    weights=np.concatenate(
                        (tail_weights * (1 - extra_theta), tail_weights * extra_theta)
                    ),
                    minlength=n_count,
                )
                nonzero = np.flatnonzero(node_weights)
                row_parts.append(np.full(nonzero.size, (s + 1) * n_count - 1, dtype=int))
                column_parts.append(t * n_count + nonzero)
                value_parts.append(node_weights[nonzero] * weight)

    if has_reset:
        reset_index, reset_theta = _locate([float(x0)], grid)
        lower_index = int(reset_index[0])
        upper_weight = float(reset_theta[0])
        all_rows = np.arange(n_count)
        for s in range(s_count):
            for t in range(s_count):
                weight = (1 - v_array[s, t]) * ps_array[s, t]
                if weight == 0:
                    continue
                rows = s * n_count + all_rows
                row_parts.append(np.concatenate((rows, rows)))
                column_parts.append(
                    np.concatenate(
                        (
                            np.full(n_count, t * n_count + lower_index),
                            np.full(n_count, t * n_count + lower_index + 1),
                        )
                    )
                )
                value_parts.append(
                    np.concatenate(
                        (
                            np.full(n_count, weight * (1 - upper_weight)),
                            np.full(n_count, weight * upper_weight),
                        )
                    )
                )

    size = s_count * n_count
    if row_parts:
        rows = np.concatenate(row_parts)
        columns = np.concatenate(column_parts)
        values = np.concatenate(value_parts)
        q = coo_matrix((values, (rows, columns)), shape=(size, size)).tocsr()
        q.sum_duplicates()
        q.eliminate_zeros()
    else:
        q = csr_matrix((size, size), dtype=float)

    row_error = float(np.max(np.abs(np.asarray(q.sum(axis=1)).reshape(-1) - 1)))
    if row_error > 1e-8:
        warnings.warn(
            f"rows of Q deviate from 1 by as much as {row_error:g}",
            RuntimeWarning,
            stacklevel=2,
        )

    stationary = _stationary_distribution(q) if compute_stationary else None
    return q, stationary
