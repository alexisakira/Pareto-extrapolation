"""Top wealth shares."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

from ._validation import real_finite_array


def _interpolate(x: np.ndarray, y: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    if x_query.size == 0:
        return x_query.copy()
    if x.size < 2:
        raise ValueError("not enough distinct probability knots to interpolate")
    query = np.clip(x_query, x[0], x[-1])
    if x.size == 2:
        return np.interp(query, x, y)
    return np.asarray(PchipInterpolator(x, y, extrapolate=False)(query), dtype=float)


def get_top_shares(
    top_prob: ArrayLike,
    w_grid: ArrayLike,
    w_dist: ArrayLike,
    zeta: float = 0.0,
) -> float | np.ndarray:
    """Compute top wealth shares using truncation or Pareto extrapolation.

    Negative wealth is allowed.  Consequently, valid top shares may exceed one
    and need not be monotone as the requested population share increases.
    Aggregate net wealth must be strictly positive.
    """

    top_input = real_finite_array(top_prob, "top_prob")
    scalar_input = top_input.ndim == 0
    if top_input.ndim > 1:
        raise ValueError("top_prob must be a scalar or one-dimensional array")
    top = np.atleast_1d(top_input).astype(float, copy=False)
    if np.any((top < 0) | (top > 1)):
        raise ValueError("entries of top_prob must lie in [0, 1]")
    if np.any(np.diff(top) <= 0):
        raise ValueError("top_prob must be strictly increasing")

    grid = real_finite_array(w_grid, "w_grid")
    dist = real_finite_array(w_dist, "w_dist")
    if grid.ndim != 1 or grid.size == 0 or dist.ndim != 1 or dist.size == 0:
        raise ValueError("w_grid and w_dist must be nonempty one-dimensional arrays")
    if grid.size != dist.size:
        raise ValueError("w_grid and w_dist must have the same length")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("w_grid must be strictly increasing")
    if np.any(dist < 0):
        raise ValueError("w_dist must be nonnegative")
    dist_sum = float(dist.sum())
    if not np.isclose(dist_sum, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"w_dist must sum to 1; its sum is {dist_sum:g}")
    dist = dist / dist_sum

    if not np.isscalar(zeta) or not np.isreal(zeta) or not np.isfinite(zeta) or zeta < 0:
        raise ValueError("zeta must be a nonnegative finite real scalar")
    zeta = float(zeta)

    tail_prob = np.cumsum(dist[::-1])
    unique_tail, unique_index = np.unique(tail_prob, return_index=True)

    if zeta == 0:
        aggregate = float(np.dot(dist, grid))
        if aggregate <= 0:
            raise ValueError("aggregate net wealth must be strictly positive")
        top_wealth = np.cumsum((dist * grid)[::-1]) / aggregate
        all_prob = np.concatenate(([0.0], unique_tail))
        all_wealth = np.concatenate(([0.0], top_wealth[unique_index]))
        prob_knots, knot_index = np.unique(all_prob, return_index=True)
        result = _interpolate(prob_knots, all_wealth[knot_index], top)
    else:
        if zeta <= 1:
            raise ValueError("zeta must exceed 1 for a finite mean")
        if grid[-1] <= 0:
            raise ValueError("the largest wealth grid point must be positive")
        corrected_grid = grid.copy()
        corrected_grid[-1] = zeta / (zeta - 1) * grid[-1]
        aggregate = float(np.dot(dist, corrected_grid))
        if aggregate <= 0:
            raise ValueError(
                "aggregate net wealth including the Pareto correction must be positive"
            )
        top_wealth = np.cumsum((dist * corrected_grid)[::-1]) / aggregate
        result = np.zeros_like(top)
        extrapolate = top <= tail_prob[0]
        result[extrapolate] = (
            zeta
            / (zeta - 1)
            * dist[-1] ** (1 / zeta)
            * (grid[-1] / aggregate)
            * top[extrapolate] ** (1 - 1 / zeta)
        )
        interpolate = ~extrapolate
        result[interpolate] = _interpolate(
            unique_tail, top_wealth[unique_index], top[interpolate]
        )

    if not np.all(np.isfinite(result)):
        raise RuntimeError("failed to compute finite top wealth shares")
    return float(result[0]) if scalar_input else result
