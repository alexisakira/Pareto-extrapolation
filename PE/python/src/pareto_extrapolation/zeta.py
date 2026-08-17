"""Pareto exponent and upper-tail type distribution."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ._validation import expand_state_rows, prepare_markov_inputs, real_finite_array


def get_zeta(
    ps: ArrayLike,
    pj: ArrayLike,
    v: ArrayLike,
    g: ArrayLike,
    zeta_bound: ArrayLike = (1e-2, 100.0),
) -> tuple[float, np.ndarray]:
    """Compute the Pareto exponent and upper-tail type distribution.

    Rows indexed by a state pair use the order ``S*s + t``.  If no root is
    identified inside ``zeta_bound``, the relevant endpoint and an empty type
    distribution are returned with a ``RuntimeWarning``.
    """

    bound = real_finite_array(zeta_bound, "zeta_bound").reshape(-1)
    if bound.size != 2 or bound[0] <= 0 or bound[0] >= bound[1]:
        raise ValueError("zeta_bound must contain two increasing positive values")
    lower, upper = float(bound[0]), float(bound[1])

    ps_array, pj_array, v_array = prepare_markov_inputs(ps, pj, v)
    s = ps_array.shape[0]
    j = pj_array.shape[1]
    g_array = expand_state_rows(g, "G", s, j)
    if np.any(g_array < 0):
        raise ValueError("G must be nonnegative")

    if np.max(g_array) <= 1:
        warnings.warn("the model does not generate Pareto tails", RuntimeWarning, stacklevel=2)
        return upper, np.empty(0)

    def a_matrix(z: float) -> np.ndarray:
        moments = np.sum(pj_array * np.power(g_array, z), axis=1)
        return ps_array * v_array * moments.reshape(s, s)

    def log_spectral_radius(z: float) -> float:
        rho = float(np.max(np.abs(np.linalg.eigvals(a_matrix(z)))))
        if not np.isfinite(rho):
            raise ValueError("the spectral radius is not finite; check PS, V, and G")
        return -np.inf if rho <= 0 else float(np.log(rho))

    f_lower = log_spectral_radius(lower)
    f_upper = log_spectral_radius(upper)
    bound_tol = 1e-12

    if f_lower > bound_tol:
        warnings.warn("zeta is below the lower bound", RuntimeWarning, stacklevel=2)
        return lower, np.empty(0)
    if f_upper < -bound_tol:
        warnings.warn("zeta is above the upper bound", RuntimeWarning, stacklevel=2)
        return upper, np.empty(0)

    if abs(f_lower) <= bound_tol:
        zeta = lower
    elif abs(f_upper) <= bound_tol:
        zeta = upper
    else:
        zeta = float(brentq(log_spectral_radius, lower, upper))

    eigenvalues, eigenvectors = np.linalg.eig(a_matrix(zeta).T)
    index = int(np.argmax(eigenvalues.real))
    type_dist = np.abs(np.real(eigenvectors[:, index]))
    total = float(type_dist.sum())
    if not np.isfinite(total) or total <= 0:
        warnings.warn(
            "the upper-tail type distribution could not be normalized",
            RuntimeWarning,
            stacklevel=2,
        )
        return zeta, np.empty(0)
    return zeta, type_dist / total
