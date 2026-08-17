"""Shared validation and state-index expansion helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def real_finite_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert an input to a real, finite float array."""

    raw = np.asarray(value)
    if not np.isrealobj(raw):
        raise ValueError(f"{name} must contain only real values")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def row_array(value: ArrayLike, name: str) -> FloatArray:
    """Convert a scalar or one-dimensional input to a single-row array."""

    array = real_finite_array(value, name)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")
    return array


def prepare_markov_inputs(
    ps: ArrayLike, pj: ArrayLike, v: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate probabilities and expand ``PJ`` and ``V`` to full shapes."""

    ps_array = real_finite_array(ps, "PS")
    if ps_array.ndim == 0:
        ps_array = ps_array.reshape(1, 1)
    if ps_array.ndim != 2 or ps_array.shape[0] != ps_array.shape[1] or ps_array.size == 0:
        raise ValueError("PS must be a nonempty square matrix")
    if np.any(ps_array < 0) or not np.allclose(
        ps_array.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("PS must be nonnegative with rows summing to 1")

    s = ps_array.shape[0]
    pj_array = row_array(pj, "PJ")
    if pj_array.shape[1] == 0:
        raise ValueError("PJ must contain at least one transitory state")
    if np.any(pj_array < 0) or not np.allclose(
        pj_array.sum(axis=1), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("PJ must be nonnegative with rows summing to 1")

    if pj_array.shape[0] == 1:
        pj_array = np.tile(pj_array, (s * s, 1))
    elif pj_array.shape[0] == s:
        pj_array = np.repeat(pj_array, s, axis=0)
    elif pj_array.shape[0] != s * s:
        raise ValueError("the row count of PJ must be 1, S, or S^2")

    v_array = real_finite_array(v, "V")
    if v_array.size == 1:
        v_array = np.full((s, s), float(v_array.reshape(-1)[0]))
    elif v_array.ndim != 2 or v_array.shape != (s, s):
        raise ValueError("V must be a scalar or an S by S matrix")
    if np.any((v_array < 0) | (v_array > 1)):
        raise ValueError("entries of V must lie in [0, 1]")

    return ps_array, pj_array, v_array


def expand_state_rows(
    value: ArrayLike,
    name: str,
    s: int,
    columns: int,
    *,
    allow_one_row: bool = False,
) -> FloatArray:
    """Expand an S-row array so row ``S*s+t`` represents state pair ``(s,t)``."""

    array = row_array(value, name)
    if array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    if allow_one_row and array.shape[0] == 1:
        return np.tile(array, (s * s, 1))
    if array.shape[0] == s:
        return np.repeat(array, s, axis=0)
    if array.shape[0] != s * s:
        expected = f"1, {s}, or {s * s}" if allow_one_row else f"{s} or {s * s}"
        raise ValueError(f"the row count of {name} must be {expected}")
    return array
