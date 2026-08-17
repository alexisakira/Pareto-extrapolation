"""Exponential grids."""

from __future__ import annotations

import numpy as np


def exp_grid(a: float, b: float, c: float, n: int) -> np.ndarray:
    """Return an ``n``-point shifted-log grid on the closed interval ``[a, b]``.

    ``c`` controls grid concentration and must satisfy ``a < c < (a+b)/2``.
    For odd ``n``, it is the middle grid point.
    """

    values = (a, b, c, n)
    if any(not np.isscalar(value) for value in values):
        raise ValueError("a, b, c, and n must be scalars")
    if any(not np.isreal(value) or not np.isfinite(value) for value in values):
        raise ValueError("a, b, c, and n must be finite and real")
    if a >= b or c <= a or c >= (a + b) / 2:
        raise ValueError("the inputs must satisfy a < c < (a+b)/2")
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < 2:
        raise ValueError("n must be an integer no smaller than 2")

    n = int(n)
    shift = (c**2 - a * b) / (a + b - 2 * c)
    if a + shift <= 0:
        raise ValueError("the shift is too small; choose a larger c")

    log_grid = np.linspace(np.log(a + shift), np.log(b + shift), n)
    grid = np.exp(log_grid) - shift
    grid[0] = a
    grid[-1] = b
    return grid
