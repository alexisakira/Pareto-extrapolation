import numpy as np
import pytest

from pareto_extrapolation import exp_grid


def test_exp_grid_includes_endpoints_and_midpoint() -> None:
    grid = exp_grid(0.0, 10_000.0, 1.0, 101)
    assert grid.shape == (101,)
    assert grid[0] == 0.0
    assert grid[-1] == 10_000.0
    assert grid[50] == pytest.approx(1.0)
    assert np.all(np.diff(grid) > 0)


@pytest.mark.parametrize("n", [1, 2.5, np.inf])
def test_exp_grid_rejects_invalid_size(n: float) -> None:
    with pytest.raises(ValueError):
        exp_grid(0.0, 100.0, 1.0, n)
