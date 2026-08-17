import numpy as np
import pytest

from pareto_extrapolation import get_zeta


def example_parameters() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    beta = 0.96
    ps = np.array([[0.95, 0.05], [0.05, 0.95]])
    pj = np.array([0.5, 0.5])
    mu = np.array([0.03, 0.07])
    sigma = 0.10
    growth = beta * np.exp(np.column_stack((mu - sigma, mu + sigma)))
    return ps, pj, 0.96, growth


def test_get_zeta_matches_matlab_example() -> None:
    ps, pj, survival, growth = example_parameters()
    zeta, type_dist = get_zeta(ps, pj, survival, growth, (0.1, 10.0))
    assert zeta == pytest.approx(1.7218991409289821, abs=2e-12)
    assert type_dist.shape == (2,)
    assert type_dist.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bound", [(2.0, 3.0), (1.0, 2.0)])
def test_get_zeta_accepts_root_on_search_bound(bound: tuple[float, float]) -> None:
    zeta, type_dist = get_zeta(1.0, 1.0, 0.25, 2.0, bound)
    assert zeta == pytest.approx(2.0)
    np.testing.assert_allclose(type_dist, [1.0])


def test_get_zeta_rejects_nonfinite_probabilities() -> None:
    with pytest.raises(ValueError, match="finite"):
        get_zeta([[np.nan]], [1.0], 0.9, [1.1])
