import numpy as np
import pytest

from pareto_extrapolation import get_top_shares


def test_negative_wealth_is_not_clamped_under_truncation() -> None:
    result = get_top_shares([0.75, 1.0], [-1.0, 1.0], [0.25, 0.75], 0.0)
    np.testing.assert_allclose(result, [1.5, 1.0], atol=1e-14)


def test_negative_wealth_is_not_clamped_with_pareto_tail() -> None:
    result = get_top_shares([0.75, 1.0], [-1.0, 1.0], [0.25, 0.75], 2.0)
    np.testing.assert_allclose(result, [1.2, 1.0], atol=1e-14)


def test_zero_probability_at_largest_grid_point() -> None:
    result = get_top_shares([0.1, 1.0], [0.0, 1.0, 2.0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(result, [0.296, 1.0], atol=1e-14)


def test_nonpositive_aggregate_wealth_is_rejected() -> None:
    with pytest.raises(ValueError, match="aggregate net wealth"):
        get_top_shares(0.5, [-2.0, 1.0], [0.5, 0.5])
