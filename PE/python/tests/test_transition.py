import numpy as np

from pareto_extrapolation import get_q


def reference_inputs() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    ps = np.array([[0.8, 0.2], [0.3, 0.7]])
    pj = np.array([[0.6, 0.4], [0.25, 0.75]])
    survival = np.array([[0.9, 0.85], [0.8, 0.95]])
    grid = np.array([-1.0, 0.0, 1.0, 3.0, 10.0])
    slopes = np.array([[0.8, 1.1], [0.9, 1.2]])
    intercept = np.array([[0.4, 0.2], [0.6, 0.1]])
    law = np.zeros((2, grid.size * pj.shape[1]))
    for shock in range(pj.shape[1]):
        law[:, grid.size * shock : grid.size * (shock + 1)] = (
            slopes[:, [shock]] * grid + intercept[:, [shock]]
        )
    return ps, pj, survival, grid, slopes, law


def test_get_q_matches_matlab_reference() -> None:
    ps, pj, survival, grid, slopes, law = reference_inputs()
    q, pi_star = get_q(ps, pj, survival, 0.5, grid, law, slopes, 2.0)
    expected_q = np.array(
        [
            [0.432, 0.328, 0.04, 0, 0, 0.102, 0.083, 0.015, 0, 0],
            [0, 0.5296, 0.2704, 0, 0, 0, 0.1306, 0.0694, 0, 0],
            [0, 0.04, 0.6736, 0.0864, 0, 0, 0.015, 0.1646, 0.0204, 0],
            [0, 0.04, 0.0832, 0.6562285714285716, 0.02057142857142859, 0,
             0.015, 0.0252, 0.15494285714285716, 0.004857142857142862],
            [0, 0.04, 0.04, 0.07320095708219267, 0.6467990429178074, 0,
             0.015, 0.015, 0.017283559311073268, 0.15271644068892676],
            [0.198, 0.072, 0.03, 0, 0, 0.548625, 0.133875, 0.0175, 0, 0],
            [0, 0.216, 0.084, 0, 0, 0, 0.532875, 0.167125, 0, 0],
            [0, 0.03, 0.228, 0.042, 0, 0, 0.0175, 0.566125, 0.116375, 0],
            [0, 0.03, 0.03, 0.21942857142857142, 0.020571428571428567, 0,
             0.0175, 0.0175, 0.608, 0.057],
            [0, 0.03, 0.03, 0.0025416998986872463, 0.23745830010131275, 0,
             0.0175, 0.0175, 0.007042626802612578, 0.6579573731973873],
        ]
    )
    expected_pi = np.array(
        [
            0.0,
            0.09042466050626302,
            0.2708195245814838,
            0.1870134323012346,
            0.05174238261101852,
            0.0,
            0.05458465979783136,
            0.15842363697735526,
            0.13820295425876436,
            0.04878874896604893,
        ]
    )
    np.testing.assert_allclose(q.toarray(), expected_q, atol=3e-15, rtol=0)
    assert pi_star is not None
    np.testing.assert_allclose(pi_star, expected_pi, atol=2e-12, rtol=0)
    np.testing.assert_allclose(np.asarray(q.sum(axis=1)).reshape(-1), 1.0, atol=1e-14)
    assert np.linalg.norm(q.T @ pi_star - pi_star, ord=1) < 1e-10


def test_get_q_can_skip_stationary_distribution() -> None:
    ps, pj, survival, grid, slopes, law = reference_inputs()
    q, pi_star = get_q(
        ps,
        pj,
        survival,
        0.5,
        grid,
        law,
        slopes,
        2.0,
        compute_stationary=False,
    )
    assert q.shape == (10, 10)
    assert pi_star is None


def test_get_q_rejects_nonfinite_transition_probabilities() -> None:
    try:
        get_q(np.array([[np.nan]]), [1.0], 1.0, None, [0.0, 1.0], [[0.0, 1.0]], [1.0], 2.0)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("get_q accepted a NaN transition probability")
