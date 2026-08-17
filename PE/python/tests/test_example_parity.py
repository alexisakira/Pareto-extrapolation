import numpy as np

from pareto_extrapolation import exp_grid, get_q, get_top_shares, get_zeta


def test_end_to_end_example_matches_matlab() -> None:
    beta = 0.96
    ps = np.array([[0.95, 0.05], [0.05, 0.95]])
    pj = np.array([0.5, 0.5])
    mu = np.array([0.03, 0.07])
    sigma = 0.10
    slopes = beta * np.exp(np.column_stack((mu - sigma, mu + sigma)))
    grid = exp_grid(0.0, 1e4, 1.0, 100)
    law = np.kron(slopes, grid)

    zeta, _ = get_zeta(ps, pj, 0.96, slopes, (0.1, 10.0))
    q, pi_star = get_q(ps, pj, 0.96, 1.0, grid, law, slopes, zeta)
    assert pi_star is not None
    wealth_dist = pi_star.reshape(2, 100).sum(axis=0)
    top_share = get_top_shares([0.001, 0.01, 0.1], grid, wealth_dist, zeta)

    np.testing.assert_allclose(
        top_share,
        [0.07361939103625187, 0.17708532141804265, 0.42610240273978323],
        # MATLAB's interp1(...,'pchip') and SciPy's PchipInterpolator differ
        # slightly even when their probability and wealth knots agree.
        atol=1e-10,
        rtol=0,
    )
    assert np.linalg.norm(q.T @ pi_star - pi_star, ord=1) < 1e-10
