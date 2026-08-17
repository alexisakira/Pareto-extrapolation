"""Minimal end-to-end example of the Python implementation."""

import matplotlib.pyplot as plt
import numpy as np

from pareto_extrapolation import exp_grid, get_q, get_top_shares, get_zeta


def main() -> None:
    beta = 0.96
    death_probability = 0.04
    survival = 1 - death_probability
    transition_probability = 0.05
    ps = np.array(
        [
            [1 - transition_probability, transition_probability],
            [transition_probability, 1 - transition_probability],
        ]
    )
    mu = np.array([0.03, 0.07])
    sigma = 0.10
    pj = np.array([0.5, 0.5])
    slopes = beta * np.exp(np.column_stack((mu - sigma, mu + sigma)))

    n = 100
    x_grid = exp_grid(0.0, 1e4, 1.0, n)
    gstjn = np.kron(slopes, x_grid)

    zeta, type_dist = get_zeta(ps, pj, survival, slopes, (0.1, 10.0))
    q, pi_star = get_q(ps, pj, survival, 1.0, x_grid, gstjn, slopes, zeta)
    assert pi_star is not None

    x_dist = pi_star.reshape(ps.shape[0], n).sum(axis=0)
    tail_probability = 1 - np.cumsum(x_dist)
    positive = tail_probability > 0

    top_prob = np.array([0.001, 0.01, 0.1])
    top_share = get_top_shares(top_prob, x_grid, x_dist, zeta)

    print(f"Pareto exponent zeta = {zeta:.10f}")
    print(f"Upper-tail type distribution = {type_dist}")
    for probability, share in zip(top_prob, top_share, strict=True):
        print(f"Top {100 * probability:g}% wealth share: {share:.10f}")
    print(f"Q shape = {q.shape}, nonzeros = {q.nnz}")

    plt.loglog(x_grid[positive], tail_probability[positive])
    plt.xlabel("Wealth")
    plt.ylabel("Tail probability")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
