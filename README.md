# Pareto Extrapolation

MATLAB and Python implementations of the Pareto-extrapolation method developed
by Gouin-Bonenfant and Toda (2023). The method augments a conventional finite
grid with an analytical Pareto tail, making it possible to compute stationary
distributions and top wealth shares more accurately in heavy-tailed models.

The detailed mathematical and API guide is available as
[`PE/readme.pdf`](PE/readme.pdf), with its editable LaTeX source in
[`PE/readme.tex`](PE/readme.tex).

## Repository layout

```text
PE/
|-- matlab/                         MATLAB implementation and example
|-- python/
|   |-- src/pareto_extrapolation/   Python package
|   |-- examples/                   Python example
|   `-- tests/                      Parity and edge-case tests
|-- readme.tex                      Detailed documentation source
`-- readme.pdf                      Compiled documentation
```

## Introduction

The package provides four main operations:

1. Construct a shifted-log grid.
2. Compute the Pareto exponent and upper-tail type distribution.
3. Construct a sparse joint transition matrix with Pareto extrapolation and,
   optionally, its stationary distribution.
4. Compute top wealth shares, including for distributions with negative
   wealth and positive aggregate net wealth.

Transition matrices are row-stochastic. State-pair rows are ordered as
`(1,1), ..., (1,S), ..., (S,1), ..., (S,S)`, and the grid index varies fastest
in the joint state. Both implementations use the same ordering and numerical
conventions.

## MATLAB

The MATLAB implementation is in [`PE/matlab`](PE/matlab) and has been tested
with MATLAB R2024b. From the `PE` directory, run:

```matlab
addpath('matlab')
example
```

### `expGrid`

```matlab
grid = expGrid(a,b,c,N)
```

Returns an `N`-point shifted-log grid on `[a,b]`. The parameter `c` controls
grid concentration and must satisfy `a < c < (a+b)/2`.

### `getZeta`

```matlab
[zeta,typeDist] = getZeta(PS,PJ,V,G,zetaBound)
```

Computes the Pareto exponent and the distribution of exogenous states in the
upper tail. `zetaBound` is optional and defaults to `[1e-2,100]`.

### `getQ`

```matlab
[Q,piStar] = getQ(PS,PJ,V,x0,xGrid,gstjn,Gstj,zeta)
```

Constructs the sparse joint transition matrix. It can also compute the
stationary distribution. The asymptotic slopes `Gstj` and exponent `zeta` may
be omitted and computed internally.

### `getTopShares`

```matlab
topShare = getTopShares(topProb,wGrid,wDist,zeta)
```

Computes top wealth shares using truncation or Pareto extrapolation. Negative
wealth is allowed, so a valid top share can exceed one and the reported shares
need not be monotone across population groups. Aggregate net wealth must be
strictly positive.

### MATLAB example

[`PE/matlab/example.m`](PE/matlab/example.m) builds the example model, computes
the exponent and transition matrix, reports the top 0.1%, 1%, and 10% wealth
shares, and plots the wealth-tail probability.

## Python

The Python implementation requires Python 3.10 or later, NumPy, and SciPy.
From `PE/python`, install the package and optional example and test dependencies:

```console
python -m pip install -e ".[example,test]"
pytest
python examples/example.py
```

The public API is:

```python
from pareto_extrapolation import exp_grid, get_q, get_top_shares, get_zeta
```

### `exp_grid`

```python
grid = exp_grid(a, b, c, n)
```

NumPy counterpart of `expGrid`, returning a one-dimensional grid with exact
endpoints.

### `get_zeta`

```python
zeta, type_dist = get_zeta(ps, pj, v, g, zeta_bound=(1e-2, 100.0))
```

Computes the Pareto exponent and upper-tail type distribution using the same
array ordering as MATLAB.

### `get_q`

```python
q, pi_star = get_q(
    ps, pj, v, x0, x_grid, gstjn,
    g_stj=None, zeta=None,
    compute_stationary=True,
)
```

Returns `q` as a SciPy CSR sparse matrix and `pi_star` as a NumPy array. Set
`compute_stationary=False` to skip the stationary-distribution calculation.

### `get_top_shares`

```python
top_share = get_top_shares(top_prob, w_grid, w_dist, zeta=0.0)
```

Python counterpart of the revised MATLAB routine, including support for
negative wealth.

### Python example and tests

[`PE/python/examples/example.py`](PE/python/examples/example.py) reproduces the
MATLAB example. The test suite covers MATLAB parity, exact-bound roots,
negative wealth, transition-matrix construction, and invalid inputs.

## Citation

If you use these routines in research, please cite:

> Émilien Gouin-Bonenfant and Alexis Akira Toda (2023), "Pareto
> Extrapolation: An Analytical Framework for Studying Tail Inequality,"
> *Quantitative Economics* 14(1), 201-233.
> [https://doi.org/10.3982/QE1817](https://doi.org/10.3982/QE1817)

The Pareto-exponent characterization builds on:

> Brendan K. Beare and Alexis Akira Toda (2022), "Determination of Pareto
> Exponents in Economic Models Driven by Markov Multiplicative Processes,"
> *Econometrica* 90(4), 1811-1833.
> [https://doi.org/10.3982/ECTA17984](https://doi.org/10.3982/ECTA17984)

## Maintainer

Alexis Akira Toda, Department of Economics, Emory University

[alexis.akira.toda@emory.edu](mailto:alexis.akira.toda@emory.edu)

The software is provided without warranty. Users should verify numerical
accuracy and economic assumptions for their own applications.
