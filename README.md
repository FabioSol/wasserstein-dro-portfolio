# Wasserstein DRO Portfolio Optimization

Replication of the mean-CVaR portfolio optimization experiments from
**Mohajerin Esfahani & Kuhn (2018)**, "Data-driven distributionally robust
optimization using the Wasserstein metric" (*Mathematical Programming*,
170(1), 115--166).

## Overview

This project implements a Wasserstein distributionally robust optimization
(DRO) approach to portfolio selection with mean-CVaR objectives, and
replicates the numerical experiments (Figures 4, 5, 6, and 8) from the paper.

The key idea: instead of optimizing against the empirical distribution of
returns (SAA), the DRO model hedges against all distributions within a
Wasserstein ball of radius epsilon around the empirical measure, providing
finite-sample performance guarantees.

## Project Structure

```
wasserstein-dro-portfolio/
├── tools/                      # Core library
│   ├── generate.py             # CAPM-based return generation
│   ├── evaluate.py             # Out-of-sample performance (analytical + MC)
│   ├── calibration.py          # Wasserstein radius calibration methods
│   └── models/
│       ├── saa_solver.py       # Sample Average Approximation solver
│       └── dro_solver.py       # Wasserstein DRO solver
├── experiments/                # Experiment scripts
│   ├── fig4_portfolio_composition.py   # Portfolio weights vs epsilon
│   ├── fig5_out_of_sample.py           # OOS performance vs epsilon
│   ├── fig6_performance_comparison.py  # SAA vs DRO calibration methods
│   ├── fig8_radius_vs_N.py             # Wasserstein radius vs sample size
│   └── run_all.py                      # Run all experiments
├── results/                    # Generated figures (PDF + PNG)
└── docs/                       # Sphinx documentation
```

## Installation

```bash
pip install numpy scipy cvxpy clarabel matplotlib
```

## Usage

### Run all experiments

```bash
python experiments/run_all.py
```

### Quick test run (5 simulations, reduced sample sizes)

```bash
python experiments/run_all.py --quick
```

### Run a specific figure

```bash
python experiments/run_all.py --fig 4    # Figure 4 only
python experiments/run_all.py --fig 5    # Figure 5 only
python experiments/run_all.py --fig 6    # Figure 6 only
python experiments/run_all.py --fig 8    # Figure 8 only
```

### Individual experiment scripts

```bash
python experiments/fig4_portfolio_composition.py
```

## Model

The loss function for asset $i$ is $\ell(x, \xi) = -\langle x, \xi \rangle$, and
the objective is:

$$J(x) = \mathbb{E}[\ell(x,\xi)] + \rho \cdot \text{CVaR}_\alpha(\ell(x,\xi))$$

The DRO formulation minimizes the worst-case $J(x)$ over all distributions
$\mathbb{Q}$ within a type-1 Wasserstein ball of radius $\varepsilon$
around the empirical distribution $\hat{\mathbb{P}}_N$.

Using the 1-norm Wasserstein metric and unconstrained support $\Xi = \mathbb{R}^m$,
the dual reformulation becomes a finite convex program (LP).

## Experiments

| Figure | Description |
|--------|-------------|
| **4** | Portfolio composition (stacked area) vs Wasserstein radius. Shows convergence to equally-weighted portfolio as $\varepsilon \to \infty$ (Proposition 7.2). |
| **5** | Out-of-sample performance and reliability vs $\varepsilon$. Performance improves up to a critical radius, then deteriorates. |
| **6** | Comparison of SAA vs DRO with holdout, k-fold CV, and oracle calibration. 3x3 grid: OOS performance, certificate, and reliability vs sample size $N$. |
| **8** | Selected Wasserstein radius vs $N$ on log-log scale. All methods decay as $\sim N^{-1/2}$. |

## Parameters

Following the paper (Section 7):
- $m = 10$ assets
- $\alpha = 0.2$ (CVaR level)
- $\rho = 10$ (risk-aversion)
- CAPM model: $\sigma_s = 0.02$, $\mu_c = 0.03$, $\sigma_c = 0.025$

## References

Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally
robust optimization using the Wasserstein metric: performance guarantees
and tractable reformulations. *Mathematical Programming*, 170(1), 115--166.