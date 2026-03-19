"""
Replicate Figure 4 from Mohajerin Esfahani & Kuhn (2018):
Optimal portfolio composition as a function of the Wasserstein radius epsilon,
averaged over multiple simulations.

Shows that as epsilon -> inf, the portfolio converges to the equally weighted
portfolio (1/m, ..., 1/m), confirming Proposition 7.2.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tools.generate import generate_returns
from tools.models.dro_solver import solve_dro

# Parameters matching the paper
M = 10
ALPHA = 0.2
RHO = 10.0
N_SIMS = 200
SAMPLE_SIZES = [30, 300, 3000]
EPSILONS = np.logspace(-3, 0, 30)


def run_experiment():
    results = {}
    for N in SAMPLE_SIZES:
        print(f"N = {N}")
        all_weights = np.zeros((N_SIMS, len(EPSILONS), M))

        for sim in range(N_SIMS):
            if (sim + 1) % 50 == 0:
                print(f"  sim {sim + 1}/{N_SIMS}")
            xi = generate_returns(N, m=M, seed=sim)

            for j, eps in enumerate(EPSILONS):
                try:
                    x_val, _, _ = solve_dro(xi, eps, alpha=ALPHA, rho=RHO)
                    all_weights[sim, j, :] = x_val
                except RuntimeError:
                    all_weights[sim, j, :] = np.nan

        results[N] = np.nanmean(all_weights, axis=0)  # (len_eps, m)

    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, N in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        weights = results[N]  # (len_eps, m)

        # Sort weights in ascending order at each epsilon for stacking
        sorted_weights = np.sort(weights, axis=1)

        # Stacked area chart
        ax.stackplot(EPSILONS, sorted_weights.T,
                     colors=plt.cm.coolwarm(np.linspace(0, 1, M)))
        ax.set_xscale('log')
        ax.set_xlim(EPSILONS[0], EPSILONS[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Average portfolio weights')
        ax.set_title(f'({"abc"[idx]}) N = {N} training samples')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig4_portfolio_composition.pdf', bbox_inches='tight')
    plt.savefig('results/fig4_portfolio_composition.png', bbox_inches='tight', dpi=150)
    print("Saved results/fig4_portfolio_composition.{pdf,png}")
    plt.close()


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)