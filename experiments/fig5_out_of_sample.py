"""
Replicate Figure 5 from Mohajerin Esfahani & Kuhn (2018):
Out-of-sample performance J(x_N(eps)) and reliability P^N[J(x_N(eps)) <= J_N(eps)]
as a function of the Wasserstein radius epsilon.

Key insight: performance improves up to a critical eps_crit then deteriorates.
Reliability is nondecreasing in epsilon and sharply rises near eps_crit.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tools.generate import generate_returns
from tools.models.dro_solver import solve_dro
from tools.evaluate import out_of_sample_performance

M = 10
ALPHA = 0.2
RHO = 10.0
N_SIMS = 200
SAMPLE_SIZES = [30, 300, 3000]
EPSILONS = np.logspace(-4, -0.5, 25)


def run_experiment():
    results = {}
    for N in SAMPLE_SIZES:
        print(f"N = {N}")
        oos_perf = np.zeros((N_SIMS, len(EPSILONS)))
        certificates = np.zeros((N_SIMS, len(EPSILONS)))

        for sim in range(N_SIMS):
            if (sim + 1) % 50 == 0:
                print(f"  sim {sim + 1}/{N_SIMS}")
            xi = generate_returns(N, m=M, seed=sim)

            for j, eps in enumerate(EPSILONS):
                try:
                    x_val, _, cert = solve_dro(xi, eps, alpha=ALPHA, rho=RHO)
                    J_oos = out_of_sample_performance(x_val, alpha=ALPHA, rho=RHO)
                    oos_perf[sim, j] = J_oos
                    certificates[sim, j] = cert
                except RuntimeError:
                    oos_perf[sim, j] = np.nan
                    certificates[sim, j] = np.nan

        results[N] = {
            'oos': oos_perf,
            'cert': certificates,
        }

    return results


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, N in enumerate(SAMPLE_SIZES):
        ax = axes[idx]
        oos = results[N]['oos']
        cert = results[N]['cert']

        # Out-of-sample performance: mean and 20/80 quantiles
        mean_oos = np.nanmean(oos, axis=0)
        q20 = np.nanquantile(oos, 0.2, axis=0)
        q80 = np.nanquantile(oos, 0.8, axis=0)

        ax.fill_between(EPSILONS, q20, q80, alpha=0.3, color='C0')
        ax.plot(EPSILONS, mean_oos, '-', color='C0', linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel('Out-of-sample performance', color='C0')
        ax.set_title(f'({"abc"[idx]}) N = {N}')
        ax.tick_params(axis='y', labelcolor='C0')

        # Reliability on right axis
        ax2 = ax.twinx()
        reliability = np.nanmean(oos <= cert, axis=0)
        ax2.plot(EPSILONS, reliability, '--', color='C3', linewidth=1.5)
        ax2.set_ylabel('Reliability', color='C3')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor='C3')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig5_oos_performance.pdf', bbox_inches='tight')
    plt.savefig('results/fig5_oos_performance.png', bbox_inches='tight', dpi=150)
    print("Saved results/fig5_oos_performance.{pdf,png}")
    plt.close()


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)