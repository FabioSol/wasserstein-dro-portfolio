"""
Replicate Figure 8 from Mohajerin Esfahani & Kuhn (2018):
Wasserstein radius vs sample size N for different calibration methods.

Shows that all radii decay as ~N^{-1/2}, which is faster than the
conservative a priori rate N^{-1/m} from Theorem 3.4.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tools.generate import generate_returns
from tools.models.dro_solver import solve_dro
from tools.evaluate import out_of_sample_performance, out_of_sample_performance_mc
from tools.calibration import holdout, kfold_cv, optimal_radius

M = 10
ALPHA = 0.2
RHO = 10.0
N_SIMS = 200
SAMPLE_SIZES = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000]

EPSILON_CANDIDATES = np.sort(np.concatenate([
    [b * 10**c for b in range(1, 10) for c in [-3, -2, -1]]
]))


def run_experiment():
    results = {method: {N: [] for N in SAMPLE_SIZES}
               for method in ['holdout', 'kfold', 'optimal']}

    for N in SAMPLE_SIZES:
        print(f"N = {N}")
        for sim in range(N_SIMS):
            if (sim + 1) % 50 == 0:
                print(f"  sim {sim + 1}/{N_SIMS}")
            xi = generate_returns(N, m=M, seed=sim * 10000 + N)

            # Holdout
            try:
                eps_hm, _, _ = holdout(xi, EPSILON_CANDIDATES,
                                       alpha=ALPHA, rho=RHO, seed=sim)
                results['holdout'][N].append(eps_hm)
            except Exception:
                results['holdout'][N].append(np.nan)

            # k-fold CV
            try:
                eps_cv, _, _ = kfold_cv(xi, EPSILON_CANDIDATES, k=5,
                                        alpha=ALPHA, rho=RHO, seed=sim)
                results['kfold'][N].append(eps_cv)
            except Exception:
                results['kfold'][N].append(np.nan)

            # Optimal
            try:
                eps_opt, _, _ = optimal_radius(xi, EPSILON_CANDIDATES,
                                               alpha=ALPHA, rho=RHO)
                results['optimal'][N].append(eps_opt)
            except Exception:
                results['optimal'][N].append(np.nan)

    return results


def plot_results(results):
    fig, ax = plt.subplots(figsize=(7, 5))

    Ns = np.array(SAMPLE_SIZES)
    styles = {
        'holdout': ('v--', 'C0', r'$\hat{\varepsilon}_N^{\rm hm}$ Holdout'),
        'kfold': ('s--', 'C1', r'$\hat{\varepsilon}_N^{\rm cv}$ k-fold'),
        'optimal': ('o-', 'C2', r'$\hat{\varepsilon}_N^{\rm opt}$ Optimal'),
    }

    for method, (marker, color, label) in styles.items():
        means = [np.nanmean(results[method][N]) for N in SAMPLE_SIZES]
        ax.plot(Ns, means, marker, color=color, label=label, markersize=5)

    # Reference line ~ N^{-1/2}
    ax.plot(Ns, 0.5 * Ns.astype(float) ** (-0.5), 'k:', linewidth=0.8,
            label=r'$\propto N^{-1/2}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('Average Wasserstein radii')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig8_radius_vs_N.pdf', bbox_inches='tight')
    plt.savefig('results/fig8_radius_vs_N.png', bbox_inches='tight', dpi=150)
    print("Saved results/fig8_radius_vs_N.{pdf,png}")
    plt.close()


if __name__ == '__main__':
    results = run_experiment()
    plot_results(results)