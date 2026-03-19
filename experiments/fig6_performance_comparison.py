"""
Replicate Figure 6 from Mohajerin Esfahani & Kuhn (2018):
Out-of-sample performance, certificate, and reliability for
performance-driven SAA and Wasserstein solutions as a function of N.

Three modes:
  (a-c) Holdout method
  (d-f) k-fold cross validation
  (g-i) Optimal epsilon (oracle, requires knowledge of P)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from tools.generate import generate_returns
from tools.models.saa_solver import solve_saa
from tools.models.dro_solver import solve_dro
from tools.evaluate import out_of_sample_performance
from tools.calibration import holdout, kfold_cv, optimal_radius

M = 10
ALPHA = 0.2
RHO = 10.0
N_SIMS = 200
SAMPLE_SIZES = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000]

# Candidate radii: E = {b * 10^c : b in {1,...,9}, c in {-3,-2,-1}} union {0}
EPSILON_CANDIDATES = np.sort(np.concatenate([
    [b * 10**c for b in range(1, 10) for c in [-3, -2, -1]]
]))

# True optimal value J* (computed with huge sample SAA)
_xi_huge = generate_returns(1_000_000, m=M, seed=9999)
_, _, J_STAR = solve_saa(_xi_huge, alpha=ALPHA, rho=RHO)
print(f"J* ≈ {J_STAR:.4f}")


def run_single_method(method, xi, seed):
    """Run a single calibration method. Returns (x, cert)."""
    if method == 'saa':
        x, _, cert = solve_saa(xi, alpha=ALPHA, rho=RHO)
        return x, cert
    elif method == 'holdout':
        _, x, cert = holdout(xi, EPSILON_CANDIDATES,
                             alpha=ALPHA, rho=RHO, seed=seed)
        return x, cert
    elif method == 'kfold':
        _, x, cert = kfold_cv(xi, EPSILON_CANDIDATES, k=5,
                              alpha=ALPHA, rho=RHO, seed=seed)
        return x, cert
    elif method == 'optimal':
        _, x, cert = optimal_radius(xi, EPSILON_CANDIDATES,
                                    alpha=ALPHA, rho=RHO)
        return x, cert


def run_experiment(methods):
    results = {m: {N: {'oos': [], 'cert': []} for N in SAMPLE_SIZES}
               for m in methods}

    for N in SAMPLE_SIZES:
        print(f"N = {N}")
        for sim in range(N_SIMS):
            if (sim + 1) % 50 == 0:
                print(f"  sim {sim + 1}/{N_SIMS}")
            xi = generate_returns(N, m=M, seed=sim * 10000 + N)

            for method in methods:
                try:
                    x, cert = run_single_method(method, xi, seed=sim)
                    J_oos = out_of_sample_performance(x, alpha=ALPHA, rho=RHO)
                    results[method][N]['oos'].append(J_oos)
                    results[method][N]['cert'].append(cert)
                except Exception:
                    results[method][N]['oos'].append(np.nan)
                    results[method][N]['cert'].append(np.nan)

    return results


def _safe_stats(values):
    """Compute mean, q20, q80 with clipping to avoid overflow."""
    arr = np.array(values, dtype=float)
    arr = np.clip(arr, -1e6, 1e6)  # clip extreme values
    return np.nanmean(arr), np.nanquantile(arr, 0.2), np.nanquantile(arr, 0.8)


def plot_row(results, methods, labels, colors, row_title, fig, axes_row):
    """Plot one row (3 panels): OOS performance, certificate, reliability."""
    Ns = SAMPLE_SIZES

    # Panel 1: Out-of-sample performance
    ax = axes_row[0]
    for method, label, color in zip(methods, labels, colors):
        stats = [_safe_stats(results[method][N]['oos']) for N in Ns]
        means = [s[0] for s in stats]
        q20 = [s[1] for s in stats]
        q80 = [s[2] for s in stats]
        ax.fill_between(Ns, q20, q80, alpha=0.15, color=color)
        ax.plot(Ns, means, 'o-', color=color, label=label, markersize=3)
    ax.axhline(J_STAR, color='gray', linestyle='--', linewidth=0.8, label='$J^\\star$')
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('Out-of-sample performance')
    ax.set_title(f'{row_title}')
    ax.legend(fontsize=7)

    # Panel 2: Certificate
    ax = axes_row[1]
    for method, label, color in zip(methods, labels, colors):
        stats = [_safe_stats(results[method][N]['cert']) for N in Ns]
        means = [s[0] for s in stats]
        ax.plot(Ns, means, 'o-', color=color, label=label, markersize=3)
    ax.axhline(J_STAR, color='gray', linestyle='--', linewidth=0.8, label='$J^\\star$')
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('Certificate')
    ax.set_title(f'{row_title}')
    ax.legend(fontsize=7)

    # Panel 3: Reliability
    ax = axes_row[2]
    for method, label, color in zip(methods, labels, colors):
        rel = []
        for N in Ns:
            oos_arr = np.array(results[method][N]['oos'])
            cert_arr = np.array(results[method][N]['cert'])
            valid = ~(np.isnan(oos_arr) | np.isnan(cert_arr))
            if valid.sum() > 0:
                rel.append(np.mean(oos_arr[valid] <= cert_arr[valid]))
            else:
                rel.append(np.nan)
        ax.plot(Ns, rel, 'o-', color=color, label=label, markersize=3)
    ax.set_xscale('log')
    ax.set_xlabel('N')
    ax.set_ylabel('Reliability')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'{row_title}')
    ax.legend(fontsize=7)


def main():
    methods_list = ['saa', 'holdout']
    labels_list = ['SAA', 'Wass (holdout)']
    colors_list = ['C0', 'C2']

    print("=== Running holdout experiment ===")
    results_hm = run_experiment(['saa', 'holdout'])

    print("=== Running k-fold CV experiment ===")
    results_cv = run_experiment(['saa', 'kfold'])

    print("=== Running optimal experiment ===")
    results_opt = run_experiment(['saa', 'optimal'])

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    plot_row(results_hm, ['saa', 'holdout'], ['SAA', 'Wass (holdout)'],
             ['C0', 'C2'], 'Holdout method', fig, axes[0])
    plot_row(results_cv, ['saa', 'kfold'], ['SAA', 'Wass (k-fold)'],
             ['C0', 'C2'], 'k-fold cross validation', fig, axes[1])
    plot_row(results_opt, ['saa', 'optimal'], ['SAA', 'Wass (optimal)'],
             ['C0', 'C2'], 'Optimal size', fig, axes[2])

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fig6_performance_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/fig6_performance_comparison.png', bbox_inches='tight', dpi=150)
    print("Saved results/fig6_performance_comparison.{pdf,png}")
    plt.close()


if __name__ == '__main__':
    main()