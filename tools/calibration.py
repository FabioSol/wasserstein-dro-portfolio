"""
Methods for calibrating the Wasserstein radius epsilon.

Implements the procedures from Section 7.2 of Mohajerin Esfahani & Kuhn (2018):
    - Holdout method
    - k-fold cross validation
    - Optimal radius (requires knowledge of P, for benchmarking only)
"""
import numpy as np
from .models.dro_solver import solve_dro
from .evaluate import out_of_sample_performance, out_of_sample_performance_mc


def holdout(xi, epsilon_candidates, alpha=0.2, rho=10.0,
            train_fraction=0.8, seed=None):
    """
    Holdout method: split data into train/validation, pick epsilon that
    minimizes estimated out-of-sample performance on the validation set.

    Returns
    -------
    best_eps : float
    best_x : ndarray of shape (m,)
    best_cert : float, J_N(best_eps)
    """
    rng = np.random.default_rng(seed)
    N = len(xi)
    idx = rng.permutation(N)
    N_T = int(train_fraction * N)

    xi_train = xi[idx[:N_T]]
    xi_val = xi[idx[N_T:]]

    best_perf = np.inf
    best_eps = epsilon_candidates[0]
    best_x = None
    best_cert = None

    for eps in epsilon_candidates:
        try:
            x_val, _, cert = solve_dro(xi_train, eps, alpha=alpha, rho=rho)
            perf = out_of_sample_performance_mc(x_val, xi_val, alpha=alpha, rho=rho)
            if perf < best_perf:
                best_perf = perf
                best_eps = eps
                best_x = x_val
                best_cert = cert
        except RuntimeError:
            continue

    return best_eps, best_x, best_cert


def kfold_cv(xi, epsilon_candidates, k=5, alpha=0.2, rho=10.0, seed=None):
    """
    k-fold cross validation: partition data into k folds, for each fold
    use as validation while training on the rest. Average the best epsilon
    across folds, then re-solve on all data.

    Returns
    -------
    best_eps : float
    x_val : ndarray of shape (m,)
    cert : float
    """
    rng = np.random.default_rng(seed)
    N = len(xi)
    idx = rng.permutation(N)
    folds = np.array_split(idx, k)

    best_epsilons = []
    for fold_idx in range(k):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        xi_train = xi[train_idx]
        xi_val = xi[val_idx]

        best_perf = np.inf
        best_eps_fold = epsilon_candidates[0]

        for eps in epsilon_candidates:
            try:
                x_sol, _, _ = solve_dro(xi_train, eps, alpha=alpha, rho=rho)
                perf = out_of_sample_performance_mc(x_sol, xi_val, alpha=alpha, rho=rho)
                if perf < best_perf:
                    best_perf = perf
                    best_eps_fold = eps
            except RuntimeError:
                continue

        best_epsilons.append(best_eps_fold)

    avg_eps = np.mean(best_epsilons)
    # Snap to nearest candidate
    best_eps = epsilon_candidates[np.argmin(np.abs(epsilon_candidates - avg_eps))]

    x_val, _, cert = solve_dro(xi, best_eps, alpha=alpha, rho=rho)
    return best_eps, x_val, cert


def optimal_radius(xi_train, epsilon_candidates, alpha=0.2, rho=10.0,
                    **eval_kwargs):
    """
    Find the epsilon that minimizes the TRUE out-of-sample performance.
    Requires knowledge of P (only for benchmarking).

    Returns
    -------
    best_eps : float
    best_x : ndarray of shape (m,)
    best_cert : float
    """
    best_perf = np.inf
    best_eps = epsilon_candidates[0]
    best_x = None
    best_cert = None

    for eps in epsilon_candidates:
        try:
            x_val, _, cert = solve_dro(xi_train, eps, alpha=alpha, rho=rho)
            perf = out_of_sample_performance(x_val, alpha=alpha, rho=rho,
                                             **eval_kwargs)
            if perf < best_perf:
                best_perf = perf
                best_eps = eps
                best_x = x_val
                best_cert = cert
        except RuntimeError:
            continue

    return best_eps, best_x, best_cert