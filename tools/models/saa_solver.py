"""
Sample Average Approximation (SAA) solver for the mean-CVaR portfolio problem.

Solves (Eq. 4 in the paper, specialized to the mean-CVaR loss):

    min_{x in X, tau}  (1/N) sum_i max_k { a_k * <x, xi_i> + b_k * tau }

where K=2 with:
    a_1 = -1,           b_1 = rho
    a_2 = -(1+rho/a),   b_2 = rho*(1 - 1/alpha)

This is a linear program when formulated with epigraphical variables.
"""
import numpy as np
import cvxpy as cp


def solve_saa(xi_train, alpha=0.2, rho=10.0):
    """
    Parameters
    ----------
    xi_train : ndarray of shape (N, m)
    alpha : float, CVaR confidence level
    rho : float, risk-aversion parameter

    Returns
    -------
    x_val : ndarray of shape (m,), optimal portfolio weights
    tau_val : float, optimal VaR threshold
    obj_val : float, optimal SAA objective value
    """
    N, m = xi_train.shape

    a_coefs = np.array([-1.0, -(1.0 + rho / alpha)])
    b_coefs = np.array([rho, rho * (1.0 - 1.0 / alpha)])
    K = len(a_coefs)

    x = cp.Variable(m, nonneg=True)
    tau = cp.Variable()
    s = cp.Variable(N)

    constraints = [cp.sum(x) == 1]

    # Vectorized: for each k, a_k * Xi @ x + b_k * tau <= s
    # Xi @ x has shape (N,), broadcast with scalar b_k * tau
    Xi = xi_train  # (N, m)
    for k in range(K):
        constraints.append(a_coefs[k] * (Xi @ x) + b_coefs[k] * tau <= s)

    objective = cp.Minimize((1.0 / N) * cp.sum(s))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"SAA solver failed: {prob.status}")

    return x.value, tau.value, prob.value