"""
Distributionally Robust Optimization (DRO) solver for the mean-CVaR portfolio
problem using the Wasserstein ambiguity set.

Implements Eq. (27) from Mohajerin Esfahani & Kuhn (2018):

    min_{x,tau,lambda,s,gamma}  lambda*epsilon + (1/N) sum_i s_i
    s.t.  x in X (probability simplex)
          b_k*tau + a_k*<x, xi_i> + <gamma_ik, d - C*xi_i> <= s_i   for all i,k
          ||C^T gamma_ik - a_k x||_* <= lambda                       for all i,k
          gamma_ik >= 0                                               for all i,k

Using the 1-norm in the uncertainty space => dual norm is the inf-norm.
The support set is Xi = R^m (no constraints) => C=0, d=0 => gamma drops out,
and the norm constraint simplifies to ||a_k x||_inf <= lambda.
"""
import numpy as np
import cvxpy as cp


def solve_dro(xi_train, epsilon, alpha=0.2, rho=10.0):
    """
    Solve the Wasserstein DRO mean-CVaR portfolio problem.

    Uses 1-norm Wasserstein metric (dual norm = inf-norm).
    Support set Xi = R^m (unconstrained).

    Parameters
    ----------
    xi_train : ndarray of shape (N, m)
    epsilon : float, Wasserstein ball radius
    alpha : float, CVaR confidence level
    rho : float, risk-aversion parameter

    Returns
    -------
    x_val : ndarray of shape (m,), optimal portfolio weights
    tau_val : float, optimal VaR threshold
    obj_val : float, optimal objective value (certificate)
    """
    N, m = xi_train.shape

    a_coefs = np.array([-1.0, -(1.0 + rho / alpha)])
    b_coefs = np.array([rho, rho * (1.0 - 1.0 / alpha)])
    K = len(a_coefs)

    x = cp.Variable(m, nonneg=True)
    tau = cp.Variable()
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N)

    constraints = [cp.sum(x) == 1]

    # With Xi = R^m: C=0, d=0 so gamma drops out.
    # Constraint 1: b_k*tau + a_k*(Xi @ x) <= s  for each k
    # Constraint 2: ||a_k * x||_inf <= lambda  for each k
    Xi = xi_train  # (N, m)
    for k in range(K):
        constraints.append(a_coefs[k] * (Xi @ x) + b_coefs[k] * tau <= s)
        constraints.append(cp.norm(a_coefs[k] * x, 'inf') <= lam)

    objective = cp.Minimize(lam * epsilon + (1.0 / N) * cp.sum(s))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"DRO solver failed: {prob.status}")

    return x.value, tau.value, prob.value


def solve_dro_support(xi_train, epsilon, C, d, alpha=0.2, rho=10.0):
    """
    Solve the Wasserstein DRO mean-CVaR portfolio problem with support
    constraints Xi = {xi : C*xi <= d}.

    Uses 1-norm Wasserstein metric (dual norm = inf-norm).

    Parameters
    ----------
    xi_train : ndarray of shape (N, m)
    epsilon : float, Wasserstein ball radius
    C : ndarray of shape (p, m), support constraint matrix
    d : ndarray of shape (p,), support constraint RHS
    alpha : float, CVaR confidence level
    rho : float, risk-aversion parameter

    Returns
    -------
    x_val : ndarray of shape (m,), optimal portfolio weights
    tau_val : float, optimal VaR threshold
    obj_val : float, optimal objective value (certificate)
    """
    N, m = xi_train.shape
    p = C.shape[0]

    a_coefs = np.array([-1.0, -(1.0 + rho / alpha)])
    b_coefs = np.array([rho, rho * (1.0 - 1.0 / alpha)])
    K = len(a_coefs)

    x = cp.Variable(m, nonneg=True)
    tau = cp.Variable()
    lam = cp.Variable(nonneg=True)
    s = cp.Variable(N)
    gamma = {}
    for i in range(N):
        for k in range(K):
            gamma[i, k] = cp.Variable(p, nonneg=True)

    constraints = [cp.sum(x) == 1]

    Xi = xi_train
    for k in range(K):
        for i in range(N):
            g = gamma[i, k]
            constraints.append(
                b_coefs[k] * tau
                + a_coefs[k] * (x @ Xi[i])
                + g @ (d - C @ Xi[i]) <= s[i]
            )
            residual = C.T @ g - a_coefs[k] * x
            constraints.append(cp.norm(residual, 'inf') <= lam)

    objective = cp.Minimize(lam * epsilon + (1.0 / N) * cp.sum(s))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"DRO solver (support) failed: {prob.status}")

    return x.value, tau.value, prob.value