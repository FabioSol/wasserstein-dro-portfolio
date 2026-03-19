import numpy as np
import cvxpy as cp


def solve_saa(xi_train, alpha=0.2, rho=10):
    N, m = xi_train.shape

    x = cp.Variable(m, nonneg=True)  # portfolio weights
    tau = cp.Variable()  # VaR threshold
    s = cp.Variable(N)  # auxiliary per-sample

    a1, a2 = -1, -(1 + rho / alpha)
    b1, b2 = rho, rho * (1 - 1 / alpha)

    constraints = [cp.sum(x) == 1]
    for i in range(N):
        constraints += [
            a1 * (x @ xi_train[i]) + b1 * tau <= s[i],
            a2 * (x @ xi_train[i]) + b2 * tau <= s[i],
        ]

    objective = cp.Minimize((1 / N) * cp.sum(s))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    return x.value, tau.value, prob.value