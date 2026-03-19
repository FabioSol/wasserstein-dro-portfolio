"""
Out-of-sample performance evaluation for mean-CVaR portfolios.

The out-of-sample performance is:
    J(x) = E^P[ -<x,xi> ] + rho * P-CVaR_alpha( -<x,xi> )

For a normal distribution of returns, this can be computed analytically.
We also provide a Monte Carlo estimator.
"""
import numpy as np
from scipy.stats import norm


def portfolio_loss_stats(x, m=10, systematic_sigma=0.02,
                         idiosyncratic_mu_coef=0.03,
                         idiosyncratic_sigma_coef=0.025):
    """
    Compute mean and std of the portfolio loss L = -<x, xi>.

    Under the CAPM model:
        xi_i = psi + zeta_i
        psi ~ N(0, sigma_s^2)
        zeta_i ~ N(i*mu_c, (i*sigma_c)^2)  independent

    Portfolio return R = <x, xi> has:
        E[R] = sum_i x_i * i * mu_c
        Var[R] = sigma_s^2 * (sum_i x_i)^2 + sum_i x_i^2 * (i*sigma_c)^2

    Loss L = -R, so E[L] = -E[R], Std[L] = Std[R].
    """
    indices = np.arange(1, m + 1, dtype=float)
    means = indices * idiosyncratic_mu_coef
    variances = (indices * idiosyncratic_sigma_coef) ** 2

    mu_R = x @ means
    var_R = (systematic_sigma ** 2) * (np.sum(x) ** 2) + x ** 2 @ variances
    std_R = np.sqrt(var_R)

    mu_L = -mu_R
    std_L = std_R
    return mu_L, std_L


def analytical_cvar_normal(mu, sigma, alpha):
    """
    CVaR_alpha of a N(mu, sigma^2) random variable.

    CVaR_alpha(L) = mu + sigma * phi(Phi^{-1}(1-alpha)) / alpha

    where phi is the standard normal PDF and Phi^{-1} is the quantile function.
    See Rockafellar & Uryasev (2000), p. 29.
    """
    z = norm.ppf(1 - alpha)
    return mu + sigma * norm.pdf(z) / alpha


def out_of_sample_performance(x, alpha=0.2, rho=10.0, **kwargs):
    """
    Compute the exact out-of-sample performance analytically.

    J(x) = E[-<x,xi>] + rho * CVaR_alpha(-<x,xi>)

    Parameters
    ----------
    x : ndarray of shape (m,), portfolio weights
    alpha, rho : CVaR/risk parameters
    **kwargs : passed to portfolio_loss_stats

    Returns
    -------
    float, the out-of-sample performance J(x)
    """
    mu_L, std_L = portfolio_loss_stats(x, **kwargs)
    cvar = analytical_cvar_normal(mu_L, std_L, alpha)
    return mu_L + rho * cvar


def out_of_sample_performance_mc(x, xi_test, alpha=0.2, rho=10.0):
    """
    Estimate out-of-sample performance via Monte Carlo.

    Parameters
    ----------
    x : ndarray of shape (m,)
    xi_test : ndarray of shape (N_test, m)
    alpha, rho : CVaR/risk parameters

    Returns
    -------
    float, estimated J(x)
    """
    losses = -(xi_test @ x)  # -<x, xi>
    mean_loss = np.mean(losses)

    # CVaR = E[L | L >= VaR_alpha] via sorting
    sorted_losses = np.sort(losses)
    cutoff = int(np.ceil((1 - alpha) * len(losses)))
    cvar = np.mean(sorted_losses[cutoff:])

    return mean_loss + rho * cvar