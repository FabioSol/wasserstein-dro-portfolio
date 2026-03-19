import numpy as np


def generate_returns(N,
                     m=10,
                     systematic_sigma=0.02,
                     idiosyncratic_mu_coef=0.03,
                     idiosyncratic_sigma_coef=0.025,
                     seed=None):
    r"""
    this function generates returns based on CAPM model where:

    systematic risk factor $\psi \sim \mathcal{N}(0, \sigma_s)$

    idiosyncratic risk factor $\zeta_i \sim \mathcal{N}(i*\mu_c, i*\sigma_c)$

    thus returns:
    $$\xi_i=\psi+\zeta_i$$

    :param N: number of samples
    :param m: number of assets
    :param systematic_sigma: $\sigma_s$
    :param idiosyncratic_mu_coef: $\mu_c$
    :param idiosyncratic_sigma_coef: $\sigma_c$
    :param seed: random seed
    :return: $\xi_i$
    """
    rng = np.random.default_rng(seed)

    # systematic factor
    psi = rng.normal(0, systematic_sigma, size=N)  # N(0, 2%)

    # idiosyncratic factors
    means = np.array([i * idiosyncratic_mu_coef for i in range(1, m + 1)])  # i×3%
    stds = np.array([i * idiosyncratic_sigma_coef for i in range(1, m + 1)])  # i×2.5%
    zeta = rng.normal(0, 1, size=(N, m)) * stds + means  # (N, m)

    # asset returns
    xi = psi[:, None] + zeta  # (N, m)
    return xi

if __name__ == '__main__':
    N = 1000
    xi = generate_returns(N)
    print(xi.mean(axis=0))
    print(xi.std(axis=0))
    

