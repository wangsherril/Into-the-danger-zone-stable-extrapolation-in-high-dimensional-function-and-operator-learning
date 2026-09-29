import numpy as np
from scipy.fft import idctn


# Return a sample of a Gaussian random field on [0,1]^2 with:
#       mean 0
#       covariance operator C = (-Delta + tau^2)^(-alpha)
# where Delta is the Laplacian with zero Neumann boundary conditions.


def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.normal(0.0, 1.0, size=(s, s))

    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s), indexing="xy")

    # coef = (pi^2*(K1.^2+K2.^2) + tau^2).^(-alpha/2);
    coef = tau ** (alpha - 1) * (np.pi**2 * (K1**2 + K2**2) + tau**2) ** (-alpha / 2)

    # coef = (pi^2*(K1.^2+K2.^2)).^(-alpha/2);

    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0.0

    # MATLAB idct2
    U = idctn(L, type=2, norm='ortho')

    return U