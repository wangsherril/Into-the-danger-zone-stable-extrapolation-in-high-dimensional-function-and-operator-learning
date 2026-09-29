"""
@converted from MATLAB code https://github.com/scaomath/fourier_neural_operator/blob/master/data_generation/darcy/demo.m
"""

import numpy as np
from scipy.fft import idctn
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.normal(0.0, 1.0, size=(s, s))

    # Define the (square root of) eigenvalues of the covariance operator
    K1, K2 = np.meshgrid(np.arange(s), np.arange(s), indexing="xy")

    coef = tau ** (alpha - 1) * (np.pi**2 * (K1**2 + K2**2) + tau**2) ** (-alpha / 2)

    # Construct the KL coefficients
    L = s * coef * xi
    L[0, 0] = 0.0

    U = idctn(L, type=2, norm='ortho')

    return U


def solve_gwf(coef, F, _=None):
    coef = np.asarray(coef, dtype=float)
    F = np.asarray(F, dtype=float)

    K = len(coef)

    x1 = np.arange(1, 2 * K, 2) / (2 * K)
    y1 = np.arange(1, 2 * K, 2) / (2 * K)
    x2 = np.linspace(0.0, 1.0, K)
    y2 = np.linspace(0.0, 1.0, K)

    coef_spline = RectBivariateSpline(y1, x1, coef, kx=3, ky=3)
    F_spline = RectBivariateSpline(y1, x1, F, kx=3, ky=3)

    coef = coef_spline(y2, x2)
    F = F_spline(y2, x2)

    F = F[1:K-1, 1:K-1]

    n = K - 2
    N = n * n
    A = lil_matrix((N, N))

    def idx(i, j):
        return i + j * n

    for j in range(1, K - 1):
        for i in range(1, K - 1):
            row = idx(i - 1, j - 1)

            west = (coef[i - 1, j] + coef[i, j]) / 2.0
            east = (coef[i + 1, j] + coef[i, j]) / 2.0
            south = (coef[i, j - 1] + coef[i, j]) / 2.0
            north = (coef[i, j + 1] + coef[i, j]) / 2.0

            A[row, row] = west + east + south + north

            if i > 1:
                A[row, idx(i - 2, j - 1)] = -west
            if i < K - 2:
                A[row, idx(i, j - 1)] = -east
            if j > 1:
                A[row, idx(i - 1, j - 2)] = -south
            if j < K - 2:
                A[row, idx(i - 1, j)] = -north

    A = A.tocsr() * (K - 1) ** 2

    rhs = F.reshape(-1, order='F')
    sol = spsolve(A, rhs)

    p_int = sol.reshape((n, n), order='F')

    P = np.block([
        [np.zeros((1, K))],
        [np.hstack([np.zeros((n, 1)), p_int, np.zeros((n, 1))])],
        [np.zeros((1, K))]
    ])

    P_spline = RectBivariateSpline(y2, x2, P, kx=3, ky=3)
    P = P_spline(y1, x1).T

    return P