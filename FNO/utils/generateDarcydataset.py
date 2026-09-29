#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:14:16 2026

@author: sherril
"""

import numpy as np
from darcy_utils import GRF, solve_gwf


def generate_one_sample(s, alpha, tau, forcing_type="ones"):
    # Step 1: sample Gaussian random field
    norm_a = GRF(alpha, tau, s)

    # Step 2: exponentiate to make permeability positive
    a = np.exp(norm_a)

    # Step 3: choose forcing
    if forcing_type == "ones":
        f = np.ones((s, s))
    else:
        raise ValueError(f"Unknown forcing_type: {forcing_type}")

    # Step 4: solve PDE
    u = solve_gwf(a, f)

    return a, u

def generate_dataset(n_samples, s, alpha, tau, forcing_type="ones"):
    X_list = []
    Y_list = []

    for i in range(n_samples):
        a, u = generate_one_sample(s, alpha, tau, forcing_type=forcing_type)

        X_list.append(a)
        Y_list.append(u)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"Generated {i + 1}/{n_samples} samples")

    X = np.stack(X_list, axis=0)   # shape: (N, s, s)
    Y = np.stack(Y_list, axis=0)   # shape: (N, s, s)

    return X, Y

# Generate training data
X_train, Y_train = generate_dataset(
    n_samples=1000,
    s=64,
    alpha=2,
    tau=3,
    forcing_type="ones"
)

# Generate test data
X_test, Y_test = generate_dataset(
    n_samples=200,
    s=64,
    alpha=2,
    tau=3,
    forcing_type="ones"
)

print(X_train.shape)  # (1000, 64, 64)
print(Y_train.shape)  # (1000, 64, 64)
