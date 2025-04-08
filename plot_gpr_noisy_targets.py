"""
=========================================================
Gaussian Processes regression: basic introductory example
=========================================================

A simple one-dimensional regression example computed in two different ways:

1. A noise-free case
2. A noisy case with known noise-level per datapoint

In both cases, the kernel's parameters are estimated using the maximum
likelihood principle.

The figures illustrate the interpolating property of the Gaussian Process model
as well as its probabilistic nature in the form of a pointwise 95% confidence
interval.

Note that `alpha` is a parameter to control the strength of the Tikhonov
regularization on the assumed training points' covariance matrix.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Dataset generation
# ------------------
# We start by generating a synthetic dataset. The true generative process
# is defined as a continuous piecewise linear function.

X = np.linspace(start=0, stop=50, num=1_000).reshape(-1, 1)

def generate_parameters():
    """Generate 6 random parameters with specific ranges for each."""
    return [
        random.uniform(15, 25),     # y0
        random.uniform(5, 15),      # x1
        random.uniform(35, 45),     # x2
        random.uniform(3.5, 4.5),   # m1
        random.uniform(-0.1, 0.1),  # m2
        random.uniform(-3.5, -2.5), # m3
    ]

def piecewise_function(x, y0, x1, x2, slope1, slope2, slope3):
    y = np.zeros_like(x)
    
    # First segment: positive slope, intersects the origin
    y[x < x1] = y0 + slope1 * x[x < x1]
    
    # Second segment: slightly negative slope, continuous at x1
    intercept2 = y0 + slope1 * x1
    y[(x >= x1) & (x < x2)] = slope2 * (x[(x >= x1) & (x < x2)] - x1) + intercept2
    
    # Third segment: steep negative slope, continuous at x2
    intercept3 = slope2 * (x2 - x1) + intercept2
    y[x >= x2] = slope3 * (x[x >= x2] - x2) + intercept3
    
    return y

# Generate target values
params = generate_parameters()
y = piecewise_function(X, *params)
y = np.squeeze(y)

# Plot the true function
plt.plot(X, y, label=r"$f(x)$ piecewise", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Continuous Piecewise Function with Adjustable Slopes and Transition Points")
plt.show()

# Example with noise-free target
# ------------------------------
# Use the true function without added noise. Select a few samples for training.
rng = np.random.RandomState(1)
num_samples = 30
segment_size = y.size // num_samples

training_indices = np.array([
    rng.choice(np.arange(i * segment_size, (i + 1) * segment_size))
    for i in range(num_samples)
])
X_train, y_train = X[training_indices], y[training_indices]

# Fit a Gaussian Process using an RBF kernel
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)

# Predict the mean and std over the whole dataset
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# Plot predictions and uncertainty
plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression on noise-free dataset")
plt.show()

# Example with noisy targets
# --------------------------
# Add random Gaussian noise to the training targets.
noise_std = 0.75
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

# Fit a Gaussian Process, accounting for the noise variance with alpha
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# Plot predictions and uncertainty with noisy data
plt.plot(X, y, label=r"$f(x)$", linestyle="dotted")
noise_std_array = np.full_like(y_train_noisy, noise_std)
plt.errorbar(
    X_train,
    y_train_noisy,
    yerr=noise_std_array,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Gaussian process regression on a noisy dataset")
plt.show()

# The noise affects predictions near the training samples.
# The uncertainty increases even for regions close to observed data,
# reflecting the modeled observation noise.
