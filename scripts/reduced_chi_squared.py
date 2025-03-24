import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate test data
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_data = x_data + np.random.normal(scale=1, size=len(x_data))  # Linear trend with noise
errors = np.full_like(y_data, 1.0)  # Assume constant uncertainty

# Define model functions
def linear_model(x, a, b):
    """
        a = T0
        b = P
        x = E
    """
    return a + b * x

def quadratic_model(x, a, b, c):
    """
        a = T0
        b = P
        c = dPdE
        x = E
    """
    return a + b * x + 0.5 * c * x**2

def sinusoidal_model(x, a, b, c, d, e):
    """
        a = T0
        b = P
        c = e
        d = dwdE
        e = w0
        x = E
    """
    return a + (b * x) - ((c * (b / (1 - ((1 / (2 * np.pi)) * d)))) / np.pi) * np.cos(e + (d * x))

# Fit models
params_linear, _ = curve_fit(linear_model, x_data, y_data)
params_quad, _ = curve_fit(quadratic_model, x_data, y_data)
params_sin, _ = curve_fit(sinusoidal_model, x_data, y_data)

# Compute residuals and chi-squared
def reduced_chi_squared(obs, fit, errors):
    return np.sum(((obs - fit) / errors) ** 2)

def bic(k, N, chi2):
    return chi2 + (k*np.log(N))

y_linear_fit = linear_model(x_data, *params_linear)
y_quad_fit = quadratic_model(x_data, *params_quad)
y_sin_fit = sinusoidal_model(x_data, *params_sin)

chi2_linear = reduced_chi_squared(y_data, y_linear_fit, errors)
chi2_quad = reduced_chi_squared(y_data, y_quad_fit, errors)
chi2_sin = reduced_chi_squared(y_data, y_sin_fit, errors)

bic_linear = bic(2, len(x_data), chi2_linear)
bic_quad = bic(3, len(x_data), chi2_quad)
bic_sin = bic(5, len(x_data), chi2_sin)

# Create smooth x values for plotting
x_smooth = np.linspace(0, 10, 100)
y_linear = linear_model(x_smooth, *params_linear)
y_quad = quadratic_model(x_smooth, *params_quad)
y_sin = sinusoidal_model(x_smooth, *params_sin)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear fit
axes[0].scatter(x_data, y_data, color='black', label='Data')
axes[0].plot(x_smooth, y_linear, color='red', label='Linear Fit')
for i in range(len(x_data)):
    axes[0].plot([x_data[i], x_data[i]], [y_data[i], y_linear_fit[i]], 'r--', alpha=0.6)
axes[0].set_title(f"Linear Fit (2-Parameter Model)\nReduced χ² = {chi2_linear:.2f}\nBIC = {bic_linear:.2f}")
axes[0].legend()

# Quadratic fit
axes[1].scatter(x_data, y_data, color='black', label='Data')
axes[1].plot(x_smooth, y_quad, color='blue', label='Quadratic Fit')
for i in range(len(x_data)):
    axes[1].plot([x_data[i], x_data[i]], [y_data[i], y_quad_fit[i]], 'b--', alpha=0.6)
axes[1].set_title(f"Quadratic Fit (3-Parameter Model)\nReduced χ² = {chi2_quad:.2f}\nBIC = {bic_quad:.2f}")
axes[1].legend()

# Sinusoidal fit
axes[2].scatter(x_data, y_data, color='black', label='Data')
axes[2].plot(x_smooth, y_sin, color='green', label='Sinusoidal Fit')
for i in range(len(x_data)):
    axes[2].plot([x_data[i], x_data[i]], [y_data[i], y_sin_fit[i]], 'g--', alpha=0.6)
axes[2].set_title(f"Sinusoidal Fit (5-Parameter Model)\nReduced χ² = {chi2_sin:.2f}\nBIC = {bic_sin:.2f}")
axes[2].legend()

plt.tight_layout()
plt.show()
