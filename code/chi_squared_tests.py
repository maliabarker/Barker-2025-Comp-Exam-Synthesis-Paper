import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from susie import TimingData, Ephemeris

# Generate test data
np.random.seed(42)
x_data = np.linspace(0, 10, 20)
y_data = x_data + np.random.normal(0, 3, len(x_data))  # Linear trend with noise

# Define model functions
def linear_model(x, a, b):
    return a * x + b

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

def sinusoidal_model(x, a, b, c, d, e):
    return a * np.sin(b * x + c) + d * x + e

# Fit models
params_linear, _ = curve_fit(linear_model, x_data, y_data)
params_quad, _ = curve_fit(quadratic_model, x_data, y_data)
params_sin, _ = curve_fit(sinusoidal_model, x_data, y_data)

# Create smooth x values for plotting
x_smooth = np.linspace(0, 10, 100)
y_linear = linear_model(x_smooth, *params_linear)
y_quad = quadratic_model(x_smooth, *params_quad)
y_sin = sinusoidal_model(x_smooth, *params_sin)

# Compute residuals
y_linear_fit = linear_model(x_data, *params_linear)
y_quad_fit = quadratic_model(x_data, *params_quad)
y_sin_fit = sinusoidal_model(x_data, *params_sin)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Linear fit
axes[0].scatter(x_data, y_data, color='black', label='Data')
axes[0].plot(x_smooth, y_linear, color='red', label='Linear Fit')
for i in range(len(x_data)):
    axes[0].plot([x_data[i], x_data[i]], [y_data[i], y_linear_fit[i]], 'r--', alpha=0.6)
axes[0].set_title("Linear Fit (2-Parameter Model)")
axes[0].legend()

# Quadratic fit
axes[1].scatter(x_data, y_data, color='black', label='Data')
axes[1].plot(x_smooth, y_quad, color='blue', label='Quadratic Fit')
for i in range(len(x_data)):
    axes[1].plot([x_data[i], x_data[i]], [y_data[i], y_quad_fit[i]], 'b--', alpha=0.6)
axes[1].set_title("Quadratic Fit (3-Parameter Model)")
axes[1].legend()

# Sinusoidal fit
axes[2].scatter(x_data, y_data, color='black', label='Data')
axes[2].plot(x_smooth, y_sin, color='green', label='Sinusoidal Fit')
for i in range(len(x_data)):
    axes[2].plot([x_data[i], x_data[i]], [y_data[i], y_sin_fit[i]], 'g--', alpha=0.6)
axes[2].set_title("Sinusoidal Fit (5-Parameter Model)")
axes[2].legend()

plt.tight_layout()
plt.show()
