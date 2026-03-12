import numpy as np
from scipy.optimize import minimize

# Data from your measurements
data = [
    {'P_size': 10e-6, 'V': 0.01407, 'fwhm_obs': 0.698e-3},  # 10 µm
    {'P_size': 4e-6,  'V': 0.02567, 'fwhm_obs': 0.45e-3},   # 4 µm
    {'P_size': 2e-6,  'V': 0.02504, 'fwhm_obs': 0.384e-3},  # 2 µm
]
S_l = 70e-6  # Spot size from beam profiler
theta = 82   # Incident angle in degrees
sin_theta = np.sin(np.radians(theta))

# Function to compute FWHM given x, y, and parameters
def compute_fwhm(P_size, V, S_l, x, y):
    tau = (P_size * x + S_l * y) / (V * sin_theta)
    return 2.355 * tau

# Cost function for optimization
def cost_function(params):
    x, y = params
    error = 0
    for d in data:
        fwhm_pred = compute_fwhm(d['P_size'], d['V'], S_l, x, y)
        error += (fwhm_pred - d['fwhm_obs'])**2
    return error

# Initial guess for x and y
initial_guess = [0.2, 0.05]

# Run optimization
result = minimize(cost_function, initial_guess, method='Nelder-Mead', bounds=[(0, 1), (0, 1)])
x_opt, y_opt = result.x

# Output results
print(f"Optimized x: {x_opt:.4f}, Optimized y: {y_opt:.4f}")
print(f"Optimization success: {result.success}, Error: {result.fun:.2e}")

# Verify fit
for d in data:
    fwhm_pred = compute_fwhm(d['P_size'], d['V'], S_l, x_opt, y_opt)
    print(f"P_size: {d['P_size']*1e6:.0f} µm, Predicted FWHM: {fwhm_pred*1e3:.3f} ms, Observed: {d['fwhm_obs']*1e3:.3f} ms")