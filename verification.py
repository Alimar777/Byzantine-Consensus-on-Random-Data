"""Grid search and diagnostic plots for MW weight dynamics.

This script simulates MW updates under Byzantine contamination, sweeps
eta/tau, and generates plots comparing empirical vs theoretical convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# -----------------------
# 1. Global Settings
# -----------------------
N_SAMPLES = 20
N_FEATURES = 1
BYZANTINE_FRACTION = 0.3
COLLUDING = True
MAX_ITERS = 50
SEED = 42
CONFIDENCE_LEVEL = 0.99
REQUIRED_BYZANTINE_SUCCESS = 0.8
BYZANTINE_DISTANCE = 5.0  # Fixed for this sweep

NORMALIZE_NONE = True
NORMALIZE_UNIT_INTERVAL = False
NORMALIZE_SUM_TO_ONE = False

np.random.seed(SEED)

# -----------------------
# 2. Functions
# -----------------------
def generate_data(byzantine_distance):
    """Generate honest and Byzantine sample means with optional collusion."""
    n_byz = int(N_SAMPLES * BYZANTINE_FRACTION)
    n_honest = N_SAMPLES - n_byz

    mu_true = np.random.randn(N_FEATURES) * 2

    honest_means = mu_true + 0.5 * np.random.randn(n_honest, N_FEATURES)

    if COLLUDING:
        byzantine_means = mu_true + byzantine_distance + 0.5 * np.random.randn(n_byz, N_FEATURES)
    else:
        byzantine_means = byzantine_distance * (np.random.randn(n_byz, N_FEATURES))

    sample_means = np.vstack([honest_means, byzantine_means])
    labels = np.array([0] * n_honest + [1] * n_byz)

    perm = np.random.permutation(N_SAMPLES)
    return sample_means[perm], labels[perm]

def robust_estimate(points):
    """Return a robust location estimate (coordinate-wise median)."""
    return np.median(points, axis=0)

def normalize_to_unit_interval(weights, epsilon=1e-8):
    """Rescale weights to [0, 1] for stabilization or visualization."""
    min_w = np.min(weights)
    max_w = np.max(weights)
    return (weights - min_w) / (max_w - min_w + epsilon)

def normalize_sum_to_one(weights, epsilon=1e-8):
    """Normalize weights to sum to one."""
    total_w = np.sum(weights)
    return weights / (total_w + epsilon)

def run_experiment(eta, tau, byzantine_distance):
    """Run a single MW experiment and compute success statistics.

    Returns honest/byzantine threshold rates plus empirical/theoretical T.
    """
    sample_means, labels = generate_data(byzantine_distance)
    weights = np.ones(N_SAMPLES)
    weight_history = np.zeros((N_SAMPLES, MAX_ITERS + 1))
    weight_history[:, 0] = weights

    for t in range(1, MAX_ITERS + 1):
        queries = sample_means + np.random.randn(N_SAMPLES, N_FEATURES) * 0.1
        mu_hat = robust_estimate(queries)
        c = 1.0
        residuals = np.linalg.norm(queries - mu_hat, axis=1)
        pseudo_losses = np.minimum(1, residuals / c)
        weights *= (1 - eta * pseudo_losses)

        if NORMALIZE_UNIT_INTERVAL:
            weights = normalize_to_unit_interval(weights)
        elif NORMALIZE_SUM_TO_ONE:
            weights = normalize_sum_to_one(weights)

        weight_history[:, t] = weights

    final_weights = weight_history[:, -1]
    below_threshold = final_weights < tau

    honest_below = np.sum((labels == 0) & below_threshold)
    byzantine_below = np.sum((labels == 1) & below_threshold)

    p_honest = honest_below / max(1, np.sum(labels == 0))
    p_byzantine = byzantine_below / max(1, np.sum(labels == 1))

    avg_honest = np.mean(weight_history[labels == 0], axis=0)
    avg_byzantine = np.mean(weight_history[labels == 1], axis=0)
    empirical_T = np.argmax(avg_byzantine < tau)
    theoretical_T = int(np.ceil((1 / eta) * np.log(1 / tau)))

    return p_honest, p_byzantine, empirical_T, theoretical_T, weight_history, labels

# -----------------------
# 3. Finer Grid Search
# -----------------------
eta_values = np.linspace(0.01, 0.1, 10)
tau_values = np.linspace(0.01, 0.1, 10)

results = []
all_histories = {}

for eta in eta_values:
    for tau in tau_values:
        p_honest, p_byzantine, empirical_T, theoretical_T, weight_history, labels = run_experiment(eta, tau, BYZANTINE_DISTANCE)
        n_honest = int(N_SAMPLES * (1 - BYZANTINE_FRACTION))
        delta = 1 - CONFIDENCE_LEVEL
        threshold_prob = delta / n_honest
        success_honest = (p_honest <= threshold_prob)
        success_byzantine = (p_byzantine >= REQUIRED_BYZANTINE_SUCCESS)
        overall_success = success_honest and success_byzantine
        results.append((eta, tau, BYZANTINE_DISTANCE, p_honest, p_byzantine, overall_success, empirical_T, theoretical_T))
        all_histories[(eta, tau)] = (weight_history, labels)

# -----------------------
# 4. Results
# -----------------------
results_df = pd.DataFrame(results, columns=["ETA", "TAU", "DIST", "P_HONEST", "P_BYZANTINE", "PASS_CONFIDENCE", "EMPIRICAL_T", "THEORETICAL_T"])

results_df = results_df.sort_values(by=["PASS_CONFIDENCE", "P_HONEST"], ascending=[False, True])

output_dir = "mw_weight_plots"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "finer_grid_search_results.csv")
results_df.to_csv(output_csv, index=False)

print("\nFiner Grid Search Results:")
print(results_df)
print(f"\nResults saved to {output_csv}")

best_params = results_df.iloc[0]
print("\nRecommended Parameters:")
print(f"ETA={best_params['ETA']:.4f}, TAU={best_params['TAU']:.4f}, DIST={best_params['DIST']}")

# -----------------------
# 5. Plot Heatmaps
# -----------------------
pivot_honest = results_df.pivot(index="ETA", columns="TAU", values="P_HONEST")
pivot_byzantine = results_df.pivot(index="ETA", columns="TAU", values="P_BYZANTINE")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

c1 = ax[0].imshow(pivot_honest.values, origin='lower', aspect='auto', extent=[tau_values.min(), tau_values.max(), eta_values.min(), eta_values.max()])
ax[0].set_title("P_HONEST Heatmap")
ax[0].set_xlabel("Tau")
ax[0].set_ylabel("Eta")
fig.colorbar(c1, ax=ax[0])

c2 = ax[1].imshow(pivot_byzantine.values, origin='lower', aspect='auto', extent=[tau_values.min(), tau_values.max(), eta_values.min(), eta_values.max()])
ax[1].set_title("P_BYZANTINE Heatmap")
ax[1].set_xlabel("Tau")
ax[1].set_ylabel("Eta")
fig.colorbar(c2, ax=ax[1])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmaps.png"))
plt.show()

# -----------------------
# 6. Plot Best Weight Curves
# -----------------------
best_eta = best_params['ETA']
best_tau = best_params['TAU']
weight_history, labels = all_histories[(best_eta, best_tau)]

plt.figure(figsize=(12, 8))

for i in range(N_SAMPLES):
    if labels[i] == 0:
        plt.plot(range(MAX_ITERS + 1), weight_history[i], color='blue', alpha=0.3)
    else:
        plt.plot(range(MAX_ITERS + 1), weight_history[i], color='red', alpha=0.5, linestyle='--')

avg_honest = np.mean(weight_history[labels == 0], axis=0)
avg_byzantine = np.mean(weight_history[labels == 1], axis=0)

plt.plot(range(MAX_ITERS + 1), avg_honest, color='blue', linewidth=3, label='Avg Honest')
plt.plot(range(MAX_ITERS + 1), avg_byzantine, color='red', linewidth=3, linestyle='--', label='Avg Byzantine')

plt.axhline(best_tau, color='black', linestyle='--',linewidth=3, label=f'Threshold τ={best_tau:.3f}')
plt.axvline(best_params['EMPIRICAL_T'], color='purple', linestyle=':',linewidth=3, label=f'Empirical T={int(best_params["EMPIRICAL_T"])})')
plt.axvline(best_params['THEORETICAL_T'], color='orange', linestyle=':',linewidth=3, label=f'Theoretical T={int(best_params["THEORETICAL_T"])})')

plt.xlabel("Round")
plt.ylabel("Weight")
plt.title("Best Weight Curves: Honest vs Byzantine")
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.savefig(os.path.join(output_dir, "best_weight_curves.png"))
plt.show()



# (The previous code remains unchanged up to the end of Plot Best Weight Curves)

# -----------------------
# 7. Plot Empirical vs Theoretical T Scatter
# -----------------------

import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(8, 6))

# Mask to filter points
mask_success = (results_df['EMPIRICAL_T'] > 0) | (results_df['THEORETICAL_T'] <= 60)

# Define color map: green (good) -> red (bad)
scores = []
for idx, row in results_df.loc[mask_success].iterrows():
    if row['PASS_CONFIDENCE']:
        scores.append(1.0)
    else:
        penalty = row['P_HONEST'] + (1 - row['P_BYZANTINE'])
        penalty = min(penalty, 1.0)
        scores.append(1.0 - penalty)

colors = plt.cm.RdYlGn(scores)

scatter = ax.scatter(
    results_df.loc[mask_success, 'THEORETICAL_T'],
    results_df.loc[mask_success, 'EMPIRICAL_T'],
    c=colors,
    alpha=0.8,
    edgecolors='black',
    linewidth=0.5
)

ax.plot([0, MAX_ITERS], [0, MAX_ITERS], linestyle='--', color='black', label='Ideal Line')

# Custom colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Verification Score (Green = Best)')

ax.set_xlabel("Theoretical T")
ax.set_ylabel("Empirical T")
ax.set_title("Empirical vs Theoretical Convergence Time (Color = Confidence)")
ax.grid(True)

# Annotate best point
best_eta = results_df.iloc[0]['ETA']
best_tau = results_df.iloc[0]['TAU']
best_theoretical_T = results_df.iloc[0]['THEORETICAL_T']
best_empirical_T = results_df.iloc[0]['EMPIRICAL_T']
ax.annotate("Best", (best_theoretical_T, best_empirical_T), textcoords="offset points", xytext=(10,5), ha='center', fontsize=9, color='blue', weight='bold')

plt.savefig(os.path.join(output_dir, "empirical_vs_theoretical_T.png"))
plt.show()

# -----------------------
# 8. Plot Error Histogram
# -----------------------

plt.figure(figsize=(8, 6))

error = results_df['EMPIRICAL_T'] - results_df['THEORETICAL_T']
error = error[mask_success]

plt.hist(error, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Empirical - Theoretical T Error")
plt.xlabel("Empirical T - Theoretical T")
plt.ylabel("Frequency")
plt.grid(True)

plt.savefig(os.path.join(output_dir, "error_histogram.png"))
plt.show()



# (The previous code remains unchanged up to the end of Plot Error Histogram)

# -----------------------
# 9. Plot Convergence Curves and Best Fit (Corrected for MW Formula)
# -----------------------

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Define the MW convergence model
def convergence_model(X, a, b):
    """MW theoretical convergence model used for curve fitting."""
    eta, tau = X
    return a * (1/eta) * np.log(1/tau) + b

# Prepare data: exclude only the worst strategies (where P_BYZANTINE == 0)
mask_good = (results_df['P_BYZANTINE'] > 0)
eta_data = results_df.loc[mask_success & mask_good, 'ETA'].values
tau_data = results_df.loc[mask_success & mask_good, 'TAU'].values
y_data = results_df.loc[mask_success & mask_good, 'EMPIRICAL_T'].values

# Perform curve fitting
params, covariance = curve_fit(convergence_model, (eta_data, tau_data), y_data)
a_fit, b_fit = params

# Predict using fitted model
y_pred = convergence_model((eta_data, tau_data), a_fit, b_fit)

# Calculate R^2 score
r2 = r2_score(y_data, y_pred)

# Generate fit line for plotting
x_theoretical = (1/eta_data) * np.log(1/tau_data)
x_fit = np.linspace(min(x_theoretical), max(x_theoretical), 100)
y_fit = a_fit * x_fit + b_fit

plt.figure(figsize=(8, 6))

plt.scatter(x_theoretical, y_data, c='green', alpha=0.8, edgecolors='black', linewidth=0.5, label='Empirical Data (Good Strategies)')
plt.plot(x_fit, y_fit, color='blue', linestyle='-', linewidth=3, label=f'Best Fit: y = {a_fit:.2f}x + {b_fit:.2f}')
plt.plot([0, MAX_ITERS], [0, MAX_ITERS], linestyle='--', color='black', linewidth=3, label='Ideal Line')

plt.xlabel(r"Theoretical $T = \frac{1}{\eta} \log\left( \frac{1}{\tau} \right)$")
plt.ylabel(r"Empirical $T$")
plt.title(r"Multiplicative Weights Convergence: Empirical vs. Theoretical $T$")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, "convergence_best_fit.png"))
plt.show()

# Print best fit parameters
print("\nBest Fit Parameters (using MW convergence model, Excluding Full Failures):")
print(f"Slope (a) = {a_fit:.4f}")
print(f"Intercept (b) = {b_fit:.4f}")
print(f"R^2 Score = {r2:.4f}")

# -----------------------
# 10. Additional Verification Plots
# -----------------------

# Total weight mass over time
if 'all_histories' in globals():
    best_eta = results_df.iloc[0]['ETA']
    best_tau = results_df.iloc[0]['TAU']
    weight_history, labels = all_histories[(best_eta, best_tau)]

    total_weight = np.sum(weight_history, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(total_weight)), total_weight, color='purple', linewidth=2)
    plt.title("Total Weight Mass Over Time")
    plt.xlabel("Round")
    plt.ylabel("Sum of Weights")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "total_weight_mass.png"))
    plt.show()

    # Gap between honest and byzantine mean weights
    mean_honest = np.mean(weight_history[labels == 0], axis=0)
    mean_byzantine = np.mean(weight_history[labels == 1], axis=0)
    gap = mean_honest - mean_byzantine

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(gap)), gap, color='darkgreen', linewidth=2)
    plt.axhline(0, linestyle='--', color='black')
    plt.title("Mean Honest - Mean Byzantine Weight Gap Over Time")
    plt.xlabel("Round")
    plt.ylabel("Gap (Honest - Byzantine)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "honest_byzantine_gap.png"))
    plt.show()

