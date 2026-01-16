"""Verification sweeps for Byzantine-robust MW with plots and heatmaps.

Runs multi-trial simulations, tracks blacklisting behavior, and produces
summary CSVs/plots for parameter sweeps of k/eta/tau.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from MW import run_sim, sample_means, labels, n_features, output_dir

NUM_TRIALS = 10
MAX_ITER = 50
THRESHOLD = 1e-4
K_VALUE = 0.9
LEARNING_RATE = 1.5
EPSILON_TARGET = 0.01
DELTA_CONFIDENCE = 1e-3

MU_TRUE = np.mean(sample_means[labels == 0], axis=0)

honest_loss_records = []
byzantine_blacklist_times = []
mu_deviation_records = []
runtime_records = []
trial_summaries = []

weight_decay_honest = []
weight_decay_byzantine = []


def reset_records():
    """Reset all global metric accumulators between sweeps."""
    global honest_loss_records, byzantine_blacklist_times, mu_deviation_records, runtime_records, trial_summaries
    global weight_decay_honest, weight_decay_byzantine
    honest_loss_records = []
    byzantine_blacklist_times = []
    mu_deviation_records = []
    runtime_records = []
    trial_summaries = []
    weight_decay_honest = []
    weight_decay_byzantine = []


def clean_verification_folder():
    """Remove prior sweep subfolders and plots to keep outputs clean."""
    verif_path = f"{output_dir}/verification"
    if os.path.exists(verif_path):
        for sub in os.listdir(verif_path):
            sub_path = os.path.join(verif_path, sub)
            if os.path.isdir(sub_path) and sub.startswith("sweep"):
                shutil.rmtree(sub_path)
        plots_path = os.path.join(verif_path, "plots")
        if os.path.exists(plots_path):
            shutil.rmtree(plots_path)


def verify_bounds(custom_output_dir=None, k_val=None, eta_val=None, tau_val=None):
    """Run repeated trials and record blacklist and loss statistics."""
    for trial in range(NUM_TRIALS):
        results = run_sim(labels, sample_means, max_mw_iters=MAX_ITER, kill_byz=1, plot=0)

        honest_mask = (labels == 0)
        byz_mask = (labels == 1)

        weights = np.ones(len(sample_means))
        weight_hist = []

        honest_blacklisted = False
        byzantine_survived = False

        for t in range(MAX_ITER):
            pseudo_losses = np.array([np.linalg.norm(sample_means[idx] - results['straight_means'][t]) for idx in range(len(sample_means))])
            pseudo_losses = np.clip(pseudo_losses / (np.median(pseudo_losses) * K_VALUE + 1e-8), 0, 1)
            weights *= np.exp(-LEARNING_RATE * pseudo_losses)
            weights /= np.sum(weights)
            weight_hist.append(weights.copy())

            if np.any(weights[honest_mask] < THRESHOLD):
                honest_blacklisted = True
            if np.all(weights[byz_mask] > THRESHOLD):
                byzantine_survived = True

        weight_hist = np.array(weight_hist)
        weight_decay_honest.append(weight_hist[:, honest_mask])
        weight_decay_byzantine.append(weight_hist[:, byz_mask])

        for idx in np.where(honest_mask)[0]:
            honest_loss = np.array([np.linalg.norm(sample_means[idx] - mu_t) for mu_t in results['straight_means']])
            honest_loss_records.append(np.mean(honest_loss))

        byz_blacklist_rounds = []
        for idx in np.where(byz_mask)[0]:
            for t in range(MAX_ITER):
                if weight_hist[t, idx] < THRESHOLD:
                    byz_blacklist_rounds.append(t)
                    break
            else:
                byz_blacklist_rounds.append(MAX_ITER)

        byzantine_blacklist_times.extend(byz_blacklist_rounds)

        for mu_t in results['straight_means']:
            mu_deviation_records.append(np.linalg.norm(mu_t - MU_TRUE))

        trial_summary = {
            'trial': trial+1,
            'avg_honest_loss': np.mean(honest_loss_records[-len(np.where(honest_mask)[0]):]),
            'max_honest_loss': np.max(honest_loss_records[-len(np.where(honest_mask)[0]):]),
            'avg_blacklist_round': np.mean(byz_blacklist_rounds),
            'runtime_sec': 0,
            'honest_blacklisted': honest_blacklisted,
            'byzantine_survived': byzantine_survived,
            'k': k_val,
            'eta': eta_val,
            'tau': tau_val
        }

        trial_summaries.append(trial_summary)

    if custom_output_dir:
        os.makedirs(custom_output_dir, exist_ok=True)
        pd.DataFrame(trial_summaries).to_csv(f"{custom_output_dir}/verification_results.csv", index=False)


def run_single_sweep(params):
    """Helper for multiprocessing: run a single (k, eta, tau) sweep."""
    k, eta, tau, sweep_dir = params
    global K_VALUE, LEARNING_RATE, THRESHOLD
    K_VALUE = k
    LEARNING_RATE = eta
    THRESHOLD = tau
    reset_records()
    verify_bounds(custom_output_dir=sweep_dir, k_val=k, eta_val=eta, tau_val=tau)
    return pd.DataFrame(trial_summaries)

def sweep_parameters():
    """Run full grid sweeps over k, eta, and threshold values."""
    k_values = [0.1, 0.3, 0.5, 0.75, 1.0, 1.25]
    eta_values = [0.5, 1.0, 1.5, 2.0]
    threshold_values = [1e-2, 1e-3, 1e-4]

    clean_verification_folder()
    all_sweep_data = []

    tasks = []
    for k in k_values:
        for eta in eta_values:
            for tau in threshold_values:
                sweep_dir = f"{output_dir}/verification/sweep_k_{k}_eta_{eta}_tau_{tau}"
                tasks.append((k, eta, tau, sweep_dir))

    def run_single_sweep(params):
        k, eta, tau, sweep_dir = params
        global K_VALUE, LEARNING_RATE, THRESHOLD
        K_VALUE = k
        LEARNING_RATE = eta
        THRESHOLD = tau
        reset_records()
        verify_bounds(custom_output_dir=sweep_dir, k_val=k, eta_val=eta, tau_val=tau)
        return pd.DataFrame(trial_summaries)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_sweep, tasks))

    if results:
        master_df = pd.concat(results, ignore_index=True)
        master_df.to_csv(f"{output_dir}/verification/master_results.csv", index=False)
        plot_sweep_summary(master_df)
        plot_max_deviation_vs_threshold(master_df)
        plot_heatmaps(master_df)


def plot_sweep_summary(df):
    """Plot average honest loss and blacklist time across sweeps."""
    save_path = f"{output_dir}/verification/plots"
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10,7))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    for idx, eta in enumerate(sorted(df['eta'].unique())):
        subset = df[df['eta'] == eta]
        plt.plot(subset['k'], subset['avg_honest_loss'], marker=markers[idx%len(markers)], label=f'eta={eta}')
    plt.xlabel('k')
    plt.ylabel('Avg Honest Loss')
    plt.title('Avg Honest Loss vs k for different eta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/summary_honest_loss_vs_k.png")
    plt.savefig(f"{save_path}/summary_honest_loss_vs_k.pdf")

    plt.figure(figsize=(10,7))
    for idx, k in enumerate(sorted(df['k'].unique())):
        subset = df[df['k'] == k]
        plt.plot(subset['eta'], subset['avg_blacklist_round'], marker=markers[idx%len(markers)], label=f'k={k}')
    plt.xlabel('eta')
    plt.ylabel('Avg Blacklist Round')
    plt.title('Avg Blacklist Time vs eta for different k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/summary_blacklist_time_vs_eta.png")
    plt.savefig(f"{save_path}/summary_blacklist_time_vs_eta.pdf")


def plot_max_deviation_vs_threshold(df):
    """Plot max honest deviation vs tau for each k."""
    save_path = f"{output_dir}/verification/plots"
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10,7))
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    for idx, k in enumerate(sorted(df['k'].unique())):
        subset = df[df['k'] == k]
        thresholds = subset['tau'].unique()
        max_devs = [subset[subset['tau'] == t]['max_honest_loss'].max() for t in thresholds]
        plt.plot(thresholds, max_devs, marker=markers[idx%len(markers)], label=f'k={k}')
    plt.xscale('log')
    plt.xlabel('Threshold (tau)')
    plt.ylabel('Max Honest Loss')
    plt.title('Max Deviation vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/summary_max_deviation_vs_threshold.png")
    plt.savefig(f"{save_path}/summary_max_deviation_vs_threshold.pdf")


def plot_heatmaps(df):
    """Render heatmaps for blacklist time and honest loss."""
    save_path = f"{output_dir}/verification/plots"
    os.makedirs(save_path, exist_ok=True)

    pivot_blacklist = df.pivot_table(index='eta', columns='k', values='avg_blacklist_round')
    pivot_loss = df.pivot_table(index='eta', columns='k', values='avg_honest_loss')

    plt.figure(figsize=(8,6))
    plt.imshow(pivot_blacklist, cmap='viridis', aspect='auto')
    plt.colorbar(label='Avg Blacklist Round')
    plt.xticks(ticks=np.arange(len(pivot_blacklist.columns)), labels=pivot_blacklist.columns)
    plt.yticks(ticks=np.arange(len(pivot_blacklist.index)), labels=pivot_blacklist.index)
    plt.xlabel('k')
    plt.ylabel('eta')
    plt.title('Heatmap of Avg Blacklist Time')
    plt.tight_layout()
    plt.savefig(f"{save_path}/heatmap_blacklist_time.png")
    plt.savefig(f"{save_path}/heatmap_blacklist_time.pdf")

    plt.figure(figsize=(8,6))
    plt.imshow(pivot_loss, cmap='magma', aspect='auto')
    plt.colorbar(label='Avg Honest Loss')
    plt.xticks(ticks=np.arange(len(pivot_loss.columns)), labels=pivot_loss.columns)
    plt.yticks(ticks=np.arange(len(pivot_loss.index)), labels=pivot_loss.index)
    plt.xlabel('k')
    plt.ylabel('eta')
    plt.title('Heatmap of Avg Honest Loss')
    plt.tight_layout()
    plt.savefig(f"{save_path}/heatmap_honest_loss.png")
    plt.savefig(f"{save_path}/heatmap_honest_loss.pdf")


if __name__ == "__main__":
    RUN_SWEEP = True

    if RUN_SWEEP:
        sweep_parameters()
    else:
        reset_records()
        verify_bounds()
