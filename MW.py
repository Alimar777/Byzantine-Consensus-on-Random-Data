"""Multiplicative-weights experiments for Byzantine-robust mean estimation.

This script simulates queryable samples with honest/byzantine labels, runs
several estimators (mean, MW-weighted mean, MoM-guided MW, median-of-means),
and produces error/trajectory plots for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_global(straight_means,  mw_means, mom_mw_means, median_of_means, median_of_means_means,
                straight_mean_errors, mw_errors, mom_mw_errors, median_of_means_errors, labels, sample_means, max_mw_iters=10):
    """Render global error curves and estimator trajectories for one run.

    This is a convenience wrapper that saves the standard plots used in the
    experiments: error evolution vs iteration and 2D estimator trajectories.
    """

    plt.figure(figsize=(8,6))
    plt.plot(range(1, max_mw_iters+1), straight_mean_errors, label='Straight Mean', marker='o')
    plt.plot(range(1, max_mw_iters+1), mw_errors, label='Original MW (to MW Mean)', marker='s')
    plt.plot(range(1, max_mw_iters+1), mom_mw_errors, label='MoM-Guided MW (to MoM)', marker='d')
    plt.plot(range(1, max_mw_iters+1), median_of_means_errors, label='Median of Means', marker='^')
    plt.xlabel('Iteration')
    plt.ylabel('Distance to True Mean')
    plt.title('Estimator Error Evolution (including MoM-guided MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
    #plt.show()

    # Trajectory Plot with MoM-guided MW
    plt.figure(figsize=(8,8))
    plt.scatter(sample_means[:,0], sample_means[:,1], c=colors[labels], alpha=0.2, label='Sample Means')
    plt.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')

    # Plot each estimator's trajectory
    plt.plot(straight_means[:,0], straight_means[:,1], marker='o', label='Straight Mean Trajectory')
    plt.plot(mw_means[:,0], mw_means[:,1], marker='s', label='Original MW Trajectory')
    plt.plot(mom_mw_means[:,0], mom_mw_means[:,1], marker='d', label='MoM-guided MW Trajectory')
    plt.plot(median_of_means_means[:,0], median_of_means_means[:,1], marker='^', label='Median of Means Trajectory')

    # Start points
    plt.scatter(straight_means[0,0], straight_means[0,1], c='black', marker='o', s=100, label='Straight Start')
    plt.scatter(mw_means[0,0], mw_means[0,1], c='black', marker='s', s=100, label='MW Start')
    plt.scatter(mom_mw_means[0,0], mom_mw_means[0,1], c='black', marker='d', s=100, label='MoM-MW Start')
    plt.scatter(median_of_means_means[0,0], median_of_means_means[0,1], c='black', marker='^', s=100, label='MoM Start')

    plt.title('Estimator Trajectories toward True Mean (with MoM-guided MW)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/trajectories_mom_guided.png")
    #plt.show()

    #print(f"\nPlots saved in directory: {os.path.abspath(output_dir)}")
   


def query_sample(idx, labels, sample_means, n_features, scale_good=0.1, scale_bad=3.0):
    """Sample a noisy query from a labeled mean.

    Honest samples use a small noise scale; Byzantine samples use a larger
    noise scale to simulate adversarial variance.
    """
    if labels[idx] == 0:
        return sample_means[idx] + np.random.randn(n_features) * scale_good
    else:
        return sample_means[idx] + np.random.randn(n_features) * scale_bad
def softmin(x_values, mean, c_value):
    """Compute a softmin weight distribution based on distance to mean."""
    '''residuals = np.exp(-np.sum((x_values - mean)**2, axis=1)/c_value)
    print(f"residuals {residuals/np.sum(residuals)}")
    return residuals/np.sum(residuals)'''
    loss = np.exp(-np.linalg.norm(x_values - mean, axis=1)/c_value)
    return loss/np.sum(loss)
def softmin_loss(x, mean, beta=2.0, c=0.5):
    """Softmin-style loss with tunable sharpness (beta) and scale (c)."""
    r = np.linalg.norm(x - mean, axis=1)
    return np.exp(- (r / c) ** beta)

def logistic_loss(x, mean, scale=1.0):
    """Logistic loss on distances to the current mean estimate."""
    r = np.linalg.norm(x - mean, axis=1)
    return np.log(1 + np.exp(r / scale))


def huber_loss(x_values, mean, delta=1):
    """Huber loss on distances to reduce outlier impact."""
    r = np.linalg.norm(x_values - mean, axis=1)
    return np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))


def min_loss(x_values, mean, c_value):
    """Clipped linear loss based on normalized residual size."""
    residuals = np.linalg.norm(x_values - mean, axis=1)
    normalized_losses= residuals/ (np.max(residuals) + 1e-8)
    return np.array([min(1, loss/c_value) for loss in normalized_losses])

def run_sim(new_labels, sample_means, kill_byz=0, use_softmin=0, softmin_var=1,
            huber=0, log=0, max_mw_iters=10, c_value=0.5, k=0.9, epsilon=0.1,
            threshold=0.00001, plot=1):
    """Run a MW-based robust mean estimation simulation.

    Each iteration samples noisy points around per-agent means, computes
    multiple estimators, updates MW weights using a chosen loss, optionally
    blacklists low-weight agents, and records error trajectories.
    """
    # --- Multiplicative Weights Protocol over Queryable Samples ---

    weights_mw = np.ones(len(sample_means))          # Original MW (residual to weighted mean)
    weights_mom_mw = np.ones(len(sample_means))       # New MW (residual to Median of Means)

    mw_errors = []
    mom_mw_errors = []
    straight_mean_errors = []
    median_of_means_errors = []

    mw_means = []
    mom_mw_means = []
    straight_means = []
    median_of_means_means = []
   
    mw_errors = []
    mom_mw_errors = []
    straight_mean_errors = []
    median_of_means_errors = []

    byz_name = "out Blacklisting"
    loss_name = "Min Loss"
    for iter in range(max_mw_iters):
        queried_points = np.array([query_sample(i, labels, sample_means,n_features) for i in range(len(sample_means))])

        # Straight Mean
        straight_mean = np.mean(queried_points, axis=0)
        straight_median = np.median(queried_points, axis=0)

        # MW Weighted Mean (self-centered)
        weighted_mean = np.average(queried_points, axis=0, weights=weights_mw)

        #Hi MoM
        bucket_size = max(5, len(queried_points)//50)
        buckets = np.array_split(queried_points, bucket_size)
        bucket_means = np.array([np.mean(bucket, axis=0) for bucket in buckets if len(bucket) > 0])
        median_of_means = np.median(bucket_means, axis=0)

        # MoM-guided MW Weighted Mean
        weighted_mean_mom = np.average(queried_points, axis=0, weights=weights_mom_mw)

        # --- Record Errors ---
        straight_mean_errors.append(np.linalg.norm(straight_mean - mu_true))
       
        mad_variance = np.median(np.abs((queried_points - straight_median)))
        c_value = mad_variance * k
        #print(f"c value {c_value}")
       
        mw_errors.append(np.linalg.norm(weighted_mean - mu_true))
        mom_mw_errors.append(np.linalg.norm(weighted_mean_mom - mu_true))
        median_of_means_errors.append(np.linalg.norm(median_of_means - mu_true))

        # Record Trajectories
        straight_means.append(straight_mean)
        mw_means.append(weighted_mean)
        mom_mw_means.append(weighted_mean_mom)
        median_of_means_means.append(median_of_means)

        # --- MW Weight Updates ---

        # (1) Original MW: residuals to weighted_mean
        if use_softmin:
            loss_mw = softmin(queried_points, weighted_mean, c_value)
            loss_name = "Softmin Loss"
        if softmin_var:
            loss_mw = softmin_loss(queried_points, weighted_mean, c_value)
            loss_name = "Softmin Variant Loss"
        if huber:
            loss_mw =huber_loss(queried_points, weighted_mean,)
            loss_name = "Huber Loss"
        if log:
            loss_mw = logistic_loss(queried_points, weighted_mean,)
            loss_name = "Logistic Loss"
        else:
            loss_mw = min_loss(queried_points, weighted_mean, c_value)
       
        weights_mw *= np.exp(-learning_rate_mw * loss_mw)
       
        if kill_byz: #if we want to black list
            byz_name = " Blacklisting"
            bad_guys_index = [ind for ind, w in enumerate(weights_mw) if w < threshold]
            queried_points = np.delete(queried_points, bad_guys_index, axis=0)
            weights_mw = np.delete(weights_mw, bad_guys_index)
            sample_means = np.delete(sample_means, bad_guys_index, axis=0)
            new_labels = np.delete(new_labels, bad_guys_index)
           
            weights_mom_mw = np.delete(weights_mom_mw, bad_guys_index,)
           
        weights_mw /= np.sum(weights_mw)
       
        # (2) MoM-guided MW: residuals to median_of_means
       
        if use_softmin:
            loss_mom = softmin(queried_points, median_of_means, c_value)
        if softmin_var:
            loss_mom = softmin_loss(queried_points, median_of_means, c_value)
        if huber:
            loss_mom =huber_loss(queried_points, median_of_means,)
        if log:
            loss_mom= logistic_loss(queried_points, median_of_means,)
        else:
            loss_mom = min_loss(queried_points, median_of_means, c_value)
       
        # Dynamic Learning Rate for MoM-guided MW based on bucket spread
       
        bucket_variances = np.var(bucket_means, axis=0)
        error_mom_estimate = np.linalg.norm(bucket_variances)
        dynamic_lr_mom_mw = min(learning_rate_mom_mw, 0.5 + 10.0 * error_mom_estimate)
        weights_mom_mw *= np.exp(-dynamic_lr_mom_mw * loss_mom)
        weights_mom_mw /= np.sum(weights_mom_mw)

        #print(f"Iter {iter+1}: Straight = {straight_mean_errors[-1]:.4f} | MW = {mw_errors[-1]:.4f} | MoM-MW = {mom_mw_errors[-1]:.4f} | MoM = {median_of_means_errors[-1]:.4f}")
       
    straight_means = np.array(straight_means)  
    mw_means = np.array(mw_means)
    mom_mw_means = np.array(mom_mw_means)
    median_of_means_means = np.array(median_of_means_means)
    results ={'straight_means': straight_means,
              'mw_means': mw_means,
              'mom_mw_means': mom_mw_means,
              'median_of_means_means': median_of_means_means,
              'straight_mean_errors':straight_mean_errors,
              'mw_errors':mw_errors,
              'mom_mw_errors':mom_mw_errors,
              'median_of_mean_errors': median_of_means_errors,  
              'k':k,
              'threshold': threshold,
              'byz': kill_byz,
              'log': log,
              'iters': max_mw_iters,
               'use_softmin': use_softmin,
               'softmin_var': softmin_var,
              'huber': huber,
              'labels': new_labels,
              'sample_means':sample_means,
                   
    }
    if plot:
        plt.figure(figsize=(8,6))
        plt.plot(range(1, max_mw_iters+1), straight_mean_errors, label='Straight Mean', marker='o')
        plt.plot(range(1, max_mw_iters+1), mw_errors, label='Original MW (to MW Mean)', marker='s')
        plt.plot(range(1, max_mw_iters+1), mom_mw_errors, label='MoM-Guided MW (to MoM)', marker='d')
        plt.plot(range(1, max_mw_iters+1), median_of_means_errors, label='Median of Means', marker='^')
        plt.xlabel('Iteration')
        plt.ylabel('Distance to True Mean')
        plt.title(f'Estimator Error Evolution with{byz_name} using {loss_name}, k={k}, T={threshold}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
        #plt.show()

        #Trajectory Plot with MoM-guided MW
        plt.figure(figsize=(8,8))
        plt.scatter(sample_means[:,0], sample_means[:,1], c=colors[new_labels], alpha=0.2, label='Sample Means')
        plt.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')

        #Plot each estimator's trajectory
        plt.plot(straight_means[:,0], straight_means[:,1], marker='o', label='Straight Mean Trajectory')
        plt.plot(mw_means[:,0], mw_means[:,1], marker='s', label='Original MW Trajectory')
        plt.plot(mom_mw_means[:,0], mom_mw_means[:,1], marker='d', label='MoM-guided MW Trajectory')
        plt.plot(median_of_means_means[:,0], median_of_means_means[:,1], marker='^', label='Median of Means Trajectory')

        # Start points
        plt.scatter(straight_means[0,0], straight_means[0,1], c='black', marker='o', s=100, label='Straight Start')
        plt.scatter(mw_means[0,0], mw_means[0,1], c='black', marker='s', s=100, label='MW Start')
        plt.scatter(mom_mw_means[0,0], mom_mw_means[0,1], c='black', marker='d', s=100, label='MoM-MW Start')
        plt.scatter(median_of_means_means[0,0], median_of_means_means[0,1], c='black', marker='^', s=100, label='MoM Start')

        plt.title(f'Trajectories toward True Mean with{byz_name} using {loss_name}, k={k}, T={threshold},E = {max_mw_iters}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/trajectories_mom_guided.png")
        #plt.show()

        #print(f"\nPlots saved in directory: {os.path.abspath(output_dir)}")
    return results

       
#def graphing(loss_func, testing_parameter):

def makeAvg(results_min, results_sv, results_s, results_l, results_h, name):
        """Average a named metric across multiple loss-function runs."""
        mw_vectors = [
        results_min[name],
        results_sv[name],
        results_s[name],
        results_l[name],
        results_h[name],
        ]
        mean_vector = np.mean(np.stack(mw_vectors), axis=0)
        return mean_vector
def twobytwo_trajectory_visuals(byz_name, results_min, results_sv, results_s, results_l, results_h,
                                sample_means, mu_true, colors, k_i, t, i, output_dir):
    """Create a 2x2 grid of estimator trajectories across loss types."""

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()  # flatten 2D array for easier iteration

    method_results = [results_min, results_sv, results_s, results_h]
    method_names = ['Min Loss', 'Softmin Var', 'Softmin', 'Huber']

    for idx, (results, title) in enumerate(zip(method_results, method_names)):
        ax = axs[idx]
        ax.scatter(results['sample_means'][:, 0], results['sample_means'][:, 1], c=colors[results['labels']], alpha=0.2, label='Sample Means')
        ax.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')
       

        ax.plot(results['straight_means'][:,0], results['straight_means'][:,1], marker='o', label='Straight Mean')
        ax.plot(results['mw_means'][:,0], results['mw_means'][:,1], marker='s', label='MW')
        ax.plot(results['mom_mw_means'][:,0], results['mom_mw_means'][:,1], marker='d', label='MoM MW')
        ax.plot(results['median_of_means_means'][:,0], results['median_of_means_means'][:,1], marker='^', label='Median of Means')


        ax.scatter(results['straight_means'][0,0], results['straight_means'][0,1], c='black', marker='o', s=100, label='Start: Straight')
        ax.scatter(results['mw_means'][0,0], results['mw_means'][0,1], c='black', marker='s', s=100, label='Start: MW')
        ax.scatter(results['mom_mw_means'][0,0], results['mom_mw_means'][0,1], c='black', marker='d', s=100, label='Start: MoM MW')
        ax.scatter(results['median_of_means_means'][0,0], results['median_of_means_means'][0,1], c='black', marker='^', s=100, label='Start: MoM')

        ax.set_title(f'{title} Trajectories', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)

    if len(method_results) < 4:
        axs[-1].axis('off')

    plt.suptitle(f"Trajectories Toward True Mean of Individual Loss Functions with{byz_name} | k={k_i}, T={t}, E={i}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/trajectories_fourup_with{byz_name}.png")
   

def twobytwo(byz_name, results_min, results_sv, results_s, results_l, results_h, k_i, t):
    """Create a 2x2 grid of error curves across loss types."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
    axs[0, 0].plot(results_min['straight_mean_errors'], label = "Min Loss")
    axs[0, 0].plot(results_sv['straight_mean_errors'], label = "Softmin Variant")
    axs[0, 0].plot(results_s['straight_mean_errors'], label = "Softmin")
    #axs[0, 0].plot(results_l['straight_mean_errors'], label = "Log")
    axs[0, 0].plot(results_h['straight_mean_errors'], label = "Huber")
    axs[0, 0].set_title(f'Straight Mean')
   
    axs[0, 1].plot(results_min['mw_errors'], label = "Min Loss")
    axs[0, 1].plot(results_sv['mw_errors'], label = "Softmin Variant")
    axs[0, 1].plot(results_s['mw_errors'], label = "Softmin")
    #axs[0, 1].plot(results_l['mw_errors'], label = "Log")
    axs[0, 1].plot(results_h['mw_errors'], label = "Huber")
    axs[0, 1].set_title('Original MW (to MW Mean)')
   
    axs[1, 0].plot(results_min['mom_mw_errors'], label = "Min Loss")
    axs[1, 0].plot(results_sv['mom_mw_errors'], label = "Softmin Variant")
    axs[1, 0].plot(results_s['mom_mw_errors'], label = "Softmin")
    #axs[1, 0].plot(results_l['mom_mw_errors'], label = "Log")
    axs[1, 0].plot(results_h['mom_mw_errors'], label = "Huber")
    axs[1, 0].set_title('MoM-Guided MW (to MoM)')
   
    axs[1, 1].plot(results_min['median_of_mean_errors'], label = "Min Loss")
    axs[1, 1].plot(results_sv['median_of_mean_errors'], label = "Softmin Variant")
    axs[1, 1].plot(results_s['median_of_mean_errors'], label = "Softmin")
    #axs[1, 1].plot(results_l['median_of_mean_errors'], label = "Log")
    axs[1, 1].plot(results_h['median_of_mean_errors'], label = "Huber")
    axs[1, 1].set_title('Median of Means')
    plt.suptitle(f"Error Evolution of Individual Loss Functions with{byz_name} k={k_i}, T={t} ", fontsize=16)
    axs[0, 0].legend()
    axs[1, 1].legend()
    axs[1, 0].legend()
    axs[0, 1].legend()
    plt.tight_layout()


def twobytwo_trajectory_visuals_mod(byz_name, results_min, results_sv, results_s, results_l, results_h,
                                sample_means, mu_true, colors, name, x_axis, output_dir):
    """Trajectory grid with an external x-axis parameter label."""

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()  # flatten 2D array for easier iteration

    method_results = [results_min, results_sv, results_s, results_h]
    method_names = ['Min Loss', 'Softmin Var', 'Softmin', 'Huber']

    for idx, (results, title) in enumerate(zip(method_results, method_names)):
        ax = axs[idx]
        ax.scatter(results['sample_means'][:, 0], results['sample_means'][:, 1], c=colors[results['labels']], alpha=0.2, label='Sample Means')
        ax.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')
       
        ax.plot(results['straight_means'][:,0], results['straight_means'][:,1], marker='o', label='Straight Mean')
        ax.plot(results['mw_means'][:,0], results['mw_means'][:,1], marker='s', label='MW')
        ax.plot(results['mom_mw_means'][:,0], results['mom_mw_means'][:,1], marker='d', label='MoM MW')
        ax.plot(results['median_of_means_means'][:,0], results['median_of_means_means'][:,1], marker='^', label='Median of Means')


        ax.scatter(results['straight_means'][0,0], results['straight_means'][0,1], c='black', marker='o', s=100, label='Start: Straight')
        ax.scatter(results['mw_means'][0,0], results['mw_means'][0,1], c='black', marker='s', s=100, label='Start: MW')
        ax.scatter(results['mom_mw_means'][0,0], results['mom_mw_means'][0,1], c='black', marker='d', s=100, label='Start: MoM MW')
        ax.scatter(results['median_of_means_means'][0,0], results['median_of_means_means'][0,1], c='black', marker='^', s=100, label='Start: MoM')

        ax.set_title(f'{title} Trajectories', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)

    if len(method_results) < 4:
        axs[-1].axis('off')

    plt.suptitle(f"Trajectories Toward True Mean of Individual Loss Functions with{byz_name} and {name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/trajectories_fourup_with{byz_name}.png")
   

def twobytwo_mod(byz_name, results_min, results_sv, results_s, results_l, results_h, name, x_axis):
    """Error grid with a custom x-axis (e.g., k or threshold sweeps)."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
    axs[0, 0].plot(x_axis, results_min['straight_mean_errors'], label = "Min Loss")
    axs[0, 0].plot(x_axis,results_sv['straight_mean_errors'], label = "Softmin Variant")
    axs[0, 0].plot(x_axis,results_s['straight_mean_errors'], label = "Softmin")
    #axs[0, 0].plot(x_axis,results_l['straight_mean_errors'], label = "Log")
    axs[0, 0].plot(x_axis,results_h['straight_mean_errors'], label = "Huber")
    axs[0, 0].set_xlabel(name)
    axs[0, 0].set_ylabel('Distance to True Mean')
    axs[0, 0].set_title(f'Straight Mean')
   
    axs[0, 1].plot(x_axis,results_min['mw_errors'], label = "Min Loss")
    axs[0, 1].plot(x_axis,results_sv['mw_errors'], label = "Softmin Variant")
    axs[0, 1].plot(x_axis,results_s['mw_errors'], label = "Softmin")
    #axs[0, 1].plot(x_axis,results_l['mw_errors'], label = "Log")
    axs[0, 1].plot(x_axis,results_h['mw_errors'], label = "Huber")
    axs[0, 1].set_xlabel(name)
    axs[0, 1].set_ylabel('Distance to True Mean')
    axs[0, 1].set_title('Original MW (to MW Mean)')
   
    axs[1, 0].plot(x_axis,results_min['mom_mw_errors'], label = "Min Loss")
    axs[1, 0].plot(x_axis,results_sv['mom_mw_errors'], label = "Softmin Variant")
    axs[1, 0].plot(x_axis,results_s['mom_mw_errors'], label = "Softmin")
    #axs[1, 0].plot(x_axis,results_l['mom_mw_errors'], label = "Log")
    axs[1, 0].plot(x_axis,results_h['mom_mw_errors'], label = "Huber")
    axs[1, 0].set_xlabel(name)
    axs[1, 0].set_ylabel('Distance to True Mean')
    axs[1, 0].set_title('MoM-Guided MW (to MoM)')
   
    axs[1, 1].plot(x_axis,results_min['median_of_mean_errors'], label = "Min Loss")
    axs[1, 1].plot(x_axis,results_sv['median_of_mean_errors'], label = "Softmin Variant")
    axs[1, 1].plot(x_axis,results_s['median_of_mean_errors'], label = "Softmin")
    #axs[1, 1].plot(x_axis,results_l['median_of_mean_errors'], label = "Log")
    axs[1, 1].plot(x_axis,results_h['median_of_mean_errors'], label = "Huber")
    axs[1, 1].set_title('Median of Means')
    axs[1, 1].set_xlabel(name)
    axs[1, 1].set_ylabel('Distance to True Mean')
    plt.suptitle(f"Error Evolution of Individual Loss Functions with{byz_name} and {name} ", fontsize=16)
    axs[0, 0].legend()
    axs[1, 1].legend()
    axs[1, 0].legend()
    axs[0, 1].legend()
    plt.tight_layout()

np.random.seed(42)

# Parameters
n_samples = 3000
n_features = 2
delta = 1/3
bias_contamination = True
learning_rate_mw = 1.5   # Learning rate for original MW
learning_rate_mom_mw = 1.5  # Higher learning rate for MoM-guided MW
n_queries_per_sample = 5  # Number of times each sample can be queried

n_good = int((1 - delta) * n_samples)
n_bad = n_samples - n_good
good_variance = 0.5
bad_variance = 0.5
mu_true = np.random.randn(n_features)

# Honest samples are small Gaussian around true mean
honest_means = np.random.randn(n_good, n_features) * good_variance+ mu_true

# Byzantine samples are adversarial
if bias_contamination:
    bias_direction = np.array([5.0, 0.0])
    byzantine_means = np.random.randn(n_bad, n_features) * bad_variance + mu_true + bias_direction
else:
    byzantine_means = 50 * (np.random.randn(n_bad, n_features))

# Combined
sample_means = np.vstack([honest_means, byzantine_means])
labels = np.array([0] * n_good + [1] * n_bad)

perm = np.random.permutation(len(sample_means))
sample_means = sample_means[perm]
labels = labels[perm]
colors = np.array(['green', 'red'])  # 0 = honest, 1 = byzantine
output_dir = "filtering_plots"
os.makedirs(output_dir, exist_ok=True)


def blacklist_test(labels, i = 10, t = 0.00001, k_i = 0.9):
    for b in [0,1]:
        if b == 1:
            results_min= run_sim(labels,
                sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_sv= run_sim(labels,
                    sample_means, use_softmin = 0, softmin_var = 1, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                    c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_s= run_sim(labels,
                    sample_means, use_softmin = 1, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                    c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_l= run_sim(labels,
                    sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 1,kill_byz = b, max_mw_iters=i,
                    c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_h= run_sim(labels,
                    sample_means, use_softmin = 0, softmin_var = 0, huber = 1, log = 0, kill_byz = b, max_mw_iters=i,
                    c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
                
            straight_mean_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'straight_mean_errors')
            mw_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mw_errors')
            mom_mw_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mom_mw_errors')
            median_of_means_errors = makeAvg(results_min, results_s, results_s, results_l, results_h,'median_of_mean_errors')
        
            '''straight_means = makeAvg(results_min_b, results_sv_b, results_s_b, results_l_b, results_h_b,'straight_means')
            mw_means = makeAvg(results_min_b, results_sv_b, results_s_b, results_l_b, results_h_b,'mw_means')
            mom_mw_means = makeAvg(results_min_b, results_sv_b, results_s_b, results_l_b, results_h_b,'mom_mw_means')
            median_of_means_means = makeAvg(results_min_b, results_sv_b, results_s_b, results_l, results_h_b,'median_of_means_means')'''
            ####################################################-----Blacklist---------####################################################
            twobytwo( " Blacklisting", results_min, results_sv, results_s, results_l, results_h, k_i, t )
            twobytwo_trajectory_visuals(" Blacklisting", results_min, results_sv, results_s, results_l, results_h,
                                            sample_means, mu_true, colors, k_i, t, i, output_dir)



            plt.figure(figsize=(8,6))
            plt.plot(range(1, i+1), straight_mean_errors, label='Straight Mean Avg', marker='o')
            plt.plot(range(1, i+1), mw_errors, label='Original MW (to MW Mean) Avg', marker='s')
            plt.plot(range(1, i+1), mom_mw_errors, label='MoM-Guided MW (to MoM) Avg', marker='d')
            plt.plot(range(1, i+1), median_of_means_errors, label='Median of Means Avg', marker='^')
            plt.xlabel('Iteration')
            plt.ylabel('Distance to True Mean')
            plt.title(f'Error Evolution with Blacklisting using Loss Function Avgs k={k_i}, T={t}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
            #plt.show()              
        
        if b == 0:

            results_min= run_sim(labels,
                sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_sv= run_sim(labels,
                sample_means, use_softmin = 0, softmin_var = 1, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_s= run_sim(labels,
                sample_means, use_softmin = 1, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_l= run_sim(labels,
                sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 1,kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
            results_h= run_sim(labels,
                sample_means, use_softmin = 0, softmin_var = 0, huber = 1, log = 0, kill_byz = b, max_mw_iters=i,
                c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
    
            straight_mean_errors_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'straight_mean_errors')
            mw_errors_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mw_errors')
            mom_mw_errors_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mom_mw_errors')
            median_of_means_errors_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'median_of_mean_errors')
        
            straight_means_n =  makeAvg(results_min, results_sv, results_s, results_l, results_h,'straight_means')
            mw_means_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mw_means')
            mom_mw_means_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mom_mw_means')
            median_of_means_means_n = makeAvg(results_min, results_sv, results_s, results_l, results_h,'median_of_means_means')
    
            ####################################################-----No Blacklist---------####################################################
            twobytwo( "out Blacklisting", results_min, results_sv, results_s, results_l, results_h, k_i, t  )
            twobytwo_trajectory_visuals("out Blacklisting", results_min, results_sv, results_s, results_l, results_h,
                                            sample_means, mu_true, colors, k_i, t, i, output_dir)
        

            plt.figure(figsize=(8,6))
            plt.plot(range(1, i+1), straight_mean_errors_n, label='Straight Mean Avg', marker='o')
            plt.plot(range(1, i+1), mw_errors_n, label='Original MW (to MW Mean) Avg', marker='s')
            plt.plot(range(1, i+1), mom_mw_errors_n, label='MoM-Guided MW (to MoM) Avg', marker='d')
            plt.plot(range(1, i+1), median_of_means_errors_n, label='Median of Means Avg', marker='^')
            plt.xlabel('Iteration')
            plt.ylabel('Distance to True Mean')
            plt.title(f'Error Evolution without Blacklisting using Loss Function Avgs k={k_i}, T={t}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")


            #------------------------------------------------------------------------------------------------------------------
            #direction graph
            plt.figure(figsize=(8,8))
            plt.scatter(sample_means[:,0], sample_means[:,1], c=colors[labels], alpha=0.2, label='Sample Means')
            plt.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')


            plt.plot(straight_means_n[:,0], straight_means_n[:,1], marker='o', label='Straight Mean Trajectory Avg')
            plt.plot(mw_means_n[:,0], mw_means_n[:,1], marker='s', label='Original MW Trajectory Avg')
            plt.plot(mom_mw_means_n[:,0], mom_mw_means_n[:,1], marker='d', label='MoM-guided MW Trajectory')
            plt.plot(median_of_means_means_n[:,0], median_of_means_means_n[:,1], marker='^', label='Median of Means Trajectory')


            plt.scatter(straight_means_n[0,0], straight_means_n[0,1], c='black', marker='o', s=100, label='Straight Start')
            plt.scatter(mw_means_n[0,0], mw_means_n[0,1], c='black', marker='s', s=100, label='MW Start')
            plt.scatter(mom_mw_means_n[0,0], mom_mw_means_n[0,1], c='black', marker='d', s=100, label='MoM-MW Start')
            plt.scatter(median_of_means_means_n[0,0], median_of_means_means_n[0,1], c='black', marker='^', s=100, label='MoM Start')

            plt.title(f'Trajectories toward True Mean without Blacklisting using Loss Function Avgs,  k={k_i}, T={t},E={i}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/trajectories_mom_guided.png")
            #plt.show()          

def add_values(name, results_min1, results_sv1, results_s1, results_l1, results_h1,
                   results_min, results_sv, results_s, results_l, results_h):
        """Append the latest value for a metric across loss-function runs."""
        results_h1[name].append(results_h[name][-1])
        results_l1[name].append(results_l[name][-1])
        results_min1[name].append(results_min[name][-1])
        results_s1[name].append(results_s[name][-1])
        results_sv1[name].append(results_sv[name][-1])
        return results_min1, results_sv1, results_s1, results_l1, results_h1
   
def parameter_test(labels, results_dict_entry, results_min1, results_sv1, results_s1, results_l1,
                   results_h1, i=10, t=0.00001, k_i=0.9, b=1, graph=0):
    """Sweep a single parameter setting and aggregate metrics across losses."""
   
    results_min= run_sim(labels,
        sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
        c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
    results_sv= run_sim(labels,
            sample_means, use_softmin = 0, softmin_var = 1, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
            c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
    results_s= run_sim(labels,
            sample_means, use_softmin = 1, softmin_var = 0, huber = 0, log = 0,kill_byz = b, max_mw_iters=i,
            c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
    results_l= run_sim(labels,
            sample_means, use_softmin = 0, softmin_var = 0, huber = 0, log = 1,kill_byz = b, max_mw_iters=i,
            c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
    results_h= run_sim(labels,
            sample_means, use_softmin = 0, softmin_var = 0, huber = 1, log = 0, kill_byz = b, max_mw_iters=i,
            c_value = 0.5, k = k_i, epsilon = 0.1, threshold = t, plot = 0)
           
    straight_mean_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'straight_mean_errors')
    mw_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mw_errors')
    mom_mw_errors = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mom_mw_errors')
    median_of_means_errors = makeAvg(results_min, results_s, results_s, results_l, results_h,'median_of_mean_errors')
   
    straight_means = makeAvg(results_min, results_sv, results_s, results_l, results_h,'straight_means')
    mw_means = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mw_means')
    mom_mw_means = makeAvg(results_min, results_sv, results_s, results_l, results_h,'mom_mw_means')
    median_of_means_means = makeAvg(results_min, results_sv, results_s, results_l, results_h,'median_of_means_means')
   
    if b == 1 and graph == 1:
        ####################################################-----Blacklist---------####################################################
        twobytwo( " Blacklisting", results_min, results_sv, results_s, results_l, results_h, k_i, t )
        twobytwo_trajectory_visuals(" Blacklisting", results_min, results_sv, results_s, results_l, results_h,
                                        sample_means, mu_true, colors, k_i, t, i, output_dir)

        plt.figure(figsize=(8,6))
        plt.plot(range(1, i+1), straight_mean_errors, label='Straight Mean Avg', marker='o')
        plt.plot(range(1, i+1), mw_errors, label='Original MW (to MW Mean) Avg', marker='s')
        plt.plot(range(1, i+1), mom_mw_errors, label='MoM-Guided MW (to MoM) Avg', marker='d')
        plt.plot(range(1, i+1), median_of_means_errors, label='Median of Means Avg', marker='^')
        plt.xlabel('Iteration')
        plt.ylabel('Distance to True Mean')
        plt.title(f'Error Evolution with Blacklisting using Loss Function Avgs k={k_i}, T={t}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
        #plt.show()              
       
    if b == 0 and graph == 1:    
        ####################################################-----No Blacklist---------####################################################
        twobytwo( "out Blacklisting", results_min, results_sv, results_s, results_l, results_h, k_i, t  )
        twobytwo_trajectory_visuals("out Blacklisting", results_min, results_sv, results_s, results_l, results_h,
                                        sample_means, mu_true, colors, k_i, t, i, output_dir)


        plt.figure(figsize=(8,6))
        plt.plot(range(1, i+1), straight_mean_errors, label='Straight Mean Avg', marker='o')
        plt.plot(range(1, i+1), mw_errors, label='Original MW (to MW Mean) Avg', marker='s')
        plt.plot(range(1, i+1), mom_mw_errors, label='MoM-Guided MW (to MoM) Avg', marker='d')
        plt.plot(range(1, i+1), median_of_means_errors, label='Median of Means Avg', marker='^')
        plt.xlabel('Iteration')
        plt.ylabel('Distance to True Mean')
        plt.title(f'Error Evolution without Blacklisting using Loss Function Avgs k={k_i}, T={t}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")


        #------------------------------------------------------------------------------------------------------------------
        #direction graph
        plt.figure(figsize=(8,8))
        plt.scatter(sample_means[:,0], sample_means[:,1], c=colors[labels], alpha=0.2, label='Sample Means')
        plt.scatter(mu_true[0], mu_true[1], c='green', marker='x', s=150, label='True Mean')


        plt.plot(straight_means[:,0], straight_means[:,1], marker='o', label='Straight Mean Trajectory Avg')
        plt.plot(mw_means[:,0], mw_means[:,1], marker='s', label='Original MW Trajectory Avg')
        plt.plot(mom_mw_means[:,0], mom_mw_means[:,1], marker='d', label='MoM-guided MW Trajectory')
        plt.plot(median_of_means_means[:,0], median_of_means_means[:,1], marker='^', label='Median of Means Trajectory')


        plt.scatter(straight_means[0,0], straight_means[0,1], c='black', marker='o', s=100, label='Straight Start')
        plt.scatter(mw_means[0,0], mw_means[0,1], c='black', marker='s', s=100, label='MW Start')
        plt.scatter(mom_mw_means[0,0], mom_mw_means[0,1], c='black', marker='d', s=100, label='MoM-MW Start')
        plt.scatter(median_of_means_means[0,0], median_of_means_means[0,1], c='black', marker='^', s=100, label='MoM Start')

        plt.title(f'Trajectories toward True Mean without Blacklisting using Loss Function Avgs,  k={k_i}, T={t},E={i}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/trajectories_mom_guided.png")
        #plt.show()
       
    results_dict_entry['straight_mean_errors'].append(straight_mean_errors[-1])
    results_dict_entry['mw_errors'].append(mw_errors[-1])
    results_dict_entry['mom_mw_errors'].append(mom_mw_errors[-1])
    results_dict_entry['median_of_mean_errors'].append(median_of_means_errors[-1])
   

   
    results_min1, results_sv1, results_s1, results_l1, results_h1 = add_values('straight_mean_errors', results_min1, results_sv1, results_s1,
                                                                               results_l1, results_h1, results_min, results_sv,
                                                                               results_s, results_l, results_h)
    results_min1, results_sv1, results_s1, results_l1, results_h1 = add_values('mw_errors', results_min1, results_sv1, results_s1,
                                                                               results_l1, results_h1, results_min, results_sv,
                                                                               results_s, results_l, results_h)
    results_min1, results_sv1, results_s1, results_l1, results_h1 = add_values('mom_mw_errors', results_min1, results_sv1,
                                                                               results_s1, results_l1, results_h1,results_min,
                                                                               results_sv, results_s, results_l, results_h)
    results_min1, results_sv1, results_s1, results_l1, results_h1 = add_values('median_of_mean_errors', results_min1, results_sv1,
                                                                               results_s1, results_l1, results_h1, results_min,
                                                                               results_sv, results_s, results_l, results_h)
   
    results_total = [results_min, results_sv, results_s, results_l, results_h]
    results_dict_entry['straight_means'].append(straight_means[-1,-1])  
    results_dict_entry['mw_means'].append(mw_means[-1, -1])  
    results_dict_entry['mom_mw_means'].append(mom_mw_means[-1, -1])  
    results_dict_entry['median_of_means_means'].append(median_of_means_means[-1, -1])
   
           
    return results_dict_entry, results_min1, results_sv1, results_s1, results_l1, results_h1, results_total

               
def makeCompGraph(results_min, results_sv, results_s, results_l, results_h, results_list, x_axis, x_name, k_i, t,
                  straight_mean_errors, mw_errors, mom_mw_errors, median_of_means_errors, byz_name):
    """Render comparison plots for a parameter sweep."""
    
    twobytwo_mod(byz_name, results_min, results_sv, results_s, results_l, results_h, x_name, x_axis )

    '''twobytwo_trajectory_visuals_mod(byz_name, results_list[0], results_list[1], results_list[2], results_list[3], results_list[4],
                                   sample_means, mu_true, colors, x_name, x_axis, output_dir)'''

    plt.figure(figsize=(8,6))
    plt.plot(x_axis, straight_mean_errors, label='Straight Mean Avg', marker='o')
    plt.plot(x_axis, mw_errors, label='Original MW (to MW Mean) Avg', marker='s')
    plt.plot(x_axis, mom_mw_errors, label='MoM-Guided MW (to MoM) Avg', marker='d')
    plt.plot(x_axis, median_of_means_errors, label='Median of Means Avg', marker='^')
    plt.xlabel(x_name)
    plt.ylabel('Distance to True Mean')
    plt.title(f'Error Evolution using Loss Function Avgs with{byz_name} and {x_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
    #plt.show()                    
             
       

results_final = []
i = 10
ks = [0.1, 0.3, 0.6, 0.9, 1, 2]
thresholds = [ 0.00005, 0.00001,  0.000005, 0.000001]
#blacklist_test( labels,  i = 10, t = 0.00001, k_i = 0.9, b = 1)



def init_results(sample_means):
    """Initialize the results dict for aggregating sweep metrics."""
    return {
        'straight_means': [],  
        'mw_means': [],
        'mom_mw_means': [],
        'median_of_means_means': [],
        'straight_mean_errors': [],
        'mw_errors': [],
        'mom_mw_errors': [],
        'median_of_mean_errors': [],  
        'k': [],
        'threshold': [],
        'byz': [],
        'log': [],
        'iters': [],
        'use_softmin': [],
        'softmin_var': [],
        'huber': [],
        'labels': [],
        'sample_means': sample_means,
    }
results_min1 = init_results(sample_means)
results_sv1 = init_results(sample_means)
results_s1 = init_results(sample_means)
results_l1 = init_results(sample_means)
results_h1 = init_results(sample_means)

results_min2 = init_results(sample_means)
results_sv2 = init_results(sample_means)
results_s2 = init_results(sample_means)
results_l2 = init_results(sample_means)
results_h2 = init_results(sample_means)

results_dict_b= init_results(sample_means)
results_dict = init_results(sample_means)
t = 0.00009
#uncomment for k value testing
'''for k in ks:
    results_dict, results_min1, results_sv1, results_s1, results_l1, results_h1, results_totals = parameter_test(labels, results_dict,results_min1,
                                                                                                  results_sv1, results_s1, results_l1, results_h1,
                                                                                            i = 10, t = 0.00001, k_i = k, b = 1)
    results_dict_b, results_min2, results_sv2, results_s2, results_l2, results_h2, results_totals2= parameter_test(labels, results_dict_b,
                                                                                                   results_min2, results_sv2, results_s2,
                                                                                                   results_l2, results_h2,
                                                                                                       i = 10, t = 0.00001, k_i = k, b = 0)
   
   

makeCompGraph(results_min1, results_sv1, results_s1, results_l1, results_h1, results_totals, ks, 'K values', k, t,
                  results_dict['straight_mean_errors'],results_dict['mw_errors'],
                  results_dict['mom_mw_errors'],results_dict['median_of_mean_errors'], " Blacklisting")
makeCompGraph( results_min2, results_sv2, results_s2, results_l2, results_h2,results_totals2, ks, 'K values',k, t,
                  results_dict_b['straight_mean_errors'], results_dict_b['mw_errors'],
                  results_dict_b['mom_mw_errors'],results_dict_b['median_of_mean_errors'], "out Blacklisting")'''


#uncomment for threshold testing
'''k = 1
for t in thresholds:
    results_dict, results_min1, results_sv1, results_s1, results_l1, results_h1, results_totals = parameter_test(labels, results_dict,results_min1,
                                                                                                  results_sv1, results_s1, results_l1, results_h1,
                                                                                            i = 10, t = t, k_i = k, b = 1)
    results_dict_b, results_min2, results_sv2, results_s2, results_l2, results_h2, results_totals2= parameter_test(labels, results_dict_b,
                                                                                                   results_min2, results_sv2, results_s2,
                                                                                                   results_l2, results_h2,
                                                                                                       i = 10, t = t, k_i = k, b = 0)
   
   

makeCompGraph(results_min1, results_sv1, results_s1, results_l1, results_h1, results_totals, thresholds, 'Threshold values', k, t,
                  results_dict['straight_mean_errors'],results_dict['mw_errors'],
                  results_dict['mom_mw_errors'],results_dict['median_of_mean_errors'], " Blacklisting")
makeCompGraph( results_min2, results_sv2, results_s2, results_l2, results_h2,results_totals2, thresholds, 'Threshold values',k, t,
                  results_dict_b['straight_mean_errors'], results_dict_b['mw_errors'],
                  results_dict_b['mom_mw_errors'],results_dict_b['median_of_mean_errors'], "out Blacklisting")'''


#################################-----------Single---------###################################3

blacklist_test(labels, i = 10, t = 0.00001, k_i = 0.9)
       
'''plt.figure(figsize=(8,6))
plt.plot(range(1, iters+1), straight_mean_errors, label='Straight Mean Avg Blacklist', marker='o', color = 'green')
plt.plot(range(1, iters+1), mw_errors, label='Original MW (to MW Mean) Avg Blacklist', marker='s', color = 'green')
plt.plot(range(1, iters+1), mom_mw_errors, label='MoM-Guided MW (to MoM) Avg Blacklist', marker='d', color = 'green')
plt.plot(range(1, iters+1), median_of_means_errors, label='Median of Means Avg Blacklist', marker='^', color = 'green')
plt.plot(range(1, iters+1), straight_mean_errors_n, label='Straight Mean Avg', marker='o',linestyle='--',color='blue',)
plt.plot(range(1, iters+1), mw_errors_n, label='Original MW (to MW Mean) Avg', marker='s', linestyle='--',color='blue',)
plt.plot(range(1, iters+1), mom_mw_errors_n, label='MoM-Guided MW (to MoM) Avg', marker='d', linestyle='--',color='blue',)
plt.plot(range(1, iters+1), median_of_means_errors_n, label='Median of Means Avg', marker='^', linestyle='--',color='blue',)
plt.xlabel('Iteration')
plt.ylabel('Distance to True Mean')
plt.title(f'Estimator Error Evolution with Blacklisting k={k_i}, T={t}')
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/error_evolution_mom_guided.png")
plt.show()'''



