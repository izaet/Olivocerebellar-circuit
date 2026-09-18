from matplotlib.gridspec import GridSpec
import networkx as nx
from itertools import product
from collections import defaultdict
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import circmean
from scipy.stats import circstd
from pathlib import Path
import sys
import numpy as np
import glob
import os
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter




def weight_convergence(jobs):

    # Load baseline run datasets
    datapaths = [job["run_path"] for job in jobs]
    chunk_size = 3000.0  # ms

    all_rows = []

    for dataset in datapaths:
        label = Path(dataset).stem
        with np.load(dataset, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}

        ts = data['ts']
        
        weights = data['pfpc_weights']

        weights_p_chunk = weights[::int(chunk_size)]
        ts_chunks = ts[::int(chunk_size)]
        n_chunks = weights_p_chunk.shape[0]
        if n_chunks < 2:
            continue

        init_weights = weights_p_chunk[0]
        delta_weights_abs = np.diff(weights_p_chunk, axis=0)
        delta_weights_rel = delta_weights_abs / init_weights

        for epoch_idx in range(delta_weights_abs.shape[0]):
            for weight_idx in range(delta_weights_abs.shape[1]):
                all_rows.append({
                    'dataset': label,
                    'epoch': epoch_idx + 1,
                    'weight_index': weight_idx,
                    'abs_delta': delta_weights_abs[epoch_idx, weight_idx],
                    'rel_delta': delta_weights_rel[epoch_idx, weight_idx],
                    'time_ms': ts_chunks[epoch_idx + 1],
                })

    all_df = pd.DataFrame(all_rows)

    if all_df.empty:
        print("No epoch-level weight-change rows found; adjust chunk_size or dataset lengths.")
    else:
        if 'dataset' not in all_df.columns:
            dataset_groups = [("all", all_df)]
        else:
            dataset_groups = all_df.groupby('dataset', sort=False)

        for label, group in dataset_groups:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

            sns.boxplot(
                x='epoch',
                y='abs_delta',
                data=group,
                ax=axes[0],
                color='lightsteelblue',
                whis=[0, 100],
                showfliers=False,
            )
            sns.stripplot(
                x='epoch',
                y='abs_delta',
                data=group,
                ax=axes[0],
                color='navy',
                alpha=0.6,
                size=3,
                jitter=True,
            )
            axes[0].set_title(f'Absolute weight changes per epoch ({label})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Absolute Δ weight')

            epoch_order = sorted(group['epoch'].unique())
            for _, weight_group in group.groupby('weight_index'):
                weight_data = weight_group.sort_values('epoch')
                axes[1].plot(
                    epoch_order,
                    weight_data['abs_delta'],
                    color='gray',
                    alpha=0.15,
                    linewidth=1,
                )

            median_by_epoch = group.groupby('epoch')['abs_delta'].median().reindex(epoch_order)
            axes[1].plot(epoch_order, median_by_epoch, color='black', linewidth=2, label='Median trend')
            axes[1].set_title(f'Weight trendlines across epochs ({label})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Absolute Δ weight')
            axes[1].legend()

            plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        x='epoch',
        y='abs_delta',
        data=all_df,
        ax=ax,
        color='lightcoral',
        whis=[0, 100],
        showfliers=False,
    )
    sns.stripplot(
        x='epoch',
        y='abs_delta',
        data=all_df,
        ax=ax,
        color='darkred',
        alpha=0.4,
        size=3,
        jitter=True,
    )
    ax.set_title('Absolute weight changes across all epochs (all datasets combined)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Absolute Δ weight')
    plt.xticks(rotation=15)
    plt.show()

    # Plot convergence toward a stable mean within the first 200 ms
    window = 500 # ms
    convergence_rows = []
    for dataset in datapaths:
        label = Path(dataset).stem
        with np.load(dataset, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}

        ts = data['ts']
        weights = data['pfpc_weights']
        mask = ts <= window
        if np.sum(mask) == 0:
            continue

        ts_short = ts[mask]
        weights_short = weights[mask]

        for weight_idx in range(weights_short.shape[1]):
            for t, w in zip(ts_short, weights_short[:, weight_idx]):
                convergence_rows.append({
                    'dataset': label,
                    'weight_idx': weight_idx,
                    'time_ms': t,
                    'weight': w,
                })

    convergence_df = pd.DataFrame(convergence_rows)
    if not convergence_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Plot individual weight trajectories (faintly)
        # for (label, weight_idx), group in convergence_df.groupby(['dataset', 'weight_idx']):
            # plt.plot(group['time_ms'], group['weight'], alpha=0.01, color='gray')

        # Interpolate all individual weight trajectories to a common 1-ms grid
        time_grid = np.arange(0, window + 1, 1)
        interpolated = []
        for (label, weight_idx), group in convergence_df.groupby(['dataset', 'weight_idx']):
            group_sorted = group.sort_values('time_ms')
            interp = np.interp(time_grid, group_sorted['time_ms'], group_sorted['weight'])
            interpolated.append(interp)

        combined_means = np.vstack(interpolated).mean(axis=0)
        combined_std = np.vstack(interpolated).std(axis=0)

        plt.fill_between(
            time_grid,
            combined_means - combined_std,
            combined_means + combined_std,
            color='gray',
            alpha=0.2,
            label='±1 SD (all datasets)',
        )
        plt.plot(time_grid, combined_means, color='black', linewidth=2.5, label='Combined mean')
        plt.title(f'Convergence of mean PF-PC weights during first {window} ms (all datasets)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean PF-PC weight')
        plt.xlim(0, window)
        plt.legend()
        plt.tight_layout()
        plt.show()