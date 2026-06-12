import time
import numpy as np
import sys
import os
from tqdm.auto import tqdm
import brainpy as bp
import brainpy.math as bm
import traceback
from glob import glob
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import itertools

sys.path.append('C:/Users/HP/PycharmProjects/Internproject 2025/cerebellum-jax-main')

import models.network_plasticity as plastic_net
import models.network as static_net






### ------------- General running functions ---------------####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent

def get_network(condition):
    if condition == "plasticity":
        return plastic_net
    elif condition == "static":
        return static_net
    else:
        raise ValueError(f"Unknown condition: {condition}")

def run_experiment(seed, duration, dt, net_params, run_path, condition, downsample=20,  base_net_params = {}):
    """
    :type seed: int
    :type duration: float
    :type dt: float
    :type net_params: dict
    :type downsample: int
    :type run_path: str
    :type condition: str

    """

    net = get_network(condition)
    current_net_params = base_net_params.copy()
    current_net_params.update(net_params)

    start_time = time.time()
    try:
        runner, io_topography_params, connections_idx = net.run_simulation(
            duration=duration, dt=dt, net_params=current_net_params, seed= int(seed))
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during simulation: {e}\n{full_error}")
    end_time = time.time()

    print("Runner time taken: ", end_time - start_time)

    start_time = time.time()
    data = {k: np.array(runner.mon[k][::downsample]) for k in runner.mon}
    data.update(io_topography_params)
    data.update(connections_idx)
    data.update(net_params)
    data["condition"] = np.array(condition)
    data["seed"] = seed
    data["dt"] = dt * downsample

    np.savez(run_path, **data)
    end_time = time.time()
    print("Saving time taken: ", end_time - start_time)

    # print(f"Simulation time taken: {end_time - start_time:.2f} seconds")


#### -------------- Parameter sweeps -------------------- ####
def run_p_bridges_sweep(amount, n_seeds, n_clusters = 5, n_projections = 4, conditions = ["plasticity", "static"]):
    assert amount % 2 == 0, "Give an even amount of parameter values"

    p_bridges_list = np.linspace(0, 0.9999, amount)
    seeds = np.arange(88, 88 + n_seeds)
    net_params = {}

    total_runs = len(p_bridges_list) * len(seeds) * len(conditions)
    print("Total number of runs:", total_runs)
    print(f"Estimated time = {( total_runs * 16):.2f} seconds",  )

    parent_dir = get_parent_dir()
    results_subdir = os.path.join(
        parent_dir, "results", f"p_bridges_sweep_{time.strftime('%m-%d_%H;%M;%S')}"
    )
    os.makedirs(results_subdir, exist_ok=True)

    start_time = time.time()

    run_count = 0
    for p_bridge in p_bridges_list:

        net_params["IO_n_projections"] = n_projections
        net_params["IO_n_clusters"] = n_clusters
        net_params["IO_bridge_probability"] = p_bridge

        for condition in conditions:
            for seed in seeds:
                run_count += 1
                print(f"Running {run_count}/{total_runs}")

                run_path = os.path.join(results_subdir, f"p_bridge_{p_bridge:.3f}_condition_{condition}_seed_{seed}_runner.npz")
                run_experiment(seed, duration = 10_000.0, dt = 0.025,net_params = net_params,downsample = 30,
                                run_path = run_path, condition = condition)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return results_subdir

def run_n_clusters_sweep(amount, n_seeds, step= 1 , p_bridges = 0.1, n_projections = 4 , conditions= ["plasticity", "static"], start_seed = 88):
    assert amount % 2 == 0 ,"Give an even amount of parameter values"

    n_clusters_list = np.arange(1, amount, step)
    seeds = np.arange(start_seed, start_seed + n_seeds)
    net_params = {}

    total_runs = len(n_clusters_list) * len(seeds) * len(conditions)
    print("Total number of runs:", total_runs)
    print(f"Estimated time = {(total_runs * 16):.2f} seconds", )

    parent_dir = get_parent_dir()
    results_subdir = os.path.join(
        parent_dir, "results", f"n_clusters_sweep_{time.strftime('%m-%d_%H;%M;%S')}"
    )
    os.makedirs(results_subdir, exist_ok=True)

    start_time = time.time()

    run_count = 0
    for n_clusters in n_clusters_list:

        net_params["IO_n_projections"] = n_projections
        net_params["IO_n_clusters"] = n_clusters
        net_params["IO_bridge_probability"] = p_bridges

        for condition in conditions:
            for seed in seeds:
                run_count += 1
                print(f"Running {run_count}/{total_runs}")

                run_path = os.path.join(results_subdir,f"n_clusters_{n_clusters}_condition_{condition}_seed_{seed}_runner.npz")
                run_experiment(seed = seed,duration = 10_000, dt = 0.025,net_params = net_params,downsample = 30,run_path = run_path,condition = condition)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return results_subdir

def run_n_projections_sweep(amount, n_seeds, step=1, n_clusters = 5, p_bridges = 0.10 ,conditions = ['static', 'plasticity'], start_seed = 88):
    assert amount % 2 == 0, "Give an even amount of parameter values"

    n_projections_list = np.arange(0, amount, step)
    seeds = np.arange(start_seed, start_seed + n_seeds)
    net_params = {}

    total_runs = len(n_projections_list) * len(seeds) * len(conditions)
    print("Total number of runs:", total_runs)
    print(f"Estimated time = {(total_runs * 16):.2f} seconds", )

    parent_dir = get_parent_dir()
    results_subdir = os.path.join(
        parent_dir, "results", f"n_projections_sweep_{time.strftime('%m-%d_%H;%M;%S')}"
    )
    os.makedirs(results_subdir, exist_ok=True)

    start_time = time.time()

    run_count = 0
    for n_projections in n_projections_list:

        net_params["IO_n_projections"] = n_projections
        net_params["IO_n_clusters"] = n_clusters
        net_params["IO_bridge_probability"] = p_bridges

        for condition in conditions:
            for seed in seeds:
                run_count += 1
                print(f"Running {run_count}/{total_runs}")

                run_path = os.path.join(results_subdir,
                                        f"n_projections_{n_projections}_condition_{condition}_seed_{seed}_runner.npz")

                run_experiment(seed=seed, duration=10_000, dt=0.025, net_params=net_params, downsample=30,
                               run_path=run_path, condition=condition)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return results_subdir

def combined_sweep(amount, n_seeds,conditions = ['static', 'plasticity'], start_seed = 88):
    if amount > 40:
        amount = 40
    assert amount % 2 == 0, "Give an even amount of parameter values"

    n_clusters_list = np.arange(1,  amount)
    n_projections_list = np.arange(0,amount)
    p_bridges_list = np.linspace(0,0.9999, amount)
    seeds = np.arange(start_seed, start_seed + n_seeds)
    net_params = {}

    total_runs = len(p_bridges_list) * len(n_clusters_list) * len(n_projections_list) * len(seeds) * len(conditions)
    print("Total number of runs:", total_runs)
    print(f"Estimated time = {(total_runs * 16):.2f} seconds", )

    parent_dir = get_parent_dir()
    results_subdir = os.path.join(
        parent_dir, "results", f"combined_sweep_{time.strftime('%m-%d_%H;%M;%S')}"
    )
    os.makedirs(results_subdir, exist_ok=True)

    start_time = time.time()

    run_count = 0
    for n_projections, n_clusters, p_bridges in itertools.product(n_projections_list, n_clusters_list, p_bridges_list):

        net_params["IO_n_projections"] = n_projections
        net_params["IO_n_clusters"] = n_clusters
        net_params["IO_bridge_probability"] = p_bridges

        for condition in conditions:
            for seed in seeds:
                run_count += 1
                print(f"Running {run_count}/{total_runs}")

                run_path = os.path.join(results_subdir,f"n_clusters_{n_clusters}_p_bridges_{p_bridges}_n_projections_{n_projections}_condition_{condition}_seed_{seed}_runner.npz")

                run_experiment(seed=seed, duration=10_000, dt=0.025, net_params=net_params, downsample=30,
                               run_path=run_path, condition=condition)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return results_subdir

# def clustered_projections()



