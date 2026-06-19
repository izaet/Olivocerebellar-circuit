import time
import copy
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
import json
import brainpy.checkpoints as bc

sys.path.append('C:/Users/HP/ModellingProjects/Olivocerebellar-circuit')
from models.setup_net_run import init_net_and_runner, restore_state, run_until_convergence, run_simulation


# - command generator function for all testing (+pretraining) conditions



### ------------- General running functions ---------------####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent
    

def save_snapshot(config, state, metadata):
    snapshot_dir = config['snapshot_dir']
    os.makedirs(snapshot_dir, exist_ok=True)
    state_path = os.path.join(snapshot_dir, "_state.bp")
    bc.save_pytree(state_path, state)


def load_snapshot(snapshot_dir):
    state_path = os.path.join(snapshot_dir, "_state.bp")
    state = bc.load_pytree(state_path)
    return state

def get_connections(net):

    connections_idx = {
        "pf_pc_pre": net.pf_to_pc_BCM.pre_idx,
        "pf_pc_post": net.pf_to_pc_BCM.post_idx,
        "io_pc_pre": net.io_to_pc.io_source_indices,
        "io_pc_post": net.io_to_pc.pc_target_indptr,
    }

    return connections_idx 

def get_io_topography (net):

    io_topography = {
        "n_bridges": net.io.n_bridges,
        "io_src": np.array(net.io.neurons.gj_src),
        "io_tgt": np.array(net.io.neurons.gj_tgt),
        "io_cluster_ids": net.io.cluster_ids,
        "n_neurons": net.num_io}

    return io_topography

def run_train(config):
    current_net_params = config['net_params']
    run_params = config['run_params']
    downsample = run_params['downsample']
    max_runtime = run_params['simdur']
    epoch_time = run_params['epoch_time']

    net, runner = init_net_and_runner(current_net_params)

    start_time = time.time()
    try:
        net, runner, data, d_w_chunk_max, runtime = run_until_convergence(
            net,
            runner,
            downsample,
            max_runtime=max_runtime,
            epoch=epoch_time,
        )
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during training simulation: {e}\n{full_error}")
        raise
    end_time = time.time()

    if runtime < max_runtime:
        print(f"Converged at t={runtime} ms (max Δw={d_w_chunk_max:.2e})")
    else:
        print(f"Not converged, t={max_runtime} ms")
    print(f"Training simulation time taken = {end_time - start_time} s")

    start_time = time.time()
    state = bp.save_state(net)
    save_snapshot(config, state, copy.deepcopy(config))
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    np.savez(config["run_path"], **data)
    print(f"Saved training runner data to {config['run_path']}")
    end_time = time.time()
    print(f"Training saving time taken: {end_time - start_time} s")

    return net, data, state


def run_baseline(config):
    current_net_params = config['net_params']
    run_params = config['run_params']
    downsample = run_params['downsample']
    duration = run_params['simdur']

    net, runner = init_net_and_runner(current_net_params)

    start_time = time.time()
    try:
        net, runner, data = run_simulation(net, runner, duration, downsample)
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during baseline simulation: {e}\n{full_error}")
        raise
    end_time = time.time()
    print(f"Baseline simulation time taken = {end_time - start_time} s")

    start_time = time.time()
   
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    np.savez(config["run_path"], **data)
    print(f"Saved baseline runner data to {config['run_path']}")
    end_time = time.time()
    print(f"Baseline saving time taken: {end_time - start_time} s")

    return net, data


def run_test(config):
    current_net_params = config['net_params']
    run_params = config['run_params']
    downsample = run_params['downsample']
    duration = run_params['simdur']

    pretraining_snapshot_dir = config.get("pretraining_snapshot_dir")
    if pretraining_snapshot_dir is None:
        raise ValueError("run_test requires a pretraining_snapshot_dir in config")
    if not os.path.exists(pretraining_snapshot_dir):
        raise FileNotFoundError(f"Pretraining snapshot not found: {pretraining_snapshot_dir}")

    pretrain_state = load_snapshot(pretraining_snapshot_dir)
    net, runner = init_net_and_runner(current_net_params)
    net = restore_state(net, pretrain_state)

    start_time = time.time()
    try:
        net, runner, data = run_simulation(net, runner, duration, downsample)
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during test simulation: {e}\n{full_error}")
        raise
    end_time = time.time()
    print(f"Test simulation time taken = {end_time - start_time} s")

    start_time = time.time()
    state = bp.save_state(net)
    save_snapshot(config, state, copy.deepcopy(config))
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    np.savez(config["run_path"], **data)
    print(f"Saved test runner data to {config['run_path']}")
    end_time = time.time()
    print(f"Test saving time taken: {end_time - start_time} s")

    return net, data, state






################# -------------- Experiments / command generators  -------------- ##################

            
def specific_training_commands(ISI_values = np.linspace(20, 200, 4), n_seeds = 4):
    seedlist = np.arange(88, 88+ n_seeds)
    for ISI in ISI_values:
        for seed in seedlist:
            print(f'python3 main_entrain.py --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean{ISI} --seed {seed}')
        

    

def baseline_commands(parent_dir, n_seeds=4, simdur= 480_000, experiment = "nostim", tag = None, timestamp = None):
    seedlist = np.arange(88, 88+ n_seeds)

    
    parent_dir = Path(parent_dir)
    timestamp = timestamp or time.strftime("%m-%d_%H;%M;%S")
    tag = f"_{tag}" if tag else ""
    results_dir = parent_dir / "results" / f"stim_experiments_baseline_{experiment}_{tag}"
    figures_dir = parent_dir / "figures" / f"figs_baseline_{experiment}_{tag}"
    jobs = []


    for seed in seedlist:
        run_fname = f"baseline_{experiment}_seed{seed}_simdur{np.float64(simdur)}.npz"
        run_path = results_dir / run_fname
        


        command = (
            f"python3 main_entrain.py --run-type baseline --experiment {experiment} --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False"
            f" --seed {seed}"
            f" --simdur {np.float64(simdur)}"
            f" --parent-dir /home/izet/Olivocerebellar-circuit"
            f" --timestamp \"{timestamp}\""
            + (f" --tag \"{tag}\"" if tag else "")
        )

        jobs.append({
            "command": command,
            "run_path": str(run_path),
            "figures_dir": str(figures_dir),
            "seed": seed,
        })

    return jobs


def train_commands(parent_dir, n_seeds=4, simdur=480_000, ISI_values=None, experiment="specific-isi", ISI_std=None, tag=None, timestamp=None):
    """
    Generate training commands with structured job information.
    
    Parameters:
    -----------
    parent_dir : str or Path
        Base directory for results/states/figures
    n_seeds : int
        Number of different seeds to use (starting from 88)
    simdur : float
        Maximum simulation duration (ms)
    ISI_values : array-like or None
        If None, uses single default ISI of 120.0 ms
        If array-like, generates a job for each ISI value
    experiment : str
        Experiment type: "specific-isi", "random-isi", or "nostim"
    tag : str or None
        Optional tag appended to folder names
    timestamp : str or None
        Fixed timestamp for deterministic naming; if None uses current time
    
    Returns:
    --------
    list of dict
        Each dict contains: command, run_path, snapshot_path, snapshot_dir, figures_dir, seed, ISI
    """
    seedlist = np.arange(88, 88 + n_seeds)
    if ISI_values is None:
        ISI_values = [120.0]
    else:
        ISI_values = np.atleast_1d(ISI_values)
    
    parent_dir = Path(parent_dir)
    timestamp = timestamp or time.strftime("%m-%d_%H;%M;%S")
    tag = f"_{tag}" if tag else ""
    results_dir = parent_dir / "results" / f"stim_experiments_train_{experiment}_{tag}"
    figures_dir = parent_dir / "figures" / f"figs_train_{experiment}_{tag}"
    snapshot_dir = parent_dir / "states" / f"states_{experiment}_isi{ISI:.1f}_seed{seed}_{tag}"
    jobs = []
    
    for ISI in ISI_values:
        for seed in seedlist:
           
            snapshot_fname = f"{experiment}_isi{ISI:.1f}_seed{seed}_simdur{np.float64(simdur)}_state.bp"
            snapshot_path = snapshot_dir / snapshot_fname
            run_fname = f"train_{experiment}_isi{ISI:.1f}_seed{seed}_simdur{np.float64(simdur)}.npz"
            run_path = results_dir / run_fname
            
            command = (
                f"python3 main_entrain.py --run-type train --experiment {experiment}"
                f" --PFPC_plasticity-on True"
                f" --seed {seed}"
                f" --simdur {np.float64(simdur)}"
                f" --parent-dir {str(parent_dir)}"
                f" --timestamp \"{timestamp}\""
                + (f" --tag \"{tag}\"" if tag else "")
            )
            
            if experiment == "random-isi":
                command = command.replace("--PFPC_plasticity-on True", 
                                        "--PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-std {ISI_std} --OU-stim-isi-mean {ISI}")
            
            elif experiment == "specific-isi":
                command = command.replace("--PFPC_plasticity-on True", 
                                        "--PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-mean {ISI}")
            elif experiment == "nostim":
                command = command.replace("--PFPC_plasticity-on True", 
                                        "--PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False")
            
            jobs.append({
                "command": command,
                "run_path": str(run_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_dir": str(snapshot_dir),
                "figures_dir": str(figures_dir),
                "seed": seed,
                "ISI": ISI,
            })
    
    return jobs


def test_commands(parent_dir, n_seeds=4, simdur=480_000, ISI_values=None, ISI_std=None, experiment="specific-isi", 
                  pretraining_snapshot_paths=None, tag=None, timestamp=None):
    """
    Generate test commands with structured job information.
    
    Parameters:
    -----------
    parent_dir : str or Path
        Base directory for results/states/figures
    n_seeds : int
        Number of different seeds to use (starting from 88) for test runs
    simdur : float
        Simulation duration (ms)
    ISI_values : array-like or None
        ISI values to test. If None, uses 120.0 ms
    ISI_std : float or None
        Standard deviation for random-isi experiment
    experiment : str
        Experiment type: "specific-isi", "random-isi", or "nostim"
    pretraining_snapshot_paths : list of str or None
        List of full paths to pretraining snapshot files (.bp files)
        If None, raises ValueError
    tag : str or None
        Optional tag appended to folder names
    timestamp : str or None
        Fixed timestamp for deterministic naming; if None uses current time
    
    Returns:
    --------
    list of dict
        Each dict contains: command, run_path, snapshot_path, snapshot_dir, figures_dir, 
                          seed, ISI, pretraining_snapshot_path, pretraining_info
    """
    if pretraining_snapshot_paths is None or len(pretraining_snapshot_paths) == 0:
        raise ValueError("test_commands requires a non-empty list of pretraining_snapshot_paths")
    
    pretraining_snapshot_paths = [Path(p) for p in pretraining_snapshot_paths]
    
    seedlist = np.arange(88, 88 + n_seeds)
    if ISI_values is None:
        ISI_values = [120.0]
    else:
        ISI_values = np.atleast_1d(ISI_values)
    
    parent_dir = Path(parent_dir)
    timestamp = timestamp or time.strftime("%m-%d_%H;%M;%S")
    tag = f"_{tag}" if tag else ""
    results_dir = parent_dir / "results" / f"stim_experiments_test_{experiment}_{tag}"
    figures_dir = parent_dir / "figures" / f"figs_test_{experiment}_{tag}"
    jobs = []
    
    for pretrain_snapshot_path in pretraining_snapshot_paths:
        # Extract pretraining info from snapshot filename
        # Filename format: {experiment}_isi{ISI:.1f}_seed{seed}_simdur{simdur}_state.bp
        snapshot_fname = pretrain_snapshot_path.stem  # Remove .bp extension
        snapshot_parts = snapshot_fname.replace("_state", "").split("_")
        
        # Parse pretraining info from filename
        pretrain_experiment = None
        pretrain_isi = None
        pretrain_seed = None
        pretrain_simdur = None
        
        for i, part in enumerate(snapshot_parts):
            if part == "isi" and i + 1 < len(snapshot_parts):
                # Extract ISI value
                isi_str = snapshot_parts[i + 1]
                try:
                    pretrain_isi = float(isi_str)
                except (ValueError, IndexError):
                    pass
            elif part == "seed" and i + 1 < len(snapshot_parts):
                # Extract seed value
                seed_str = snapshot_parts[i + 1]
                try:
                    pretrain_seed = int(seed_str)
                except (ValueError, IndexError):
                    pass
            elif part == "simdur" and i + 1 < len(snapshot_parts):
                # Extract simdur value
                simdur_str = snapshot_parts[i + 1]
                try:
                    pretrain_simdur = float(simdur_str)
                except (ValueError, IndexError):
                    pass
            elif i == 0:
                pretrain_experiment = part
        
        pretrain_info_str = f"isi{pretrain_isi:.1f}_seed{pretrain_seed}" if pretrain_isi is not None else "unknown"
        
        for test_isi in ISI_values:
            for test_seed in seedlist:
                # Create output snapshot directory and path for test run
                snapshot_dir = parent_dir / "states" / f"states_test_{experiment}_isi{test_isi:.1f}_seed{test_seed}"
                snapshot_fname_out = f"test_{experiment}_pretrain_{pretrain_info_str}_isi{test_isi:.1f}_seed{test_seed}_simdur{np.float64(simdur)}_state.bp"
                snapshot_path = snapshot_dir / snapshot_fname_out
                
                # Create output run filename
                run_fname = f"test_{experiment}_pretrain_{pretrain_info_str}_isi{test_isi:.1f}_seed{test_seed}_simdur{np.float64(simdur)}.npz"
                run_path = results_dir / run_fname
                
                command = (
                    f"python3 main_entrain.py --run-type test --experiment {experiment}"
                    f" --PFPC_plasticity-on False --OU-stim-isi-mean {test_isi}"
                    f" --seed {test_seed}"
                    f" --simdur {np.float64(simdur)}"
                    f" --parent-dir {str(parent_dir)}"
                    f" --timestamp \"{timestamp}\""
                    f" --pretraining-tag \"{pretrain_snapshot_path.parent.name}\""
                    + (f" --tag \"{tag}\"" if tag else "")
                )
                
              
                
                if experiment == "random-isi":
                    command = command.replace("--PFPC_plasticity-on False", 
                                            "--PFPC_plasticity-on False --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-std {ISI_std} --OU-stim-isi-mean {test_isi}")
                
                elif experiment == "specific-isi":
                    command = command.replace("--PFPC_plasticity-on False", 
                                            "--PFPC_plasticity-on False --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-mean {test_isi}")
                elif experiment == "nostim":
                    command = command.replace("--PFPC_plasticity-on False", 
                                            "--PFPC_plasticity-on False --OU-stim-io-on False --OU-stim-pf-on False")
                
                jobs.append({
                    "command": command,
                    "run_path": str(run_path),
                    "snapshot_path": str(snapshot_path),
                    "snapshot_dir": str(snapshot_dir),
                    "figures_dir": str(figures_dir),
                    "seed": test_seed,
                    "ISI": test_isi,
                    "pretraining_snapshot_path": str(pretrain_snapshot_path),
                    "pretraining_info": pretrain_info_str,
                })
    
    return jobs

    

def random_training_commands(n_seeds=4, duration = 10_000):
    seedlist = np.arange(88, 88+ n_seeds)
    for seed in seedlist:
        print(f'python3 main_entrain.py --experiment random-isi --run-type train --PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --seed {seed} --simdur {duration}')    

def testing_commands(n_seeds=4, duration = 10_000):
    seedlist = np.arange(88, 88+ n_seeds)
    for seed in seedlist:
        print(f'python3 main_entrain.py --experiment specific-isi --run-type test --PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --seed {seed} --simdur {duration}')


########## ------------ Merging data for analysis ------------- ############

def combine_data(results_dir,pattern= "*.npz"):
    """
    Combines all runs within an experiment. Assumes number of connections between populations and the size of populations is the same for all runs in the experiment.

    :param results_subdir:
    :param variable:
    :return:
    """

    parent_dir = str(get_parent_dir())
    dataset_dir = os.path.join(parent_dir, "data", "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    base = os.path.basename(results_dir)
    base_split = base.rsplit("_", 2)
    experiment_name = base_split[0]
    experiment_time = base_split[1] + "_" + base_split[2]

    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {results_dir} with pattern {pattern}")

    runs = []
    for f in files:

        with np.load(f, allow_pickle=True) as npz:
            run_data = {k: npz[k] for k in npz.files}  # convert NpzFile -> dict
        run_data["source_file"] = np.array(f)  # optional: remember where it came from
        runs.append(run_data)


    combi_data = {}
    for key in runs[0].keys():
        arr = runs[0][key]
        shape = (len(runs),) + (() if np.isscalar(arr) or np.ndim(arr) == 0 else arr.shape)
        combi_data[key] = np.zeros(shape, dtype=arr.dtype)

        for run_id in np.arange(len(runs)):
            combi_data[key][run_id] = runs[run_id][key]


    all_seeds = np.unique(combi_data['seed'])
    all_isis = np.unique(combi_data['isi'])



    DCN_spike_presence = np.any(combi_data["CN_start_spikes"][0] != None)
    Gamma_CN_IO = combi_data["CNIO_gamma_CN_IO"][0]
    file_path = os.path.join(dataset_dir, f"{experiment_name}_gamma_CN={Gamma_CN_IO}_DCN_spikes={DCN_spike_presence}_n_runs_{len(files)}_seeds_{np.min(all_seeds)}-{np.max(all_seeds)}_combined.npz")
    np.savez(file_path, **combi_data)

    return file_path
