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
import models.load_state_patch

sys.path.append('C:/Users/HP/ModellingProjects/Olivocerebellar-circuit')
from models.setup_net_run import init_net_and_runner, run_until_convergence, run_simulation


# - command generator function for all testing (+pretraining) conditions

CLUSTER_PARENT_DIR = "/home/izet/Olivocerebellar-circuit"


### ------------- General running functions ---------------####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent
    

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


def force_net_params(net, net_params):
    
    net.pf_to_pc_BCM.plasticity_on.value = bm.asarray(net_params['PFPC_plasticity_on'])

    net.stim.stim_io_on.value = bm.asarray(net_params['OU_stim_io_on'])
    net.stim.stim_pf_on.value = bm.asarray(net_params['OU_stim_pf_on'])
    net.stim.isi_mean.value = bm.asarray(net_params['OU_stim_isi_mean'])
    net.stim.isi_std.value = bm.asarray(net_params['OU_stim_isi_std'])
    net.stim.stim_freq = net_params['OU_stim_freq']

    net.stim.dur_io = net_params['OU_stim_dur_io_mean']
    net.stim.dur_pf= net_params['OU_stim_dur_pf_mean']
    net.stim.amp_io = net_params['OU_stim_amp_io_mean']
    net.stim.amp_pf = net_params['OU_stim_amp_pf_mean']

    return net


def run_train(config):
    current_net_params = config['net_params']
    run_params = config['run_params']
    downsample = run_params['downsample']
    max_runtime = run_params['simdur']
    epoch_time = run_params['epoch_time']
    conv_thresh_m = run_params['conv_thresh_m']
    conv_thresh_var = run_params['conv_thresh_var']
    conv_chunk_thresh = run_params['conv_chunk_thresh']
   

    net, runner = init_net_and_runner(current_net_params)

    start_time = time.time()
    try:
        net, runner, data, mean_w_final, var_w_final, runtime = run_until_convergence(
            net,
            runner,
            downsample,
            max_runtime=max_runtime,
            epoch=epoch_time,
            conv_thresh_m=conv_thresh_m,
            conv_thresh_var=conv_thresh_var,
            chunk_thresh=conv_chunk_thresh,
        )
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during training simulation: {e}\n{full_error}")
        raise
    end_time = time.time()

    if runtime < max_runtime:
        print(f"Converged at t={runtime} ms (Δmu_w={mean_w_final:.2e}, Δvar_w={var_w_final:.2e})")
    else:
        print(f"Not converged, t={max_runtime} ms")
    print(f"Training simulation time taken = {end_time - start_time} s")

    start_time = time.time()

    # Save state
    state = bp.save_state(net)
    bc.save_pytree(str(config['fin_state_path']), state)


    # Prepare and save data
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    
    np.savez(config["run_path"], **data)
    print(f"Saved training runner data to {config['run_path']}")
    end_time = time.time()
    print(f"Training saving time taken: {end_time - start_time} s")

    return net, data, config['fin_state_path']


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

    pretraining_state_path = str(config["pretraining_state_path"])

    if not os.path.exists(pretraining_state_path):
        raise FileNotFoundError(f"Pretraining state path not found: {pretraining_state_path}")

    start_time = time.time()
    net, runner = init_net_and_runner(current_net_params)

    # Load pretraining state
    state = bc.load_pytree(pretraining_state_path)
    result = bp.load_state(net, state)
    net = force_net_params(net, current_net_params)  # Ensure current net params are applied after loading state
    end_time = time.time()

    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"State loading failed. Missing keys: {result.missing_keys}, Unexpected keys: {result.unexpected_keys}")  

    print(f"State loading time taken = {end_time - start_time} s")


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
    
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    np.savez(config["run_path"], **data)
    print(f"Saved test runner data to {config['run_path']}")
    end_time = time.time()
    print(f"Test saving time taken: {end_time - start_time} s")

    return net, data






################# -------------- Experiments / command generators  -------------- ##################

# def run_splitter():
#     return []

def baseline_commands(parent_dir, monitor= "plasticity_min", n_seeds=4, simdur= 480_000, experiment = "nostim", downsample = 80,tag = None):
    seedlist = np.arange(88, 88+ n_seeds)

    parent_dir = Path(CLUSTER_PARENT_DIR)
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
            f" --monitor-preset \"{monitor}\""
            f" --simdur {np.float64(simdur)}"
            f" --parent-dir {CLUSTER_PARENT_DIR}"
            f" --timestep 0.5"
            f" --downsample {downsample}"
            + (f" --tag \"{tag}\"" if tag else "")
        )

        jobs.append({
            "command": command,
            "run_path": str(run_path),
            "figures_dir": str(figures_dir),
            "seed": seed,
        })

    return jobs


def train_commands(parent_dir, monitor, n_seeds=4, simdur=480_000, ISI_values=None, experiment="fixed-isi", ISI_std=0.0, tag=None):
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
        Experiment type: "fixed-isi", "variable-isi", or "nostim"
    tag : str or None
        Optional tag appended to folder names
 
    
    Returns:
    --------
    list of dict
        Each dict contains: command, run_path, fin_state_path, fin_state_dir, figures_dir, seed, ISI
    """
    seedlist = np.arange(88, 88 + n_seeds)
    if ISI_values is None:
        ISI_values = [120.0]
    else:
        ISI_values = np.atleast_1d(ISI_values)
    
    parent_dir = Path(CLUSTER_PARENT_DIR)
   
    tag = f"_{tag}" if tag else ""
    results_dir = parent_dir / "results" / f"stim_experiments_train_{experiment}_{tag}"
    figures_dir = parent_dir / "figures" / f"figs_train_{experiment}_{tag}"
    fin_state_dir = parent_dir / "states" / f"states_{experiment}_{tag}"
    jobs = []
    
    for ISI in ISI_values:
        for seed in seedlist:
           
            fin_state_fname = (f"{experiment}"f"_isi{ISI:.1f}_isi_std{ISI_std:.1f}_seed{seed}_simdur{simdur}_state.bp")
            fin_state_path = fin_state_dir / fin_state_fname
            run_fname = f"train_{experiment}_isi{ISI:.1f}_seed{seed}_simdur{np.float64(simdur)}.npz"
            run_path = results_dir / run_fname
            
            command = (
                f"python3 main_entrain.py --run-type train --experiment {experiment}"
                f" --PFPC_plasticity-on True"
                f" --seed {seed}"
                f" --monitor-preset \"{monitor}\""
                f" --simdur {np.float64(simdur)}"
                f" --parent-dir {CLUSTER_PARENT_DIR}"
                f" --OU-stim-isi-mean {ISI}"
                f" --OU-stim-isi-std {ISI_std}"
                + (f" --tag \"{tag}\"" if tag else "")
            )
            
            if experiment == "variable-isi":
                command = command.replace("--PFPC_plasticity-on True", 
                                        f"--PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-std {ISI_std} --OU-stim-isi-mean {ISI}")
            
            elif experiment == "fixed-isi":
                command = command.replace("--PFPC_plasticity-on True", 
                                        f"--PFPC_plasticity-on True --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-mean {ISI}")
            elif experiment == "nostim":
                command = command.replace("--PFPC_plasticity-on True", 
                                        f"--PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False")
            
            jobs.append({
                "command": command,
                "run_path": str(run_path),
                "fin_state_path": str(fin_state_path),
                "fin_state_dir": str(fin_state_dir),
                "figures_dir": str(figures_dir),
                "seed": seed,
                "ISI": ISI,
            })
    
    return jobs


def test_commands(parent_dir, monitor, n_seeds=4, simdur=480_000, ISI_values=None, ISI_std=None, experiment="fixed-isi", 
                  pretraining_snapshot_paths=None, tag=None):
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
        Standard deviation for variable-isi experiment
    experiment : str
        Experiment type: "fixed-isi", "variable-isi", or "nostim"
    pretraining_snapshot_paths : list of str or None
        List of full paths to pretraining snapshot files (.bp files)
        If None, raises ValueError
    tag : str or None
        Optional tag appended to folder names

    
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
    
    parent_dir = Path(CLUSTER_PARENT_DIR)
    tag = f"_{tag}" if tag else ""
    test_results_root = parent_dir / "results" / f"stim_experiments_test_{experiment}_{tag}"
    test_figures_root = parent_dir / "figures" / f"figs_test_{experiment}_{tag}"
    jobs = []
    
    for pretrain_snapshot_path in pretraining_snapshot_paths:
        pretraining_label = pretrain_snapshot_path.stem.replace("_state", "")
        test_results_dir = test_results_root / pretraining_label
        test_figures_dir = test_figures_root / pretraining_label

        snapshot_parts = pretraining_label.split("_")
        pretrain_isi = None
        pretrain_seed = None
        pretrain_simdur = None
        
        for i, part in enumerate(snapshot_parts):
            if part == "isi" and i + 1 < len(snapshot_parts):
                try:
                    pretrain_isi = float(snapshot_parts[i + 1])
                except (ValueError, IndexError):
                    pass
            elif part == "seed" and i + 1 < len(snapshot_parts):
                try:
                    pretrain_seed = int(snapshot_parts[i + 1])
                except (ValueError, IndexError):
                    pass
            elif part == "simdur" and i + 1 < len(snapshot_parts):
                try:
                    pretrain_simdur = float(snapshot_parts[i + 1])
                except (ValueError, IndexError):
                    pass

        pretraining_info = {
            "pretrain_isi": pretrain_isi,
            "pretrain_seed": pretrain_seed,
            "pretrain_simdur": pretrain_simdur
            }
        
        pretrain_info_str = f"isi{pretrain_isi:.1f}_seed{pretrain_seed}" if pretrain_isi is not None else pretraining_label
        for test_isi in ISI_values:
            for test_seed in seedlist:
                run_fname = f"test_{experiment}_isi{test_isi:.1f}_seed{test_seed}_simdur{np.float64(simdur)}.npz"
                run_path = test_results_dir / run_fname
                
                command = (
                    f"python3 main_entrain.py --run-type test --experiment {experiment}"
                    f" --PFPC_plasticity-on False"
                    f""
                    f" --seed {test_seed}"
                    f" --monitor-preset \"{monitor}\""
                    f" --simdur {np.float64(simdur)}"
                    f" --parent-dir {CLUSTER_PARENT_DIR}"
                    f" --pretraining-path \"{pretrain_snapshot_path}\""
                    + (f" --tag \"{tag}\"" if tag else "")
                )
                
                if experiment == "variable-isi":
                    command += f" --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-std {ISI_std} --OU-stim-isi-mean {test_isi}"
                elif experiment == "fixed-isi":
                    command += f" --OU-stim-io-on True --OU-stim-pf-on True --OU-stim-isi-mean {test_isi}"
                elif experiment == "nostim":
                    command += " --OU-stim-io-on False --OU-stim-pf-on False"
                
                jobs.append({
                    "command": command,
                    "run_path": str(run_path),
                    "figures_dir": str(test_figures_dir),
                    "seed": test_seed,
                    "ISI": test_isi,
                    "pretraining_snapshot_path": str(pretrain_snapshot_path),
                    "pretraining_info": pretraining_info,
                    "pretraining_label": pretraining_label,
                })
    
    return jobs

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
