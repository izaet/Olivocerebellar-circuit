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
import json
import brainpy.checkpoints as bc

sys.path.append('C:/Users/HP/ModellingProjects/Olivocerebellar-circuit')
from models.setup_net_run import *


# - command generator function for all testing (+pretraining) conditions



### ------------- General running functions ---------------####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent
    

def save_snapshot(config, state, metadata):
    snapshot_dir= config['snapshot_dir']

    # Save state and metadata
   
    state_path = os.path.join(snapshot_dir, "_state.bp")
    bc.save_pytree(state_path, state)
   

    # meta_path = os.path.join(snapshot_dir, "_meta.json")
    # with open(meta_path, "w") as f:
    #     json.dump(metadata, f, indent=2)


def load_snapshot(snapshot_dir):
     
    
    state_path = os.path.join(snapshot_dir, "_state.bp")
    state= bc.load_pytree(state_path)

    # meta_path = os.path.join(snapshot_dir, "_meta.json")
    # with open(meta_path, "r") as f:
    #     meta = json.load(f)
    # , meta

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
    seed =  run_params['seed']
    dt =  run_params['dt']
    downsample =  run_params['downsample']
    max_runtime = run_params['simdur']
    epoch_time = run_params['epoch_time']

    # Get network
    net, runner = init_net_and_runner(current_net_params)

    # Run experiment
    start_time = time.time()

    try:
        net, runner, data, d_w_chunk_max, runtime  = run_until_convergence(net, runner, downsample, max_runtime= max_runtime, epoch = epoch_time)
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during simulation: {e}\n{full_error}")
   
    end_time = time.time()
    
    if runtime < runtime:
        print(f"Converged at t={runtime} ms (max Δw={d_w_chunk_max:.2e})")
    else:
        print(f"Not converged, t={max_runtime} ms")

    print(f"Simulation time taken = {end_time- start_time} s")


    start_time = time.time()

    # Get network state and save
    state= bp.save_state(net)
    metadata = copy.deepcopy(config)

    save_snapshot(config, state, metadata)
    
    data.update(current_net_params)
    data.update(run_params)
    data.update(get_connections(net))
    
    np.savez(config["run_path"], **data)

    end_time = time.time()
    print(f"Saving time taken: {end_time - start_time} s ")

    return net, data, state



def run_test(config):

    current_net_params = config['net_params']
    seed = config['seed']
    sim_duration = config['simtime']
    dt = config['dt']
    downsample = config


    # Get network
    


    # Run experiment
    start_time = time.time()

    try:
        runner, io_topography_params, connections_idx = net.simulate(
            sim_duration=sim_duration, dt=dt, net_params=current_net_params, seed= int(seed))
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during simulation: {e}\n{full_error}")
   
    end_time = time.time()
    print("Runner time taken: ", end_time - start_time)

    # Save data
    start_time = time.time()
    data = {k: np.array(runner.mon[k][::downsample]) for k in runner.mon}
    data.update(net_params)






################# -------------- Experiments / command generators  -------------- ##################

# Each function should generate both a config file for local running and a command for CLI running

# def testrun(mode, n_seeds, conditions=('plasticity', 'static'), sim_duration= 10_000, stim_amp= 1.4, ISI_array= np.arange(80, 120, 20) ):
#     seeds = np.arange(88, 88 + n_seeds)

#     total_runs = len(seeds)* len(conditions) * len(ISI_array)
#     print("Total number of runs:", total_runs )
#     # print(f"Estimated time = {(total_runs * 16):.2f} seconds", )

#     parent_dir = get_parent_dir()
#     results_subdir = os.path.join(
#         parent_dir, "results", f"stim_experiments_test_{time.strftime('%m-%d_%H;%M;%S')}"
#     )
#     os.makedirs(results_subdir, exist_ok=True)

#     start_time = time.time()

#     run_count = 0
#     for cond in conditions:
#         for isi in ISI_array:
#             for seed in seeds:
#                 run_count += 1
#                 print(f"Running {run_count}/{total_runs}")

#                 net_params = {}
                
#                 net_params["OU_stim_isi_mean"] = isi
#                 net_params["OU_stim_freq"] = 700 # ms
#                 net_params["OU_stim_start"] = 1000 # ms
#                 net_params["OU_stim_amp_io_mean"] = stim_amp
#                 net_params["OU_stim_amp_pf_mean"] = stim_amp
#                 net_params[ "OU_stim_dur_io_mean"] = 300 # ms
#                 net_params["OU_stim_dur_pf_mean"] = 300 #ms


#                 run_path = os.path.join(results_subdir, f"seed_{seed}_condition_{cond}_ISI_{isi}_runner.npz")
#                 run_experiment(seed, condition= cond, sim_sim_duration=sim_sim_duration, dt=0.025, net_params=net_params, downsample=20,
#                             run_path=run_path)

#     end_time = time.time()
#     print(f"Total time taken: {end_time - start_time:.2f} seconds")

#     return results_subdir

# def run_training(ISI_values, n_seeds):

#     # for train_condition in training_conditions_all
#     # train until convergence + snapshot (via run_train)
    
#     # Directory for saved converged states
#     parent_dir = get_parent_dir()
#     snapshot_subdir = os.path.join(
#         parent_dir, "converged_states", f"{time.strftime('%m-%d--%H;%M;%S')}_"
#     )
#     os.makedirs(results_subdir, exist_ok=True)
    
#     seedlist= np.arange(88, 88 + n_seeds)

#     total_runs = len(seedlist)* len(ISI_values)
#     print("Total number of runs:", total_runs )

#     for current_isi in ISI_values:
#         for seed in seedlist:


            
def run_training_commands(ISI_values = np.linspace(20, 200, 4), n_seeds = 4):
    seedlist = np.arange(88, 88+ n_seeds)
    for ISI in ISI_values:
        for seed in seedlist:
            print(f'python3 main_entrain.py --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean{ISI} --seed {seed}')
        


def run_baseline_commands(n_seeds=4):
    seedlist = np.arange(88, 88+ n_seeds)
    for seed in seedlist:
        print(f'python3 main_entrain.py --experiment nostim --run-type train --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed {seed}')
   
    
        

# def run_training_commands(ISI_values = np.linspace(20, 200, 4), n_seeds = 4):
#     seedlist = np.arange(88, 88+ n_seeds)
#     for ISI in ISI_values:
#         for seed in seedlist:
#             print(f'python3 run_entrain.py --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean{ISI} --seed {seed}.txt')

   
# def generate_train_commands_p_bridges(
#     p_values, seeds,
#     condition="plasticity",
#     outdir_root="results",
#     script_name="run.py",
# ):
#     """
#     Return a list of commands, one per (p, seed).
#     """
#     commands = []
#     for p in p_values:
#         for seed in seeds:
#             cmd = (
#                 f"python {script_name}"
#                 f" --phase train"
#                 f" --experiment specific-isi"
#                 f" --seed {seed}"
#                 f" --condition {condition}"
#                 f" --io-bridge-probability {p:.4f}"
#                 f" --outdir {outdir_root}"
#             )
#             commands.append(cmd)
#     return commands





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
