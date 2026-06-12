from matplotlib.gridspec import GridSpec
import networkx as nx
from itertools import product
from collections import defaultdict
import scipy as sp
import matplotlib as mpl
from scipy.stats import circmean
from scipy.stats import circstd
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import glob
import os
import pickle
from collections import Counter
from networkx.linalg.algebraicconnectivity import algebraic_connectivity

sys.path.append('C:/Users/HP/PycharmProjects/Internproject 2025/cerebellum-jax-main')


#### -------------- General data organization functions  -------------------- ####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent

def nested_dict():
        return defaultdict(nested_dict)

def combine_data(results_dir,pattern= "*.npz", conditions = ["plasticity", "static"], add_graph_io = True):
    """
    Combines all runs within an experiment. Assumes number of connections between populations and the size of populations is the same for all runs in the experiment.
    For the IO, changes in the number of connections within the population are handles by making a NetworkX graph object of the IO Network (add_graph_io = True).
    Only disable add_graph_io, if the IO topography is the same across all runs.

    :type add_graph_io: object
    :param results_subdir:
    :param variable:
    :return:
    """

    parent_dir = str(get_parent_dir())
    dataset_dir = os.path.join(parent_dir, "data", "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    network_dir = os.path.join(parent_dir, "data", "networks")
    os.makedirs(network_dir, exist_ok=True)

    base = os.path.basename(results_dir)
    base_split = base.rsplit("_", 2)
    experiment_name = base_split[0]
    experiment_time = base_split[1] + "_" + base_split[2]

    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No npz files found in {results_dir} with pattern {pattern}")

    static_runs = []
    plasticity_runs = []

    for f in files:
        if "static" in f:
            with np.load(f, allow_pickle=True) as npz:
                run_data = {k: npz[k] for k in npz.files}  # convert NpzFile -> dict
            run_data["source_file"] = np.array(f)  # optional: remember where it came from
            static_runs.append(run_data)

        elif "plasticity" in f:
            with np.load(f, allow_pickle=True) as npz:
                run_data = {k: npz[k] for k in npz.files}
            run_data["source_file"] = np.array(f)
            plasticity_runs.append(run_data)

    removable_keys = np.array(['io_src', 'io_tgt'])
    network_paths = defaultdict(lambda: defaultdict(list))

    if add_graph_io:
        for run_id in np.arange(len(static_runs)):
            # Get NetworkX graph of IO network for each run
            network_file_path = os.path.join(network_dir,
                                             f"{experiment_name}_run_{run_id}_seed_{static_runs[run_id]['seed']}_static_{experiment_time}.gpickle")
            get_network_graph(network_file_path, static_runs[run_id])
            static_runs[run_id]['network_graph'] = np.array(network_file_path)

            network_file_path = os.path.join(network_dir,  f"{experiment_name}_run_{run_id}_seed_{plasticity_runs[run_id]['seed']}_plasticity_{experiment_time}.gpickle")
            get_network_graph(network_file_path, plasticity_runs[run_id])
            plasticity_runs[run_id]['network_graph'] = np.array(network_file_path)

            # Remove parameters that are already stored in NetworkX graph
            for key in removable_keys:
                static_runs[run_id].pop(key)
                plasticity_runs[run_id].pop(key)

    combi_data = {"static": {}, "plasticity": {}}
    for key in static_runs[0].keys():
        arr = static_runs[0][key]
        shape = (len(static_runs),) + (() if np.isscalar(arr) or np.ndim(arr) == 0 else arr.shape)
        combi_data['static'][key] = np.zeros(shape, dtype=arr.dtype)

        for run_id in np.arange(len(static_runs)):
            combi_data['static'][key][run_id] = static_runs[run_id][key]

    for key in plasticity_runs[0].keys():
        arr = plasticity_runs[0][key]
        shape = (len(plasticity_runs),) + (() if np.isscalar(arr) or np.ndim(arr) == 0 else arr.shape)
        combi_data['plasticity'][key] = np.zeros(shape, dtype=arr.dtype)

        for run_id in np.arange(len(plasticity_runs)):
            combi_data['plasticity'][key][run_id] = plasticity_runs[run_id][key]

    all_seeds = np.unique(combi_data['plasticity']['seed'])
    all_seeds_str = "_".join(str(int(s)) for s in all_seeds)
    file_path = os.path.join(dataset_dir, f"{experiment_name}_n_runs_{len(files)}_seeds_{all_seeds_str}_combined.npz")
    np.savez(file_path, **combi_data)

    return file_path

def get_network_graph(network_file_path, data):
    IO_map = nx.Graph()

    for i in np.arange(data["n_neurons"]):  # Add nodes and cluster identity
        IO_map.add_node(i, cluster=data["io_cluster_ids"][i].astype(int))
    for src, tgt in zip(data["io_src"], data["io_tgt"]):  # Add edges
        IO_map.add_edge(src.astype(int), tgt.astype(int))

    with open(network_file_path, "wb") as f:
        pickle.dump(IO_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    return IO_map




def group_runs_by_variable(parameter, data, cond = 'plasticity'):
    """

    :param parameter: string
    :param data: dictionairy
    :param cond: string
    :return:
    """

    parameter_list = np.unique(data[cond][parameter])
    parameter_idx = {f'{n_values}': np.where(data[cond][parameter] == n_values)[0] for n_values in parameter_list} # Keys = all possibles values for variable, items = the indices of the runs with those variables
    return parameter_idx



def calc_network_metrics(network_file_paths):
    """

    :param network_file_path: n x 1 array with paths to where networkX graph is stored
    :param cond:
    :return:

    clustering_coef; Clustering coefficient
    ratio_edge_d; Proportion of local (within cluster) edge density to density of global / bridge edges
    fiedler_eig; Laplacian eigenvalue of the network
    n_components; The number of connected components in the network


    """
    if network_file_paths is None:
        raise ValueError("Variable network_file_path must be specified")

    clustering_coef = np.zeros(network_file_paths.shape)
    av_ratio_edge_d = np.zeros(network_file_paths.shape)
    fiedler_eig = np.zeros(network_file_paths.shape)
    n_components = np.zeros(network_file_paths.shape)

    for i in np.arange(len(network_file_paths)):
        with open(network_file_paths[i], "rb") as f:
            IO_map = pickle.load(f)

        clustering_coef[i]= nx.average_clustering(IO_map)
        fiedler_eig[i]= algebraic_connectivity(IO_map)
        n_components[i] =  nx.number_connected_components(IO_map)

        # Calculate ratio of inter-intra edge density
        clusters = nx.get_node_attributes(IO_map, "cluster")
        cluster_sizes = Counter(clusters.values())
        N_nodes = IO_map.number_of_nodes()
        intra_edges, inter_edges = seperate_edges(IO_map, clusters)

        max_edges = N_nodes * (N_nodes - 1) // 2
        max_intra = sum(n * (n - 1) // 2 for n in cluster_sizes.values())
        max_inter = max_edges - max_intra

        d_intra = len(intra_edges) / max_intra if max_intra > 0 else float('nan')
        d_inter = len(inter_edges) / max_inter if max_inter > 0 else float('nan')


        av_ratio_edge_d[i] = d_inter /  d_intra if  d_intra > 0 else float('nan')  # global / local

    network_metrics= {'clustering_coef': clustering_coef, 'av_ratio_edge_d': av_ratio_edge_d, 'fiedler_eig': fiedler_eig, 'n_components': n_components}

    return network_metrics


def seperate_edges (IO_map, clusters):

    intra_edges = []
    inter_edges = []
    for u, v in IO_map.edges():  # Separate bridges from regular connections
        u = int(u)
        v = int(v)
        if clusters[u] == clusters[v]:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    return intra_edges, inter_edges

# --------------- Periodic data analysis ------------------------------


def Kuramoto_synchrony_analysis(data, id_parameter):

    """

    :param data:
    :param id_parameter:
    :return:

     R_K_sync_global =
    R_K_sync_within =
    R_K_sync_between =

    # Mean phase overall and within each cluster
    global_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    within_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    between_mean_phase = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))  # Should be same as the global mean phase

    # Synchrony to OU on multiple scales
    OU_sync_global = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_sync_within = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_sync_between = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    OU_global_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_within_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_between_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    """



    idx_runs_by_param = group_runs_by_variable(id_parameter, data)


    # Calculate phase for all io_signals + their Ohrnstein Uhlenbeck input:
    Fs = 1 / (data['dt'] / 1000)

    io_phases_all = {}
    io_amplitudes_all = {}
    OU_phases_all = {}
    OU_amplitudes_all = {}
    d_io_OU_phase = {}

    for cond in ("static", "plasticity"):
        # Calculate phases
        io_phases_all[cond], io_amplitudes_all[cond] = filter_and_Hilbert(data[cond]['io.V_soma'], Fs)
        OU_phases_all[cond], OU_amplitudes_all[cond] = filter_and_Hilbert(data[cond]['io.V_soma'], Fs)

        # Calculate phase difference between IO signal and its I_OU
        d_io_OU_phase[cond] = io_phases_all[cond] - OU_phases_all[cond]

    # Weighted and unweighted Kuramoto synchrony measures
    R_K_sync_global = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    R_K_sync_within = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    R_K_sync_between = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Mean phase overall and within each cluster
    global_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    within_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    between_mean_phase = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))  # Should be same as the global mean phase

    # Synchrony to OU on multiple scales
    OU_sync_global = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_sync_within = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_sync_between = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    OU_global_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_within_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    OU_between_mean_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for value, run_ids in idx_runs_by_param.items():
        for cond in ("static", "plasticity"):
            for run_id in run_ids:

                phase_signals = io_phases_all[cond][run_id]
                amplitude_signals = io_amplitudes_all[cond][run_id]
                phase_differences = d_io_OU_phase[cond][run_id]

                for b, weight_use in enumerate(['unweighted', 'weighted']):
                    # ------------(Un)weighted Kuramoto synchrony ------------
                    R_K_sync_global[cond][weight_use][f'{value}'], global_mean_phase[cond][weight_use][
                        f'{value}'] = Kuramoto_order(phase_signals, weighted=(b > 0), weights=amplitude_signals)

                    R_K_sync_within[cond][weight_use][f'{value}'], within_mean_phase[cond][weight_use][
                        f'{value}'], mean_amplitude_clusters = within_cluster_Kuramoto(phase_signals,
                                                                                           data[cond][
                                                                                               'io_cluster_ids'][
                                                                                               run_id],
                                                                                           weighted=(b > 0),
                                                                                           amplitudes=amplitude_signals)

                    within_mean_phase_array = np.stack(
                        [within_mean_phase[cond][weight_use][f'{value}'][clust] for clust in
                         within_mean_phase[cond][weight_use][f'{value}']], axis=1)
                    mean_amplitude_clusters_array = np.stack(
                        [mean_amplitude_clusters[clust] for clust in mean_amplitude_clusters], axis=1)

                    R_K_sync_between[cond][weight_use][f'{value}'], between_mean_phase[cond][weight_use][
                        f'{value}'] = Kuramoto_order(within_mean_phase_array, weighted=(b > 0),
                                                         weights=mean_amplitude_clusters_array)

                    # -------------- Kuramoto synchrony of IO-I_OU phase differences -------
                    # If all the phase differences are tightly clustered and the Kuramotor order is high, then all the IO's are strongly driven to their OU at a phase locked values. If the phase differences are spread out with a low Kuramoto order, then the IO's are showing behaviour independent of their OU. Do different clusters maintain a similar phase relationship to their OU input, or do they have distinct preferred IO–OU phase lags?

                    OU_sync_global[cond][weight_use][f'{value}'], OU_global_mean_phase[cond][weight_use][
                        f'{value}'] = Kuramoto_order(phase_differences, weighted=(b > 0), weights=amplitude_signals)

                    OU_sync_within[cond][weight_use][f'{value}'], OU_within_mean_phase[cond][weight_use][
                        f'{value}'], OU_mean_amplitude = within_cluster_Kuramoto(phase_signals,
                                                                                     data[cond]['io_cluster_ids'][
                                                                                         run_id], weighted=(b > 0),
                                                                                     amplitudes=amplitude_signals)

                    OU_within_mean_phase_array = np.stack(
                        [OU_within_mean_phase[cond][weight_use][f'{value}'][clust] for clust in
                         OU_within_mean_phase[cond][weight_use][f'{value}']], axis=1)
                    OU_mean_amplitude_array = np.stack([OU_mean_amplitude[clust] for clust in OU_mean_amplitude],
                                                       axis=1)

                    OU_sync_between[cond][weight_use][f'{value}'], OU_between_mean_phase[cond][weight_use][
                        f'{value}'] = Kuramoto_order(OU_within_mean_phase_array, weighted=(b > 0),
                                                         weights=OU_mean_amplitude_array)

    return {
        "R_K_sync_global": R_K_sync_global,
        "R_K_sync_within": R_K_sync_within,
        "R_K_sync_between": R_K_sync_between,
        "global_mean_phase": global_mean_phase,
        "within_mean_phase": within_mean_phase,
        "between_mean_phase": between_mean_phase,
        "OU_sync_global": OU_sync_global,
        "OU_sync_within": OU_sync_within,
        "OU_sync_between": OU_sync_between,
        "OU_global_mean_phase": OU_global_mean_phase,
        "OU_within_mean_phase": OU_within_mean_phase,
        "OU_between_mean_phase": OU_between_mean_phase,
    }


def filter_and_Hilbert(data, Fs, a = 1e-10,b = 15,  order=483, window='tukey', ):
  """
  This function creates a band-pass filter, uses this filter on the data, computes the hilbert transform and calculates the instantaneous phase.
    data.shape = (n_runs, n_timesteps, n_cells)
  """

  filt = sp.signal.firwin(
      483,
      [a, b],
      window=window,
      fs=Fs,
      pass_zero="bandpass",
  )

  filtered = sp.signal.lfilter(filt, [1.0], data, axis=1)
  analytic_signal = sp.signal.hilbert(filtered, axis=1)
  phases = np.angle(analytic_signal)
  amplitude = np.abs(analytic_signal)

  return phases, amplitude

def Kuramoto_order(phases, weights = None):
    """

    :param phases: (n_timepoints, n_oscillators) phase of a signal over time
    :param weights: (n_timepoints, n_oscillators) weight for each oscillator at every timepoint
    :return: R = Kuramoto order, psi = mean phase
    """
    complex_vectors = np.exp(1j * phases)

    if weights:
        complex_vector_mean = np.sum(weights * complex_vectors, axis=1) / np.sum(weights, axis=1)
    else:
        complex_vector_mean = np.mean(complex_vectors, axis=1)


    R = np.abs(complex_vector_mean)
    psi = np.angle(complex_vector_mean)

    return R, psi

def within_cluster_Kuramoto(phase_data,cluster_ids, amplitudes = None):
    """

    :param phase_data: (n_timepoints, n_oscillators) phase of a signal over time
    n_oscillators = n_io cells
    :param cluster_ids:

    :return:
    """
    clusters = np.int64(np.unique(cluster_ids))
    cluster_idx = [np.where( cluster_ids == cluster)[0] for cluster in
                    clusters]

    n_clusters = len(clusters)
    n_timepoints = phase_data.shape[0]
    R_cluster_within = np.zeros((n_timepoints, n_clusters))
    psi_cluster_within = np.zeros((n_timepoints, n_clusters))
    mu_amplitude_cluster = np.zeros((n_timepoints, n_clusters))

    for cluster, idx in zip(clusters, cluster_idx):
        phases_cluster = phase_data[:, idx]

        if amplitudes:
            amplitude_cluster = amplitudes[:, idx]
            mu_amplitude_cluster[:,cluster] = np.mean(amplitude_cluster, axis=1)
        else:
            amplitude_cluster = np.nan
            mu_amplitude_cluster[:,cluster] = np.nan

        R_cluster, psi_cluster = Kuramoto_order(phases_cluster, weights = amplitude_cluster)

        R_cluster_within[:,cluster] = R_cluster
        psi_cluster_within[:,cluster] = psi_cluster

    return R_cluster_within, psi_cluster_within, mu_amplitude_cluster

def filter_cspk_list(cpsk_list, dt, t_before, t_after, merge_window, overlap= False):
    """

    :param cpsk_list: n x 1 np.array
    :param dt:
    :param t_before:
    :param t_after:
    :param merge_window:
    :param overlap:
    :return:
    """

    # Merge duplicate detections of the same cspk
    merge_samples = int(round(merge_window / dt))

    if overlap:
        return merged_cspk_list

    else:
        # Optionally filter out cspks with overlapping windows.
        before_samples= int(round(t_before / dt))
        after_samples = int(round(t_after / dt ))

        start_idx = merged_cspk_list - before_samples
        end_idx = merged_cspk_list - after_samples



        return filtered_cspk_list


