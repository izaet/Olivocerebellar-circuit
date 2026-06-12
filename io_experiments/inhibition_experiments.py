import time
import numpy as np
import sys
import os
from tqdm.auto import tqdm
import brainpy as bp
import brainpy.math as bm
import traceback
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import importlib
import itertools
import scipy as sp
import glob
import pickle
from scipy.signal.windows import dpss

sys.path.append('C:/Users/HP/PycharmProjects/Internproject 2025/cerebellum-jax-main')

import models.minimal_network as minimal_net


## ------------- General running functions ---------------####
def get_parent_dir():
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:
        return Path.cwd().parent


def run_experiment(seed, duration, dt, net_params, run_path, downsample=20,  base_net_params = {}):
    """
    :type seed: int
    :type duration: float
    :type dt: float
    :type net_params: dict
    :type downsample: int
    :type run_path: str
    :type condition: str

    """

    net = minimal_net
    current_net_params = base_net_params.copy()
    current_net_params.update(net_params)

    start_time = time.time()
    try:
        runner= net.run_simulation(
            duration=duration, dt=dt, net_params=current_net_params, seed= int(seed))
    except Exception as e:
        full_error = traceback.format_exc()
        tqdm.write(f"Error during simulation: {e}\n{full_error}")
    end_time = time.time()

    print("Runner time taken: ", end_time - start_time)

    start_time = time.time()
    data = {k: np.array(runner.mon[k][::downsample]) for k in runner.mon}
    data.update(net_params)
    data["seed"] = seed
    data["dt"] = dt * downsample

    np.savez(run_path, **data)
    end_time = time.time()
    print("Saving time taken: ", end_time - start_time)

    # print(f"Simulation time taken: {end_time - start_time:.2f} seconds")




def single_DCN(n_seeds, duration=10_000.0, gamma_CN_IO= -0.02, conditions = {'connected':{'n_projections': 4, 'p_bridges': 0 , 'n_clusters': 1} ,'unconnected': {'n_projections': 0, 'p_bridges': 0 , 'n_clusters': 1}}, spike_start = 3000.0, ):
    seeds = np.arange(88, 88 + n_seeds)


    total_runs = len(seeds)* len(conditions.keys())
    print("Total number of runs:", total_runs )
    # print(f"Estimated time = {(total_runs * 16):.2f} seconds", )

    parent_dir = get_parent_dir()
    results_subdir = os.path.join(
        parent_dir, "results", f"single_DCN_inhibition_{time.strftime('%m-%d_%H;%M;%S')}"
    )
    os.makedirs(results_subdir, exist_ok=True)

    start_time = time.time()

    run_count = 0
    for condition, param_dict in conditions.items():
        for seed in seeds:
            run_count += 1
            print(f"Running {run_count}/{total_runs}")

            net_params = {}
            net_params["CN_isi_mean"] = duration  # make sure only one spike occurs per run
            net_params["CN_start_spikes"] = spike_start
            DCN_spike_presence = (spike_start is not None)

            net_params["IO_n_projections"] = param_dict['n_projections']
            net_params["IO_n_clusters"] = param_dict['n_clusters']
            net_params["IO_bridge_probability"] = param_dict['p_bridges']

            net_params["CNIO_gamma_CN_IO"] = gamma_CN_IO

            run_path = os.path.join(results_subdir, f"seed_{seed}_condition_{condition}_DCN_spike={DCN_spike_presence}_runner.npz")
            run_experiment(seed, duration=duration, dt=0.025, net_params=net_params, downsample=20,
                           run_path=run_path)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return results_subdir

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
    # all_seeds_str = "_".join(str(int(s)) for s in all_seeds)

    DCN_spike_presence = np.any(combi_data["CN_start_spikes"][0] != None)
    Gamma_CN_IO = combi_data["CNIO_gamma_CN_IO"][0]
    file_path = os.path.join(dataset_dir, f"{experiment_name}_gamma_CN={Gamma_CN_IO}_DCN_spikes={DCN_spike_presence}_n_runs_{len(files)}_seeds_{np.min(all_seeds)}-{np.max(all_seeds)}_combined.npz")
    np.savez(file_path, **combi_data)

    return file_path


def filter_and_Hilbert(data, Fs, a = 1e-10,b = 15,  order=483, window='tukey'):
  """
  This function creates a band-pass filter, uses this filter on the data, computes the hilbert transform and calculates the instantaneous phase.
    data.shape = (n_timesteps, n_cells * n_runs)
  """

  filt = sp.signal.firwin(
      numtaps=order,
      cutoff = [a, b],
      window=window,
      fs=Fs,
      pass_zero="bandpass",
  )

  filtered = sp.signal.filtfilt(filt, [1.0], data, axis=0)
  analytic_signal = sp.signal.hilbert(filtered, axis=0)
  phases = ( (np.angle(analytic_signal)) + 2 * np.pi) % 2 * np.pi
  amplitude = np.abs(analytic_signal)

  return phases, amplitude

def detrend(dat, run_idx):
    signal = np.hstack([dat[run, :,: ] for run in run_idx])
    detrended_signal = signal - np.mean(signal, axis=0)
    return detrended_signal

def multitaper_psd(x, fs, fmax=100.0, NW=3.0, Kmax=None):

    n_times, n_trials = x.shape

    if Kmax is None:
        Kmax = max(int(2 * NW) - 1, 1)

    # Generate tapers
    tapers, eigvals = dpss(n_times, NW, Kmax, return_ratios=True)  # (K, n_times)

    # Normalize via mean subtraciton
    x = x - x.mean(axis=0, keepdims=True)

    # Frequency axis
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
    fmask = freqs <= fmax

    # Accumulate power across tapers
    S = np.zeros_like(freqs)
    for k in range(Kmax):
        tapered = x * tapers[k, :, None]  # (n_times, n_trials)
        Xf = np.fft.rfft(tapered, axis=0)  # (n_freqs, n_trials)
        Sk = (np.abs(Xf) ** 2).mean(axis=1)  # average over trials
        S += Sk

    S /= Kmax  # average across tapers

    return freqs[fmask], S[fmask], Kmax


def plot_multitaper_PSD(data_detrend, fs, fmax=60.0,
                                   NW=3.0, Kmax=None, labels=None,
                                   dB=True):

    n_conditions = data_detrend.shape[0]

    if labels is None:
        labels = [f"Condition {i+1}" for i in range(n_conditions)]

    fig, ax = plt.subplots(figsize=(6, 3))

    for i in range(n_conditions):
        x = data_detrend[i]  # shape (n_timepoints, n_trials)
        freqs, psd, kmax = multitaper_psd(x, fs, fmax=fmax, NW=NW, Kmax=Kmax)
        if dB:
            ax.plot(freqs, 10 * np.log10(psd), label=labels[i])
            ax.set_ylabel("Power (dB)")
        else:
            ax.plot(freqs, psd, label=labels[i])
            ax.set_ylabel("Power")

    ax.set_xlim(0, fmax)
    ax.set_xticks(np.arange(0, fmax, 5))
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    ax.set_title(rf"DPSS tapered PSD, $n_{{\text{{tapers}}}} = {kmax}$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def assess_phase_linearity(unwrapped_phases, time_window, time, r_sqr_thresh = 0.95):
    ntrials = unwrapped_phases.shape[1]
    phi = unwrapped_phases[time_window[0]:time_window[1],:]
    t = np.repeat(time[time_window[0]:time_window[1]].reshape(-1, 1), ntrials, axis=1)

    # Fit linear line to phases
    coef= np.array( [np.polyfit(t[:,trial], phi[:,trial], 1) for trial in np.arange(ntrials)])
    omega, phi0 = coef[:,0], coef[:,1]
    phi_predicted = omega * t + phi0
    res = phi - phi_predicted # Residual / error

    # Calculate coefficient of determination
    ss_res = np.sum(res**2, axis=0)
    ss_tot = np.sum((phi - np.mean(phi, axis=0))**2, axis=0) # Total variance

    with np.errstate(divide='ignore', invalid='ignore'):
        r_sqr = 1.0 - (ss_res / ss_tot)
        r_sqr[ss_tot == 0] = np.nan


    lin_idx = np.where(r_sqr > r_sqr_thresh)[0]

    return lin_idx

def sort_spiking_data (IO_spike_dat, run_idx, trial_idx, argsort_idx):

    IO_spike_dat = np.hstack(
        [IO_spike_dat[run, :, :] for run in run_idx])
    IO_spike_dat = IO_spike_dat[:, trial_idx]
    IO_spike_dat = IO_spike_dat[:, argsort_idx]

    return IO_spike_dat

def find_first_spike (IO_spike_dat, IO_baseline_dat, center):

    IO_baseline_dat_after = IO_spike_dat[(center + 1):, :]
    IO_spike_dat_after = IO_baseline_dat[(center + 1):, :]

    valid_trials = np.unique(np.concatenate([
        np.where( IO_baseline_dat_after.any(axis=0))[0], np.where(IO_spike_dat_after.any(axis=0))[0]]))

    baseline_first_spike_idx = np.array([np.where( IO_baseline_dat_after[:, trial])[0][0] for trial in valid_trials])
    spiking_first_spike_idx = np.array([np.where(IO_spike_dat_after[:, trial])[0][0] for trial in valid_trials])

    return  spiking_first_spike_idx, baseline_first_spike_idx, valid_trials

def fit_fourier_PRC(theta, d_phi, K_max):

    N = len(theta)

    # Set up vectors of multipliers: for each data point 2*K_max + 1 multipliers
    multi_vect = np.column_stack(
        [np.ones(N)] +
        [f(k * theta) for k in range(1, K_max + 1) for f in (np.sin, np.cos)]
    )

    # Find Fourier coefficients that fit my data
    beta, resi, rank, s = np.linalg.lstsq(multi_vect, d_phi, rcond=None)

    # (Re-)construct the PRC
    theta_grid = np.linspace(0, 2* np.pi, 500, endpoint=False)
    PCR_d_phi = calculate_PCR(theta_grid, beta, K_max)

    return theta_grid, PCR_d_phi

def calculate_PCR(theta_eval, beta, K_max):
    coef_pairs = beta[1:].reshape((-1, 2))

    y = beta[0] * np.ones(theta_eval.shape[0])
    for k, (a_k, b_k) in enumerate(coef_pairs, start=1):
        y += a_k * np.sin(k * theta_eval) + b_k * np.cos(k * theta_eval)

    return y


def get_linfit_measure(phi_signal, time, idx_measure, linear_fit_size =80 ):
    """

    :param signal: shape(n_timepoints, n_trials)
    :param idx_measure:
    :param linear_fit_size:
    :return:
    """
    ntrials = phi_signal.shape[1]
    idx1 = np.int32(np.round(idx_measure - linear_fit_size/2))
    idx2= np.int32(np.round(idx_measure + linear_fit_size/2))
    phi = np.array(phi_signal[idx1: idx2, :] )
    t = np.repeat(time[idx1: idx2].reshape(-1, 1), ntrials, axis=1)

    # Fit linear line to phases
    coef = np.array([np.polyfit(t[:, trial], phi[:, trial], 1) for trial in np.arange(ntrials)])
    omega, phi0 = coef[:, 0], coef[:, 1]
    phi_predicted = omega * t + phi0
    phi_predicted_measure = phi_predicted[np.int32(np.round(linear_fit_size/2)), :]

    trial = np.random.randint(ntrials)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
    for i,ax in enumerate((ax1, ax2, ax3, ax4, ax5)):
        ax.plot(time, phi_signal[:, trial+i])
        ax.plot(t[:,5], phi_predicted[:, trial+i])
        ax.set_xlim(1800, 3000)
    plt.show()

    return phi_predicted_measure
