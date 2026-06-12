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


#### ----------- GABA input -------------------------



#### --------- Entrainment experiments --------------
def train_half_wave ():

    return weights

def waveform_untrained():

def waveform_trained( weights):

def get_waveform_input (amplitude, period, frequency, offset):
    """

    :param amplitude: Amplitude of the wave
    :param period: Duration / length of the wave
    :param frequency: How often to generate a wave
    :param offset: Time after which to generate a wave (to set ISI)
    :return: half_wave_signal
    """

    return half_wave_signal



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

    np.savez(run_path, **data)
    end_time = time.time()
    print("Saving time taken: ", end_time - start_time)

    # print(f"Simulation time taken: {end_time - start_time:.2f} seconds")
