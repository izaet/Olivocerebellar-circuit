import sys

sys.path.append('C:/Users/HP/ModellingProjects/Olivocerebellar-circuit')

import warnings
import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax
import time
import os
import json
import copy

from models.network_dyn import CerebellarNetwork


def init_net_and_runner(net_params=None, dt=0.025 , seed=88, jit=True):
    np.random.seed(seed)
    bm.random.seed(seed)

     # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}

    net = CerebellarNetwork(**net_params)


    # --- Monitors Configuration --- #c
    monitors = {
        # Neuron monitors
        "pf.I_OU": net.pf.I_OU,
        "pf.I_stim": net.pf.I_stim,
        "pf.rho": net.pf.rho,

        "pc.V": net.pc.V,
        "pc.spike": net.pc.spike,
        "pc.cspk": net.pc.cspk,
        "pc.w": net.pc.w,
        "pc.input": net.pc.input,
        "pc.rho": net.pc.rho,

        "cn.V": net.cn.V,
        "cn.spike": net.cn.spike,
        "cn.I_PC": net.cn.I_PC,

        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.input": net.io.neurons.input,
        "io.I_OU": net.io.neurons.I_OU,
        "io.I_stim": net.io.neurons.I_stim,
        "io.spike": net.io.neurons.spike,
        
        # Plasticity monitors
        "pfpc_weights": net.pf_to_pc_BCM.weights_per_conn,
        "pfpc_w_cspk" : net.pf_to_pc_BCM.w_cspk,
        "pfpc_w_BCM" : net.pf_to_pc_BCM.w_BCM,
        "pfpc_theta_M" : net.pf_to_pc_BCM.theta_M,
        "pfpc_dw_cspk": net.pf_to_pc_BCM.dw_cspk,
        "pfpc_dw_BCM": net.pf_to_pc_BCM.dw_BCM,


        # Stimulus monitors
        "stim.isi": net.stim.current_isi,
        "stim.M_io": net.stim.M_io,
        "stim.M_pf": net.stim.M_pf,

    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt, jit =jit, progress_bar= False)
 
    if jit:
        runner._fun_predict = bm.jit(runner._fun_predict)

        return net, runner
    

    

    
# def snapshot_net(config, net, extra_meta = None):
#     state = {}
   

#     # Copy state (includes bm.Variables)
#     for name, var in net.vars().items():
#         state[name] = bm.as_numpy(var.value)
   
#     meta = copy.deepcopy(config)
#     if extra_meta:
#         meta.update(extra_meta)
    

#     return state, meta

def restore_state(net, state):

    for name, var in net.vars().items():
        if name in state:
            var.value = state[name]

    return net


    
def run_until_convergence(net, runner, downsample, max_runtime = 500_000, epoch =  1000, conv_thresh= 1e-4, chunk_thresh = 3):
    net.pf_to_pc_BCM.plasticity_on.value= bm.asarray(True)


    runtime= 0.0
    w_previous = net.pf_to_pc_BCM.weights_per_conn.value

    stable_count = 0

    mon_hist = {}

    while runtime < max_runtime:
        
        runner.run(epoch)
        runtime += epoch

        # Append runners for each simulation epoch
        for k  in runner.mon:
            if k not in mon_hist:
                mon_hist[k] = [np.array(runner.mon[k][::downsample])]
            else:
                mon_hist[k].append(np.array(runner.mon[k][::downsample])) 

        # Check for convergence
        w_current = net.pf_to_pc_BCM.weights_per_conn.value
        d_w_chunk_max = np.max(np.abs(w_current - w_previous))

        # mean_w_current = np.mean(runner.mon['pf_to_pc_BCM.weights_per_conn'], axis 
        # d_w_chunk_max = np.max(np.abs(mean_w_current - mean_w_previous))

        if d_w_chunk_max < conv_thresh:
            stable_count+= 1
        else:
            stable_count = 0
        
        if stable_count > chunk_thresh:
            break
        
        w_previous = w_current

    # Combine all chunks into one runner
    full_mon = {k: np.concatenate (v, axis = 0) for k, v in mon_hist.items()}
            
        
    return net, runner, full_mon, d_w_chunk_max, runtime

    




def init_and_run(duration=1000.0, dt=0.025, net_params=None, seed=42, jit=True):
    np.random.seed(seed)
    bm.random.seed(seed)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}
    # Silence warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = CerebellarNetwork(**net_params)

    # --- Params to return ------- #
    connections_idx = {"pf_pc_pre": net.pf_to_pc_BCM.pre_idx,
                   "pf_pc_post": net.pf_to_pc_BCM.post_idx,
                   "io_pc_pre": net.io_to_pc.io_source_indices,
                   "io_pc_post": net.io_to_pc.pc_target_indptr}

    io_topography_params = {"n_bridges": net.io.n_bridges,
                            "io_src": np.array(net.io.neurons.gj_src),
                            "io_tgt": np.array(net.io.neurons.gj_tgt),
                            "io_cluster_ids": net.io.cluster_ids,
                            "n_neurons": net.num_io}

    # --- Monitors Configuration --- #
    monitors = {
        # Neuron monitors
        "pf.I_OU": net.pf.I_OU,
        "pf.I_stim": net.pf.I_stim,
        "pf.rho": net.pf.rho,
        "pc.V": net.pc.V,
        "pc.spike": net.pc.spike,
        "pc.cspk": net.pc.cspk,
        "pc.w": net.pc.w,
        "pc.input": net.pc.input,
        "pc.rho": net.pc.rho,
        "cn.V": net.cn.V,
        "cn.spike": net.cn.spike,
        "cn.I_PC": net.cn.I_PC,
        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.input": net.io.neurons.input,
        "io.I_OU": net.io.neurons.I_OU,
        "io.I_stim": net.io.neurons.I_stim,
        "io.spike": net.io.neurons.spike,
        
        # Plasticity monitors
        "pfpc_weights": net.pf_to_pc_BCM.weights_per_conn,
        "pfpc_w_cspk" : net.pf_to_pc_BCM.w_cspk,
        "pfpc_w_BCM" : net.pf_to_pc_BCM.w_BCM,
        "pfpc_theta_M" : net.pf_to_pc_BCM.theta_M,
        "pfpc_dw_cspk": net.pf_to_pc_BCM.dw_cspk,
        "pfpc_dw_BCM": net.pf_to_pc_BCM.dw_BCM,


        # Stimulus monitors
        "stim.isi": net.stim.current_isi,
        "stim.M_io": net.stim.M_io,
        "stim.M_pf": net.stim.M_pf,

    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt, jit =jit, progress_bar=True)
    runner.progress_bar = False
    if jit:
        runner._fun_predict = bm.jit(runner._fun_predict)
    runner.run(duration)



    return runner, io_topography_params, connections_idx