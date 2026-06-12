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

sys.path.append('C:/Users/HP/PycharmProjects/Internproject 2025/cerebellum-jax-main')

# Figure 2DEF
def figure_2DEF(seed, mon, save_dir, save= False,  threshold = 2, xrange = [30_000, 50_000]):
  
    ts_idx = [max(np.where(mon['ts'] <= xrange[0])[0]), min(np.where(mon['ts'] >= xrange[1])[0])]
    sl = slice(ts_idx[0], ts_idx[1] + 1)
    PC_cspks_thr = (np.where(mon["pc.cspk"][ts_idx[0]:ts_idx[1]].sum(axis=0) >= threshold)[0]) # Find PC cells with lots of cspks
    if not len(PC_cspks_thr[1:3]):
        print('no cs')
    for PC in PC_cspks_thr[1:3]:
        cons_thr_idx = np.where(np.isin(mon["pf_pc_post"], PC))[0] # Find which conn indexes these PC cells are part of
        cons_thr_idx=  np.random.choice(cons_thr_idx, size=3, replace=False)  # Only select 3 connections to plot

        for connection in cons_thr_idx:

            norm = mon["pfpc_weights"][0, connection]

            xlim = [x / 1000 for x in xrange] 
            ts_idx = [max(np.where(mon['ts'] <= xlim[0])[0]), min(np.where(mon['ts'] >= xlim[1])[0])]

            fig = plt.figure(figsize=(8, 12))
            gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1])

            fig.suptitle(f"_seed_{seed}_PCnum_{PC}_con_num{connection}")

            # Total weight
            ax_w = fig.add_subplot(gs[0, :])
            ax_w.plot(mon['ts'][sl] / 1000, (mon["pfpc_weights"][sl, connection] - norm ) / norm)
            ax_w.set_title(r"$w$")
            ax_w.set_xlim(xlim)
            ax_w.set_xlabel('time (s)')


            # CSpk weights; delta and total
            ax10 = fig.add_subplot(gs[1, 0])
            ax11 = fig.add_subplot(gs[1, 1])

            ax10.plot(mon['ts'][sl] / 1000, (mon["pfpc_dw_cspk"][sl, connection] - norm) / norm)
            ax10.set_title(r"$\Delta_{w CSpk}$")
            ax10.set_xlim(xlim)
            ax10.set_xlabel('time (s)')


            ax11.plot(mon['ts'][sl] / 1000, (mon["pfpc_w_cspk"][sl, connection] - norm) / norm)
            ax11.set_title(r"$w_{CSpk}$")
            ax11.set_xlim(xlim)
            ax11.set_xlabel('time (s)')

            # BCM weights ; delta and total
            ax20 = fig.add_subplot(gs[2, 0])
            ax21 = fig.add_subplot(gs[2, 1])

            ax20.plot(mon['ts'][sl] / 1000, (mon["pfpc_dw_BCM"][sl, connection] - norm) / norm)
            ax20.set_title(r"$\Delta_{w BCM}$")
            ax20.set_xlim(xlim)
            ax20.set_xlabel('time (s)')

            ax21.plot(mon['ts'][sl] / 1000, (mon["pfpc_w_BCM"][sl, connection] - norm) / norm)
            ax21.set_title(r"$w_{BCM}$")
            ax21.set_xlim(xlim)
            ax21.set_xlabel('time (s)')

            # PF firing
            ax_pf = fig.add_subplot(gs[3, :])
            current_PF_index = mon[ "pf_pc_pre"][connection]
            ax_pf.plot(mon['ts'][sl] / 1000, mon["pf.rho"][sl, current_PF_index])
            ax_pf.set_title(r"$\rho_{\mathrm{PF}}$")
            ax_pf.set_xlim(xlim)
            ax_pf.set_xlabel('time (s)')

            # PC firing
            ax_pc = fig.add_subplot(gs[4, :])
            ax_pc.plot(mon['ts'][sl] / 1000, mon["pc.rho"][sl, PC])
            ax_pc.set_title(r"$\rho_{\mathrm{PC}}$")
            ax_pc.set_xlim(xlim)
            ax_pc.set_xlabel('time (s)')

            # Sliding threshold
            ax_thresh = fig.add_subplot(gs[5, :])
            ax_thresh.plot(mon['ts'][sl] / 1000, mon["pfpc_theta_M"][sl, PC])
            ax_thresh.set_title(r"$\theta_{\mathrm{M}}$")
            ax_thresh.set_xlim(xlim)
            ax_thresh.set_xlabel('time (s)')


            plt.tight_layout()
            plt.show

            if save:

                fig.savefig(os.path.join(save_dir, f"BCM_related_changes_{seed}_PCnum_{PC}_con_num{connection}.png"))

