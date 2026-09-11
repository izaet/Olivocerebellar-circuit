import argparse
import sys
import os
import time
from pathlib import Path

from stimulus_experiments.experiment_functions import (
    run_train,
    run_baseline,
    run_test,
    get_parent_dir,
)

def build_net_params(args):
    net_params = {
        "PFPC_plasticity_on": args.PFPC_plasticity_on,

        "OU_stim_pf_on": args.OU_stim_pf_on,
        "OU_stim_io_on": args.OU_stim_io_on,

        "OU_stim_isi_mean": args.OU_stim_isi_mean,
        "OU_stim_isi_std": args.OU_stim_isi_std,
        "OU_stim_freq": args.OU_stim_freq,
        "OU_stim_start": args.OU_stim_start,
        "OU_stim_amp_io_mean": args.OU_stim_amp_io_mean,
        "OU_stim_amp_pf_mean": args.OU_stim_amp_pf_mean,
        "OU_stim_dur_io_mean": args.OU_stim_dur_io_mean,
        "OU_stim_dur_pf_mean": args.OU_stim_dur_pf_mean,

        "num_pf_bundles": args.num_pf_bundles,
        "num_pc": args.num_pc,
        "num_cn": args.num_cn,
        "num_io": args.num_io,

        "monitor_preset": args.monitor_preset,

        # Store the selected label as simulation metadata.
        "zebrin_condition": args.zebrin_condition,
    }

    # Add/overwrite the condition-dependent parameters.
    net_params.update(zebrin_presets[args.zebrin_condition])

    return net_params

# ------------------------------------------ Helpers ------------------------------------------------------

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

CLUSTER_PARENT_DIR = "/home/izet/Olivocerebellar-circuit"


def normalize_parent_dir(value):
    if value is None or value == "":
        return Path(CLUSTER_PARENT_DIR)
    value = str(value)
    if value.startswith(("c:", "C:")) or "\\" in value:
        return Path(CLUSTER_PARENT_DIR)
    return Path(value)

def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.strip().lower()

    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a Boolean value, got {value!r}."
    )

zebrin_presets = {
    "zebrin_upbound": {
        "CNIO_gamma_CN_IO": -2.0 ,
        "PCCN_gamma_PC": 0.005,
        "IO_I_OU0": -0.3 },

    "zebrin_downbound": {
         "CNIO_gamma_CN_IO": -1.8, 
         "PCCN_gamma_PC": 0.004,
         "IO_I_OU0": -0.6}   
    }


def parse_args(arg_list= None):
    parser = argparse.ArgumentParser(description= "ISI training runs")

    # ----------------- Network and stimulus parameters ---------------------

    # Stimulus
    parser.add_argument("--OU-stim-isi-mean",  type=float, default=120.0, help='Mean interval between PF-IO (ms)')
    parser.add_argument("--OU-stim-isi-std",  type=float, default=0.0, help='Standard deviation of interval between PF-IO (ms)')
    parser.add_argument("--OU-stim-freq", type=float, default=400.0, help='')
    parser.add_argument("--OU-stim-start",  type=float, default=0.0, help='')
    parser.add_argument("--OU-stim-amp-io-mean",  type=float, default=0.7, help='')
    parser.add_argument("--OU-stim-amp-pf-mean",  type=float, default=0.7, help='')
    parser.add_argument("--OU-stim-dur-io-mean",  type=float, default=50.0, help='')
    parser.add_argument("--OU-stim-dur-pf-mean",  type= float, default=50.0, help='')

    parser.add_argument("--OU-stim-io-on", type=str2bool, default = True, help= 'Turn IO stimulus on/off')
    parser.add_argument("--OU-stim-pf-on", type=str2bool, default = True, help= 'Turn IO stimulus on/off')

    # Network
    parser.add_argument("--PFPC-plasticity-on", type=str2bool, default = False, help= 'Turn on BCM rule for PF-PC synapse')
    parser.add_argument("--num-pf-bundles", type=int, default=5,
                    help="Number of PF bundles")
    parser.add_argument("--num-pc", type=int, default=100,
                    help="Number of Purkinje cells")
    parser.add_argument("--num-cn", type=int, default=40,
                    help="Number of cerebellar nuclei cells")
    parser.add_argument("--num-io", type=int, default=40,
                    help="Number of inferior olive cells")


    # Upbound vs Downbound
    parser.add_argument("--zebrin-condition", type =str, choices= (list(zebrin_presets)), default='zebrin_upbound', help = 'Network parameter preset to simulate zebrin (Aldolase-C) positive / negative zones')


    # -------------------- Runner / simulation parameters ------------------------------

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--simdur', type=float, default=10_000.0, help='(Maximum) simulation time')
    parser.add_argument("--dt", type=float, default=0.025,  help="Integration time-step (ms)")
    parser.add_argument("--downsample", type=int, default= 40)
    parser.add_argument("--monitor-preset", type=str, default="neuron_min", choices=["plasticity_full", "plasticity_min", "neuron_min", "stimulus", "neuron_spike", "stimulus_testing", "voltages"], help="Which set of monitors to use.")
    
    # Training / convergence parameters
    parser.add_argument("--max-runtime", type=int, default=500_000, help="Maximum runtime for convergence check (ms)")
    parser.add_argument("--epoch-time", type = int, default = 350, help= "Duration of epochs to check convergence over (ms)")
    parser.add_argument("--conv-thresh-m", type=float, default=0.1, help="Convergence threshold for mean weight change per epoch")
    parser.add_argument("--conv-thresh-var", type=float, default=0.2, help="Convergence threshold for variance weight change per epoch")
    parser.add_argument("--conv-chunk-thresh", type=int, default=1, help="Number of consecutive epochs that must satisfy convergence thresholds")

    
    # --------------------- Directories / naming ---------------------

    parser.add_argument(
        "--parent-dir",
        type=str,
        default=None,
        help="Base directory. If not given, uses get_parent_dir().",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag appended to result/state/figure folder names.",
    )

    parser.add_argument(
        "--pretraining-path",
        type=str,
        default="",
        help="Path to pretraining run to load in network state",
    )


    # ---- Run selection ----
    parser.add_argument(
        "--run-type",
        choices=["train", "baseline", "test"],
        required=True,
        help="Which type of run to perform.",
    )
    parser.add_argument(
        "--experiment",
        choices=["fixed-isi", "variable-isi", "nostim"],
        required=True,
        help="Stimulus protocol / task.",
    )

    return parser.parse_args(arg_list)



# ----------------------------------------- Config building functions -----------------------------------------
def build_train_config(args):
    parent_dir = normalize_parent_dir(args.parent_dir)

    tag = f"{tag}" if tag else ""

    results_dir = parent_dir / "results" / f"stim_experiments_train_{args.experiment}_mon_{args.monitor_preset}_{tag}"
    fin_state_dir = parent_dir / "states" / f"fin_states_{args.experiment}_{tag}"

    for d in (results_dir, fin_state_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_fname = (f"train_{args.experiment}_isi{args.OU_stim_isi_mean:.1f}_isi_std{args.OU_stim_isi_std:.1f}seed{args.seed}_zebrin_{args.zebrin_condition}_simdur{args.simdur}.npz")
    run_path = results_dir / run_fname

    fin_state_fname = (f"train_{args.experiment}_isi{args.OU_stim_isi_mean:.1f}_isi_std{args.OU_stim_isi_std:.1f}_seed{args.seed}__zebrin_{args.zebrin_condition}_simdur{args.simdur}_state.bp")
    fin_state_path = str(fin_state_dir / fin_state_fname)

    net_params = build_net_params(args)

    run_params = { 
        "seed": args.seed,
        "dt": args.dt,
        "downsample": args.downsample,
        "simdur": args.simdur,
        "epoch_time": args.epoch_time,
        "conv_thresh_m": args.conv_thresh_m,
        "conv_thresh_var": args.conv_thresh_var,
        "conv_chunk_thresh": args.conv_chunk_thresh,
    }

    config = {
        "net_params": net_params,
        "run_params": run_params,
        "fin_state_path": fin_state_path,
        "run_path": run_path,
    }
    return config


def build_test_config(args):
    parent_dir = normalize_parent_dir(args.parent_dir)

    tag = f"{tag}" if tag else ""

    if args.pretraining_path:
        pretraining_state_path_obj = Path(args.pretraining_path)
        pretraining_state_path = str(pretraining_state_path_obj)
    else:
        raise ValueError("Pretraining path must be provided for test runs.")

    test_results_root = parent_dir / "results" / f"stim_experiments_test_{args.experiment}_mon_{args.monitor_preset}_{tag}"
    pretraining_label = pretraining_state_path_obj.stem.removesuffix("_state")
    test_results_dir = test_results_root / pretraining_label

    test_results_dir.mkdir(parents=True, exist_ok=True)

    run_fname = f"test_{args.experiment}_{pretraining_label}_plasticity_{args.PFPC_plasticity_on}_seed{args.seed}_zebrin_{args.zebrin_condition}_simdur{args.simdur}.npz"
    run_path = test_results_dir / run_fname

    net_params = build_net_params(args)
    run_params = {
        "seed": args.seed,
        "dt": args.dt,
        "downsample": args.downsample,
        "simdur": args.simdur,
        "epoch_time": args.epoch_time,
    }

    config = {
        "net_params": net_params,
        "run_params": run_params,
        "run_path": run_path,
        "pretraining_state_path": pretraining_state_path
    }
    return config


def build_baseline_config(args):
    parent_dir = normalize_parent_dir(args.parent_dir)

    tag = f"{tag}" if tag else ""

    results_dir = parent_dir / "results" / f"stim_experiments_baseline_{args.experiment}_mon_{args.monitor_preset}_{tag}"

    results_dir.mkdir(parents=True, exist_ok=True)

    run_fname = f"baseline_{args.experiment}_plasticity_{args.PFPC_plasticity_on}_seed{args.seed}_zebrin_{args.zebrin_condition}_simdur{args.simdur}.npz"
    run_path = results_dir / run_fname

    net_params = build_net_params(args)

    run_params = {
        "seed": args.seed,
        "dt": args.dt,
        "downsample": args.downsample,
        "simdur": args.simdur,
        "epoch_time": args.epoch_time,
    }

    config = {
        "net_params": net_params,
        "run_params": run_params,
        "run_path": run_path,
    }
    return config

# ----------------------------------------- Main function -----------------------------------------
def main():
    args = parse_args()

    if args.run_type == "train":
        config = build_train_config(args)
        run_train(config)
    elif args.run_type == "baseline":
        config = build_baseline_config(args)
        run_baseline(config)
    elif args.run_type == "test":
        config = build_test_config(args)
        run_test(config)
    else:
        raise ValueError(f"Unknown run_type: {args.run_type}")
    

if __name__ == "__main__":
    main()






