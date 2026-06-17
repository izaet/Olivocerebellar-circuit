import argparse
import sys
import os
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from stimulus_experiments.entrainment_experiments import (
    run_train,
    run_test,
    get_parent_dir,
)



def parse_args():
    parser = argparse.ArgumentParser(description= "ISI training runs")

    # ----------------- Network and stimulus parameters ---------------------

    # Stimulus
    parser.add_argument("--OU-stim-isi-mean",  type=float, default=120.0, help='Mean interval between PF-IO (ms)')
    parser.add_argument("--OU-stim-freq", type=float, default=700.0, help='')
    parser.add_argument("--OU-stim-start",  type=float, default=200.0, help='')
    parser.add_argument("--OU-stim-amp-io-mean",  type=float, default=1.4, help='')
    parser.add_argument("--OU-stim-amp-pf-mean",  type=float, default=1.4, help='')
    parser.add_argument("--OU-stim-dur-io-mean",  type=float, default=250.0, help='')
    parser.add_argument("--OU-stim-dur-pf-mean",  type= float, default=250.0, help='')

    parser.add_argument("--OU-stim-io-on", type=bool, default = True, help= 'Turn IO stimulus on/off')
    parser.add_argument("--OU-stim-pf-on", type=bool, default = True, help= 'Turn IO stimulus on/off')

    # Network
    parser.add_argument("--PFPC_plasticity-on", type=bool, default = False, help= 'Turn on BCM rule for PF-PC synapse')
    parser.add_argument("--num-pf-bundles", type=int, default=5,
                    help="Number of PF bundles")
    parser.add_argument("--num-pc", type=int, default=100,
                    help="Number of Purkinje cells")
    parser.add_argument("--num-cn", type=int, default=40,
                    help="Number of cerebellar nuclei cells")
    parser.add_argument("--num-io", type=int, default=40,
                    help="Number of inferior olive cells")


    # -------------------- Run parameters ------------------------------
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--simdur', type=float, default=10_000.0, help='(Maximum) simulation time')
    parser.add_argument("--dt", type=float, default=0.025,  help="Integration time-step (ms)")
    parser.add_argument("--downsample", type=int, default= 40)
    parser.add_argument("--epoch-time", type = int, default = 1000, help= "Duration of epochs to check convergence over (ms)")


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
        "--timestamp",
        type=str,
        default=None,
        help="Optional fixed timestamp used for deterministic output folder names.",
    )

    parser.add_argument(
    "--pretraining-tag",
    type=str,
    default="",
    help="Tag of pretraining run to load in network state",
    )


    # ---- Run selection ----
    parser.add_argument(
        "--run-type",
        choices=["train", "test"],
        required=True,
        help="Which type of run to perform.",
    )
    parser.add_argument(
        "--experiment",
        choices=["specific-isi", "random-isi", "nostim"],
        required=True,
        help="Stimulus protocol / task.",
    )

    return parser.parse_args()

def build_train_config(args):
    if args.parent_dir is None:
        parent_dir = Path(get_parent_dir())
    else:
        parent_dir = Path(args.parent_dir)

    timestamp = args.timestamp if args.timestamp else time.strftime('%m-%d_%H;%M;%S')
    tag = f"_{args.tag}" if args.tag else ""

    results_dir = parent_dir / "results" / f"stim_experiments_{args.experiment}{tag}_{timestamp}"
    snapshot_dir = parent_dir / "states" / f"states_{args.experiment}_isi{args.OU_stim_isi_mean:.1f}_seed{args.seed}"
    figures_dir = parent_dir / "figures" / f"figs_{args.experiment}{tag}"

    for d in (results_dir, snapshot_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_fname = (f"{args.run_type}_{args.experiment}"f"_isi{args.OU_stim_isi_mean:.1f}_seed{args.seed}_simdur{args.simdur}.npz")
    run_path = results_dir / run_fname

    net_params = {
        "PFPC_plasticity_on": args.PFPC_plasticity_on, 
        "OU_stim_pf_on": args.OU_stim_pf_on,
        "OU_stim_io_on": args.OU_stim_io_on,

        "OU_stim_isi_mean": args.OU_stim_isi_mean,
        "OU_stim_freq": args.OU_stim_freq,
        "OU_stim_start": args.OU_stim_start,
        "OU_stim_amp_io_mean": args.OU_stim_amp_io_mean,
        "OU_stim_amp_pf_mean": args.OU_stim_amp_pf_mean,
        "OU_stim_dur_io_mean": args.OU_stim_dur_io_mean,
        "OU_stim_dur_pf_mean": args.OU_stim_dur_pf_mean,

        

    }

    run_params = { 
        "seed": args.seed,
        "dt": args.dt,
        "downsample": args.downsample,
        "simdur": args.simdur,
        "epoch_time": args.epoch_time
    }

    config = {
        "net_params": net_params,
        "run_params": run_params,
       
        "snapshot_dir": snapshot_dir,
        "run_path": run_path,
        "figures_dir": figures_dir,

    }
    return config


def build_test_config(args):
    if args.parent_dir is None:
        parent_dir = Path(get_parent_dir())
    else:
        parent_dir = Path(args.parent_dir)  

    timestamp = args.timestamp if args.timestamp else time.strftime('%m-%d_%H;%M;%S')
    tag = f"_{args.tag}" if args.tag else ""

   
    results_dir = parent_dir / "results" / f"stim_experiments_{args.experiment}{tag}"
    snapshot_dir = parent_dir / "states" / f"states_{args.experiment}{tag}"
    figures_dir = parent_dir / "figures" / f"figs_{args.experiment}{tag}"

    pretrain_snapshot_dir = parent_dir / "states" / f"states_{args.pretraining_tag}"
    run_fname = f"test_{args.experiment}_seed{args.seed}.npz"
    run_path = results_dir / run_fname

    net_params = {
        "PFPC_plasticity_on": args.PFPC_plasticity_on, 
        "OU_stim_pf_on": args.OU_stim_pf_on,
        "OU_stim_io_on": args.OU_stim_io_on,

        "OU_stim_isi_mean": args.OU_stim_isi_mean,
        "OU_stim_freq": args.OU_stim_freq,
        "OU_stim_start": args.OU_stim_start,
        "OU_stim_amp_io_mean": args.OU_stim_amp_io_mean,
        "OU_stim_amp_pf_mean": args.OU_stim_amp_pf_mean,
        "OU_stim_dur_io_mean": args.OU_stim_dur_io_mean,
        "OU_stim_dur_pf_mean": args.OU_stim_dur_pf_mean,

        

    }

    run_params = { 
        "seed": args.seed,
        "dt": args.dt,
        "downsample": args.downsample,
        "simdur": args.simdur,
        "epoch_time": args.epoch_time
    }

    config = {
        "net_params": net_params,
        "run_params": run_params,
       
        "snapshot_dir": snapshot_dir,
        "run_path": run_path,
        "figures_dir": figures_dir,

    }
    return config

def main():
    args = parse_args()

    if args.run_type == "train":
        config= build_train_config(args)
        run_train(config)

        
    elif args.run_type == 'test':
        config= build_test_config(args)
        # run_test(config)
    else:
        raise ValueError(f"Unknown run_type: {args.run_type}")
    

if __name__ == "__main__":
    main()






