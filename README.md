# Olivocerebellar circuit model

This repository contains a simulation of a cerebellar circuit implemented with BrainPy. The main workflow is organized around the stimulus experiment scripts in the `stimulus_experiments/` folder, with supporting model code under `models/`.

## Repository layout

- `models/` – network definition, cell models, monitors, and run helpers
- `stimulus_experiments/` – experiment entry points and command generation utilities
- `results/` – saved simulation outputs in `.npz` format
- `states/` – saved network states for restart / continuation runs
- `figures/` – generated plots and figures

## Requirements

This project depends on Python packages such as:

- `brainpy`
- `jax`
- `numpy`
- `matplotlib`

Make sure you are using the appropriate conda environment before running simulations.

## Running the model

### Option 1: Run directly from the terminal

From the repository root, you can launch a training run with the main entry point:

```bash
python stimulus_experiments/main_entrain.py \
  --seed 0 \
  --run-type train \
  --experiment nostim \
  --simdur 13000 \
  --monitor-preset plasticity_min \
  --tag state_saving_test
```

This will create output files in:

- `results/`
- `states/`
- `figures/`

### Option 2: Run from Python using the training config

You can also run the simulation programmatically by building a config and calling the training function:

```python
from stimulus_experiments.main_entrain import parse_args, build_train_config
from stimulus_experiments.entrainment_experiments import run_train

argv = [
    "--seed", "0",
    "--run-type", "train",
    "--experiment", "nostim",
    "--simdur", "13000",
    "--monitor-preset", "plasticity_min",
    "--tag", "state_saving_test"
]

args = parse_args(argv)
config = build_train_config(args)
net, data, state = run_train(config)
```

This is useful when you want to inspect the returned network object, saved data, or state directly inside Python.

### Option 3: Generate terminal commands with the command generator

The helper functions in `stimulus_experiments/entrainment_experiments.py` can generate runnable command lists for batch or parameter sweeps.

```python
from stimulus_experiments.entrainment_experiments import train_commands

jobs = train_commands(
    parent_dir=".",
    monitor="plasticity_min",
    n_seeds=1,
    simdur=13000,
    experiment="nostim",
    tag="state_saving_test",
)

for job in jobs:
    print(job["command"])
```

Each generated job includes a command string and the associated output paths for results, snapshots, and figures.

## Notes

- `train` runs a plasticity / training simulation and saves both the network state and output data.
- `baseline` runs a non-plasticity baseline simulation.
- `test` runs a test simulation using a previously saved pretraining state.
- The simulation duration is controlled by the `--simdur` argument.

## Common output files

- `results/stim_experiments_train_<experiment>_mon_<monitor>.../*.npz`
- `states/states_<experiment>.../*.bp`
- `figures/figs_train_<experiment>.../`
