export CUDA_VISIBLE_DEVICES=
taskset -c 0,1 python3 main_entrain.py --experiment nostim --run-type train --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 88
taskset -c 2,3 python3 main_entrain.py --experiment nostim --run-type train --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 89
taskset -c 4,5 python3 main_entrain.py --experiment nostim --run-type train --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 90
taskset -c 6,7 python3 main_entrain.py --experiment nostim --run-type train --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 91