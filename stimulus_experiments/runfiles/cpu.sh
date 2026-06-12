export CUDA_VISIBLE_DEVICES=
taskset -c 0,1 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 140.0 --seed 88 &
taskset -c 2,3 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 140.0 --seed 89 &
taskset -c 4,5 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 140.0 --seed 90 &
taskset -c 6,7 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 140.0 --seed 91 &
taskset -c 8,9 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 200.0 --seed 88 &
taskset -c 10,11 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 200.0 --seed 89 &
taskset -c 12,13 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 200.0 --seed 90 &
taskset -c 14,15 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 200.0 --seed 91 &
taskset -c 16,17 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 20.0 --seed 88 &
taskset -c 18,19 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 20.0 --seed 89 &
taskset -c 20,21 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 20.0 --seed 90 &
taskset -c 22,23 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 20.0 --seed 91 &
taskset -c 24,25 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 80.0 --seed 88 &
taskset -c 26,27 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 80.0 --seed 89 &
taskset -c 28,29 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 80.0 --seed 90 &
taskset -c 30,31 python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean 80.0 --seed 91
