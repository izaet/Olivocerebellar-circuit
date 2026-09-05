export CUDA_VISIBLE_DEVICES=0

python3 main_entrain.py --run-type baseline --experiment nostim --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 88 --monitor-preset "plasticity_min" --simdur 10000.0 --parent-dir /home/izet/Olivocerebellar-circuit --downsample 80
python3 main_entrain.py --run-type baseline --experiment nostim --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 89 --monitor-preset "plasticity_min" --simdur 10000.0 --parent-dir /home/izet/Olivocerebellar-circuit --downsample 80
python3 main_entrain.py --run-type baseline --experiment nostim --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 90 --monitor-preset "plasticity_min" --simdur 10000.0 --parent-dir /home/izet/Olivocerebellar-circuit --downsample 80
python3 main_entrain.py --run-type baseline --experiment nostim --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 91 --monitor-preset "plasticity_min" --simdur 10000.0 --parent-dir /home/izet/Olivocerebellar-circuit --downsample 80
python3 main_entrain.py --run-type baseline --experiment nostim --PFPC_plasticity-on True --OU-stim-io-on False --OU-stim-pf-on False --seed 92 --monitor-preset "plasticity_min" --simdur 10000.0 --parent-dir /home/izet/Olivocerebellar-circuit --downsample 80
