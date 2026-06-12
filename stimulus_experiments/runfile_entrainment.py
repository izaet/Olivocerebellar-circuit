import numpy as np

seedlist = np.arange(88, 92)
ISI_values = np.linspace(20, 200, 4)
for ISI in ISI_values:
    for seed in seedlist:
        print(f'python3 main_entrain.py --experiment specific-isi --run-type train --PFPC_plasticity-on True --OU-stim-isi-mean {ISI} --seed {seed}.txt')
        