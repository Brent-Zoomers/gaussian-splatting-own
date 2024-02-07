import os
import numpy as np


# for i in range(0,10):
#     os.system(f'python measure_fps.py -m output/0')
#     os.system(f'python measure_fps.py -m output/0.001')
#     os.system(f'python measure_fps.py -m output/0.0001')


with open("output/0.0001/fps_run.txt") as f:
    runs = f.readlines()
    runs = np.array([float(str.strip(run)) for run in runs])

    print(1000.0/np.mean(runs))