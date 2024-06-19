import os
import json
import matplotlib.pyplot as plt
import numpy as np


output_scenes = [f'weighted_scaling_opacity/{x/10.0}' for x in range(1,6)] 
output_scenes_1 = [f'opacity/{x/10.0}' for x in range(1,6)] 
output_scenes_2 = [f'scaling/{x/10.0}' for x in range(1,6)] 
output_scenes_3 = [f'top{x/10.0}percentcontributors_weighted_by_oc' for x in range(1,6)] 
output_scenes_4 = [f'mine_weighted_opacity/{x}' for x in range(1,6)] 

PSNR = []
current_metric = 'SSIM'

for scene in output_scenes:
    file_name = f'output/{scene}/results.json'
    f = open(file_name)
    data = json.load(f)
    PSNR.append(data['ours_30000'][current_metric])
plt.plot(PSNR, label="combined")


PSNR = []
for scene in output_scenes_1:
    file_name = f'output/{scene}/results.json'
    f = open(file_name)
    data = json.load(f)
    PSNR.append(data['ours_30000'][current_metric])
plt.plot(PSNR, label="opacity")


PSNR = []
for scene in output_scenes_2:
    file_name = f'output/{scene}/results.json'
    f = open(file_name)
    data = json.load(f)
    PSNR.append(data['ours_30000'][current_metric])
plt.plot(PSNR, label="scaling")

PSNR = []
for scene in output_scenes_3:
    file_name = f'output/{scene}/results.json'
    f = open(file_name)
    data = json.load(f)
    PSNR.append(data['ours_30000'][current_metric])
plt.plot(PSNR, label="mine")


PSNR = []
for scene in output_scenes_4:
    file_name = f'output/{scene}/results.json'
    f = open(file_name)
    data = json.load(f)
    PSNR.append(data['ours_30000'][current_metric])
plt.plot(PSNR, label="mine+opacity")


plt.legend()
plt.ylim(0.45,0.9)
plt.xticks(np.arange(0, 6, 1))
plt.xlabel("Percentage splats")
plt.ylabel(current_metric)
plt.show()

