import os
import numpy as np


# os.system(f'python measure_fps.py -m output/0.001_shdegree_0')




    


    # for entry in failed_runs:
    #     sh = entry[2]
    #     rc = entry[1]
    #     scene_n = entry[0]
sh_degree = [0]
reg_constant = [0]    
scenes = ["bicycle"] #,"counter", "kitchen","truck", "train","drjohnson"


for sh in sh_degree:
    for rc in reg_constant:
        for scene in scenes:
            for i in range(0,10):
                os.system(f'python measure_fps.py -m eval/full_eval_{sh}_{rc}/{scene}')



with open("output/0.001_shdegree_0/fps_run.txt") as f:
    runs = f.readlines()
    runs = np.array([float(str.strip(run)) for run in runs])

    print(1000.0/np.mean(runs))