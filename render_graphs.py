import os
import numpy as np
import json

sh_degree = [0, 3]
reg_constant = [1e-3, 1e-4, 0]


all_sources = []

common_args = " --quiet --eval --skip_train"
# for sh in sh_degree:
#     for rc in reg_constant:
        # path = f'full_eval_{sh}_{rc}'
        
#         os.system("python render.py --iteration 7000 -s " + "datasets/truck" + " -m eval/" + path + "/" + "truck" + common_args)
#         os.system("python render.py --iteration 30000 -s " + "datasets/truck" + " -m eval/" + path + "/" + "truck" + common_args)


scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"]

# scenes_string = ""
# for sh in sh_degree:
#     for rc in reg_constant:
#         pass
       



# with open(f'eval/{path}/fps_run.txt', 'r') as f:
#     runs = f.readlines()
#     runs = np.array([float(str.strip(run)) for run in runs])
#     print(1000.0/np.mean(runs))
import matplotlib.pyplot as plt


# for scene in scenes:
#     for sh in sh_degree:
#         PSNR = []
#         FPS = []
#         SIZE_IN_MB = []

#         for rc in reg_constant:
#             path = f'full_eval_{sh}_{rc}/{scene}'

#             print(path)
#             with open(f'eval/{path}/results.json', 'r') as f:
#                 data = json.load(f)
                
#                 try:
#                     print()
#                     PSNR.append(data['ours_30000']['PSNR'])
#                 except:
#                     PSNR.append(data['ours_7000']['PSNR'])


            
#             with open(f'eval/{path}/fps_run.txt', 'r') as f:
#                 runs = f.readlines()
#                 runs = np.array([float(str.strip(run)) for run in runs])
                
#                 FPS.append(1000.0/np.mean(runs))
#                 # print(runs)

#             file_path = f'eval/{path}/point_cloud/iteration_30000/point_cloud.ply'

#             # FILE SIZE
#             file_size_bytes = os.path.getsize(file_path)

#             # Convert bytes to megabytes
#             file_size_mb = file_size_bytes / (1024 * 1024)

#             SIZE_IN_MB.append(file_size_mb)

#         plt.scatter(FPS, PSNR, label=f'{scene}_{sh}')

#     # Add labels and title
#     plt.xlabel('FPS')
#     plt.ylabel('PSNR')
#     plt.legend()
#     plt.title('Scatter Plot')

#     # Show plot
#     plt.show()




min_max = [()]

for scene in scenes:
    PSNR = []
    FPS = []
    SIZE_IN_MB = []


    path = f'full_eval_{3}_{0}/{scene}'

    with open(f'eval/{path}/results.json', 'r') as f:
        data = json.load(f)
        
        try:
            print()
            PSNR.append(data['ours_30000']['PSNR'])
        except:
            PSNR.append(data['ours_7000']['PSNR'])


    
    with open(f'eval/{path}/fps_run.txt', 'r') as f:
        runs = f.readlines()
        runs = np.array([float(str.strip(run)) for run in runs])
        
        FPS.append(1000.0/np.mean(runs))
        # print(runs)

    file_path = f'eval/{path}/point_cloud/iteration_30000/point_cloud.ply'

    # FILE SIZE
    file_size_bytes = os.path.getsize(file_path)

    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)

    SIZE_IN_MB.append(file_size_mb)