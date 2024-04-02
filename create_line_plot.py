import pandas
import numpy as np
import seaborn
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

combis = [(3, 0), (3, 0.0001), (3, 0.001)]
# combis2 = [(0, 0), (0, 0)]
# combis3 = [(3, 0), (3, 0)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 


# Define the number of desired colors
num_colors = 6

# Choose a colormap
cmap = plt.cm.coolwarm
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)

# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]

for idx, scene in enumerate(scenes):

    color = colors[idx]
    PSNR = []
    SIZE_IN_MB = []
    FPS = []
    LPIPS = []
    for combi in combis:
        path = f'full_eval_{combi[0]}_{combi[1]}/{scene}'

        with open(f'eval/{path}/results.json', 'r') as f:
            data = json.load(f)
            
            try:
                print()
                PSNR.append(data['ours_30000']['PSNR'])
                LPIPS.append(data['ours_30000']['LPIPS'])
            except:
                PSNR.append(data['ours_7000']['PSNR'])
                LPIPS.append(data['ours_7000']['LPIPS'])

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

    # bar1 = plt.bar(scene, SIZE_IN_MB[0], alpha=0.5,color=color)
    # bar2 = plt.bar(scene, SIZE_IN_MB[1], alpha=0.5,color=color)

    # plt.text(bar1.get_x() + bar1.get_width() / 2.0, bar1.get_height() , f'{PSNR[0]}', ha='center', va='bottom')
    # plt.text(bar2.get_x() + bar2.get_width() / 2.0, bar2.get_height() , f'{PSNR[1]}', ha='center', va='bottom')

    # i = 0
    # for rect in bar1 + bar2:
    #     height = rect.get_height()
    #     if i % 2 == 0:
    #         plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{PSNR[0]:.1f}', ha='center', va='bottom')
    #     else:
    #         plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{PSNR[1]:.1f}', ha='center', va='bottom') 

    #     i+=1 
        


    plt.plot([SIZE_IN_MB[0]/x for x in SIZE_IN_MB], [x-PSNR[0] for x in PSNR], label=f'{scene}', c=color, )
    
    
    # plt.plot(SIZE_IN_MB, PSNR, '--', label=f'{scene}', c=color, )
    # plt.plot(FPS, PSNR, label=f'{scene}', c=color, )

    PSNR = []
    SIZE_IN_MB = []
    FPS = []
    # for combi in combis2:
    #     path = f'full_eval_{combi[0]}_{combi[1]}/{scene}'
    #     with open(f'eval/{path}/results.json', 'r') as f:
    #         data = json.load(f)
            
    #         try:
    #             print()
    #             PSNR.append(data['ours_30000']['PSNR'])
    #         except:
    #             PSNR.append(data['ours_7000']['PSNR'])

    #     with open(f'eval/{path}/fps_run.txt', 'r') as f:
    #         runs = f.readlines()
    #         runs = np.array([float(str.strip(run)) for run in runs])
            
    #         FPS.append(1000.0/np.mean(runs))
    #         # print(runs)

    #     file_path = f'eval/{path}/point_cloud/iteration_30000/point_cloud.ply'

    #     # FILE SIZE
    #     file_size_bytes = os.path.getsize(file_path)

    #     # Convert bytes to megabytes
    #     file_size_mb = file_size_bytes / (1024 * 1024)

    #     SIZE_IN_MB.append(file_size_mb)

    # plt.plot(FPS, PSNR, '-.', label=f'{scene}{combi[1]}1', c=color)

    # PSNR = []
    # SIZE_IN_MB = []
    # FPS = []
#     for combi in combis3:
#         path = f'full_eval_{combi[0]}_{combi[1]}/{scene}'
#         with open(f'eval/{path}/results.json', 'r') as f:
#             data = json.load(f)
            
#             try:
#                 print()
#                 PSNR.append(data['ours_30000']['PSNR'])
#             except:
#                 PSNR.append(data['ours_7000']['PSNR'])

#         with open(f'eval/{path}/fps_run.txt', 'r') as f:
#             runs = f.readlines()
#             runs = np.array([float(str.strip(run)) for run in runs])
            
#             FPS.append(1000.0/np.mean(runs))
#             # print(runs)

#         file_path = f'eval/{path}/point_cloud/iteration_30000/point_cloud.ply'

#         # FILE SIZE
#         file_size_bytes = os.path.getsize(file_path)

#         # Convert bytes to megabytes
#         file_size_mb = file_size_bytes / (1024 * 1024)

#         SIZE_IN_MB.append(file_size_mb)

    # plt.plot(FPS, PSNR, label=f'{scene}{combi[1]}1', c=color)

# Add labels and title
plt.xlabel('Compression Factor')
plt.ylabel('Delta PSNR')
plt.title('Compression/Delta PSNR')
plt.legend()
plt.savefig(f'deltaPSNR', bbox_inches='tight')
# plt.clf()

# # # Display the plot
# plt.show()



