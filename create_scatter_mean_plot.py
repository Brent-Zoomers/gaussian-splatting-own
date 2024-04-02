
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch
import json

combis = [(3, 0),(3, 1e-3), (3, 1e-4)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 

# Define the number of desired colors
num_colors = 6

# Choose a colormap
cmap = plt.cm.viridis
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)
print("lol")
# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]

data = []

for idx, scene in enumerate(scenes):
   
    color = colors[idx]
    for combi in combis:

        with open(f'eval/full_eval_{3}_{combi[1]}/{scene}/results.json', 'r') as f:
            with open(f'eval/full_eval_{0}_{combi[1]}/{scene}/results.json', 'r') as f1:
                data = json.load(f)
                PSNR = 0
              
                PSNR = data['ours_30000']['PSNR']


                data1 = json.load(f1)
                PSNR1 = 0
              
                PSNR1 = data1['ours_30000']['PSNR']


                PSNR -= PSNR1
               

            


        with torch.no_grad():
            path = f'eval/full_eval_{combi[0]}_{combi[1]}/{scene}'
            final_path = os.path.join(path,"point_cloud","iteration_" + str(30000),"point_cloud.ply")

            print(final_path)
            gm = GaussianModel(combi[0])
            gm.load_ply(final_path)

            higher_order_shs = gm._features_rest

            mean_shs = torch.mean(higher_order_shs)

            # data.append(mean_shs.cpu().numpy())

            plt.scatter(mean_shs.cpu().numpy(), [PSNR],c=colors[idx], label=f'{scene}')
            # plt.scatter(mean_shs.cpu().numpy(), [PSNR], bins=500, range=(0,10), c=colors[idx])


# plt.xticks([1, 2, 3, 4, 5, 6], scenes)
# plt.boxplot(data)

# Add labels and title
plt.xlabel('sum')
plt.ylabel('#occurances')
plt.title(f'{scene}')
plt.legend()
# Display the plot
plt.show()
        # plt.savefig(f'plots/ogonly{scene}.png')
        # plt.clf()



