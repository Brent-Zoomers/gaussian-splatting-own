
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch

combis = [(3, 0), (3, 1e-3)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 


# Define the number of desired colors
num_colors = 6

# Choose a colormap
cmap = plt.cm.viridis
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)
# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]

data = []

for idx, scene in enumerate(scenes):
   
    color = colors[idx]

    for combi in combis:

        with torch.no_grad():
            path = f'eval/full_eval_{combi[0]}_{combi[1]}/{scene}'
            final_path = os.path.join(path,"point_cloud","iteration_" + str(30000),"point_cloud.ply")

            print(final_path)
            gm = GaussianModel(combi[0])
            gm.load_ply(final_path)

            higher_order_shs = gm._features_rest

            sum_shs = torch.sum(torch.sum(torch.abs(gm._features_rest), dim=1), dim=1)

            data.append(sum_shs.cpu().numpy())

        plt.hist(sum_shs.cpu().numpy(), bins=500, range=(0,7), label=f'{scene}_{combi[1]}' ,weights = np.ones_like(sum_shs.cpu().numpy()) / len(sum_shs.cpu().numpy()), alpha = 0.5)


# plt.xticks([1, 2, 3, 4, 5, 6], scenes)
# plt.boxplot(data)

# Add labels and title
    plt.xlabel('sum')
    plt.ylabel('#occurances')
    plt.title(f'{scene}_{combi[1]}')
    plt.legend()
    # Display the plot
    plt.show()
        # plt.savefig(f'plots/scale_{scene}.png')
    # plt.clf()



