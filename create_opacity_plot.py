
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch

# combis = [(3, 0), (0, 0)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 


# Define the number of desired colors
num_colors = 6
# Choose a colormap
cmap = plt.cm.viridis
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)
# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]



for idx, scene in enumerate(scenes):
   
    color = colors[idx]

    # for combi in combis:

    with torch.no_grad():
        path = f'am_gaussians_opacity1e-2/{scene}'
        final_path = os.path.join(path,"point_cloud","iteration_" + str(30000),"point_cloud.ply")

        print(final_path)
        gm = GaussianModel(3)
        gm.load_ply(final_path)

        opacities = gm.get_opacity

        

        plt.hist(opacities.cpu().numpy(), bins=20, range=(0,1), density=True)


# plt.xticks([1, 2, 3, 4, 5, 6], scenes)
# plt.boxplot(data)

# Add labels and title
        plt.xlabel('sum')
        plt.ylabel('#occurances')
        plt.title(f'{scene}')
        plt.legend()
        # Display the plot
        plt.show()
        # plt.savefig(f'plots/opacity_plot_7000it_{scene}_{combi[0]}_{combi[1]}.png')
        # plt.clf()



