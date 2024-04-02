
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch
import scipy.stats as stats
from scene import Scene


combis = [(3, 0)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 

class DotDict:
    def __init__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

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
            
            # # Step 2: Filter out outliers in each dimension
            filtered_points = gm.get_xyz.cpu().numpy()
            # for dim in range(filtered_points.shape[1]):
            #     z_scores = np.abs(stats.zscore(filtered_points[:, dim]))
            #     filtered_points = filtered_points[(z_scores < 2)]  # Adjust the threshold as needed

            # Step 2: Project points onto xy-plane (top-down view)
            projected_points = filtered_points[:, :3] # Keep only x and y coordinates

            # Step 3: Define grid parameters
            grid_resolution = 0.1  # Adjust as needed
            x_min, x_max = projected_points[:, 0].min(), projected_points[:, 0].max()
            y_min, y_max = projected_points[:, 2].min(), projected_points[:, 2].max()
            
            # Step 4: Create grid and count points in each cell
            x_bins = np.arange(x_min, x_max, grid_resolution)
            y_bins = np.arange(y_min, y_max, grid_resolution)
            heatmap, _, _ = np.histogram2d(projected_points[:, 0], projected_points[:, 2], bins=[x_bins, y_bins])

            # Step 5: Plot heatmap
            plt.imshow(heatmap.T, extent=[-30, 20, -20, 20], origin='lower', cmap='hot', vmin=0, vmax=5000)
            plt.colorbar(label='Point count')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Point Cloud Heatmap (Top-Down View)')
            # plt.show()

            plt.savefig(f'plots/density_{scene}_{combi[0]}_{combi[1]}.png')
            plt.clf()





