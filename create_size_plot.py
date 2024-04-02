
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch

combis = [(3, 0)]
scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 


# Define the number of desired colors
num_colors = 6
# Choose a colormap
cmap = plt.cm.viridis
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)
# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

for idx, scene in enumerate(scenes):
   
    color = colors[idx]

    for combi in combis:

        # os.system(f'python render_n_biggest.py --amount_biggest_entry 0.5 -m eval/full_eval_{combi[0]}_{combi[1]}/{scene}')

        # image1_path = f'eval/full_eval_{combi[0]}_{combi[1]}/{scene}/test/show_largest_0.5_30000/gt'
        # image2_path = f'eval/full_eval_{combi[0]}_{combi[1]}/{scene}/test/show_largest_0.5_30000/renders'


        

        with torch.no_grad():
            path = f'eval/full_eval_{combi[0]}_{combi[1]}/{scene}'
            final_path = os.path.join(path,"point_cloud","iteration_" + str(30000),"point_cloud.ply")

            print(final_path)
            gm = GaussianModel(combi[0])
            gm.load_ply(final_path)

            scaling = torch.abs(gm.get_scaling)

            final_scaling = torch.prod(gm.get_scaling, dim=-1)

            plt.hist(final_scaling.cpu().numpy(), bins=10000, range=[0,0.000005])

            # Find the median value
            median_value = np.median(final_scaling.cpu().numpy())

            # Highlight the lowest 50% with a different color
            plt.hist(final_scaling.cpu().numpy()[final_scaling.cpu().numpy()>= median_value], range=[0,0.000005], color='green', bins=10000)
# plt.xticks([1, 2, 3, 4, 5, 6], scenes)
# plt.boxplot(data)

# Add labels and title
            plt.xlabel('Size of Gaussian')
            plt.ylabel('#occurances')
            plt.title(f'{scene}')
            # plt.legend()
            # Display the plot
            plt.ylim(0, 7500)
            # plt.show()
            
            plt.savefig(f'plots/true_scale_hist_{scene}_{combi[0]}_{combi[1]}.png')
            plt.clf()



