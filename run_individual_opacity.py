import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
from scene.gaussian_model import GaussianModel
import torch


with torch.no_grad():
    path = f'opacity_test\\counter_opacity_0.01'
    final_path = os.path.join(path,"point_cloud","iteration_" + str(7000),"point_cloud.ply")

    print(final_path)
    gm = GaussianModel(3)
    gm.load_ply(final_path)

    opacities = gm.get_opacity

    

    plt.hist(opacities.cpu().numpy(), bins=20, range=(0,1), density=True)



    plt.xlabel('sum')
    plt.ylabel('#occurances')
    plt.title(f'counter')
    plt.legend()
    # Display the plot
    plt.show()
    # plt.savefig(f'plots/opacity_plot_{scene}_{combi[0]}_{combi[1]}.png')
    # plt.clf()