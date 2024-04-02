import wandb
import pandas as pd
import matplotlib. pyplot as plt
# Read our CSV into a new DataFrame
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

new_ng_dataframe = pd.read_csv("runs_comparison.csv")

scenes = ["truck", "train", "bicycle", "kitchen"] 

num_colors = len(scenes)
# Choose a colormap
cmap = plt.cm.coolwarm
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)

# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]


i = 0
for scene in scenes:


    data = new_ng_dataframe[f'full_eval_eval/full_eval_3_0/{scene}_0.0_3 - num_gaussians'].to_numpy()
    plt.bar(scene, data, color=colors[i], label=scene, alpha=0.5)
    data = new_ng_dataframe[f'full_eval_eval/full_eval_3_0.001/{scene}_0.001_3 - num_gaussians'].to_numpy()
    plt.bar(scene, data, color=colors[i], alpha=0.5)


    i+=1
plt.legend()
plt.show()






