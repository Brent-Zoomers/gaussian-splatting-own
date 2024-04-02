import wandb
import pandas as pd
import matplotlib. pyplot as plt
# Read our CSV into a new DataFrame
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

new_ng_dataframe = pd.read_csv("num_gaussians_truck.csv")

values = [ 1 * 10**(-x) for x in range(0,9)]


num_colors = 9
# Choose a colormap
cmap = plt.cm.coolwarm
# Create a ScalarMappable object
sm = ScalarMappable(norm=Normalize(vmin=20, vmax=30), cmap=cmap)

# Discretize the colormap
colors = [sm.to_rgba(value) for value in np.linspace(20, 30, num_colors)]


i = 0
for value in values:
    data = new_ng_dataframe[f'run_{float(value)} - num_gaussians'].to_numpy()
    plt.plot([x for x in range(len(data))], data, color=colors[i], label=value)
    i+=1

plt.xlabel('Iteration')
plt.ylabel('#Gaussians')
plt.legend()
plt.title('Size(MB) over Iterations')
plt.savefig(f'wandb_tern.png', bbox_inches='tight')
plt.clf()






