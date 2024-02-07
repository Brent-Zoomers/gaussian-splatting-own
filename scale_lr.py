start_epoch = 0
finish_epoch = 15_000
total_iterations = 30000

start_value = 1e-3
finish_value = 1e-6

# start_value = 1e-6
# finish_value = 1e-3

import math
import matplotlib.pyplot as plt
import numpy as np


# start_value = 1.0
# finish_value = 0.001

# Define the scaling function using exponential decay
def exponential_scale_lr(iteration):
    scale_factor = np.log(finish_value / start_value) / finish_epoch
    return start_value * np.exp(-scale_factor * iteration)



def logarithmic_scale_lr(epoch):

    if epoch < start_epoch:
        return start_value
    if epoch > finish_epoch:
        return finish_value

    epoch = finish_epoch - epoch
   
    scale = math.log(finish_value / start_value) / (finish_epoch - start_epoch)
    return start_value - (start_value * math.exp(scale * (epoch - start_epoch)))
    

logarithmic_values = [logarithmic_scale_lr(i) for i in range(total_iterations)]
# logarithmic_values = start_value - logarithmic_values
# exp_values = [exponential_scale_lr(i) for i in range(total_iterations)]



# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(logarithmic_values, label='Logarithmic Scaling')
# plt.plot(exp_values, label='Exp Scaling')
plt.title('Scaled Values over 30,000 Iterations')
plt.xlabel('Iteration')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)
plt.show()


