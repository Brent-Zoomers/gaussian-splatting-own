import torch.distributions as distributions
import matplotlib.pyplot as plt
import torch

beta_distribution = distributions.Beta(5, 2)

y = torch.sort(beta_distribution.sample([100000]))[0].cpu().numpy()
x = [i for i in range(100000)]


plt.plot(x, y)

# Adding labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot Example')

# Display the plot
plt.show()