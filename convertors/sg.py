import torch.nn as nn
import torch.nn.functional as F
import torch

class GaussianMixtureModel(nn.Module):
    def __init__(self, num_batches, num_components, weights):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.weights = weights
        self.num_batches = num_batches

    def spherical_to_cartesian(self, theta, phi):
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        return torch.stack((x, y, z), dim=-1)

    # N x 3
    def forward(self, direction):
        # Stack the results along a new dimension (num_components1
        w_in = self.weights.view(self.num_batches, self.num_components, 4)

        mu_ = w_in[...,:2]
        alpha_ = w_in[...,2]
        lambda_ = w_in[...,3] 

        mu_normalized = self.spherical_to_cartesian(mu_[...,0], mu_[...,1])
        # Gx5
        # Nx3
        # direction = direction.unsqueeze(1)

        # Calculate the dot product between direction and mu
        dot_product = torch.matmul(direction, mu_normalized.transpose(1,2))

        # Calculate the exponential term
        exponential_term = torch.exp(lambda_.unsqueeze(1) * (dot_product - 1.0))

        # Multiply by w_in[:, 3] and sum along the second dimension
        result = torch.sum(alpha_.unsqueeze(1)  * exponential_term, dim=2)

        return result
        # B x N