import torch
import torch.nn as nn

class EnvMap():

    def __init__(self, num_gaussians):
        self._weights = nn.Parameter(torch.zeros((num_gaussians, 3, 5), device="cuda")  + 0.01, requires_grad=True)

    @property
    def get_gaussians(self):
        return self._weights