import torch
import torch.nn as nn

class EnvMap():

    def __init__(self, num_gaussians):
        data = torch.zeros((num_gaussians, 3, 5), device="cuda")
        data[...,3] = 0.01
        data[...,4] = -3
        data[...,0:3] = torch.rand_like(data[...,0:1,0:3]).repeat(1,3,1)
        
        self._weights = nn.Parameter(data, requires_grad=True)

        # self._weights = nn.Parameter(torch.zeros((num_gaussians, 3, 5), device="cuda") + 0.01, requires_grad=True)

    @property
    def get_gaussians(self):
        return self._weights