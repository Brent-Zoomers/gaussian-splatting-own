
import random
import torch.nn as nn
import torch.nn.functional as F
import torch
from convertors.sg import GaussianMixtureModel
from convertors.sh import SH
import time


def generate_uniform_rays(num_rays):
    # Generate random values for theta and phi
    theta = torch.acos(1 - 2 * torch.rand(num_rays))  # Polar angle
    phi = 2 * torch.pi * torch.rand(num_rays)        # Azimuthal angle

    # Convert spherical coordinates to Cartesian coordinates
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    # Stack the coordinates to form the rays
    rays = torch.stack((x, y, z), dim=-1)

    # normalized_rays = F.normalize(rays, p=2, dim=-1)

    return rays


def convert_sh2sg(sh_weights, amount_gaussians=10, num_rays=40):
   
    epochs = 1000
    batch_size = sh_weights.shape[0]
    weights = torch.rand(batch_size, 4 * amount_gaussians, requires_grad=True, device="cuda")

    # Nx9
    sh = SH(batch_size, torch.rand(batch_size, 9).cuda())
    sg = GaussianMixtureModel(batch_size, amount_gaussians, weights)

    optimizer = torch.optim.Adam([weights], lr=1e-2)

    rays = generate_uniform_rays(num_rays).cuda()
    color_sh = sh(rays)

    for _ in range(0, epochs):
        
        color_sg = sg(rays)
        loss = torch.mean(torch.abs(color_sh-color_sg))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    color_sh = sh(rays)
    color_sg = sg(rays)

    print(torch.mean(torch.abs(color_sh-color_sg)))

    return weights