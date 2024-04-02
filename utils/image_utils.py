#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import matplotlib.pyplot as plt
def imshow(tensor):
    # Convert tensor to NumPy array
    image_np = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Display image
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2, background_color=(0,0,0)):
    # Convert background color to torch tensor
    background_color = torch.tensor(background_color).reshape(1, 3, 1, 1).float().cuda()
    
    # Expand background color tensor to match the shape of img1 and img2
    # background_color = background_color.expand_as(img1).cuda()
    
    # Mask pixels with background color
    mask = torch.all(img1 != background_color, dim=1)
    mask = mask.repeat(3,1,1)
    mask = mask.unsqueeze(0)

    imshow(mask.type(torch.uint8) * 255)
    
    # Calculate MSE for non-background pixels
    mse = ((img1- img2) ** 2)

    masked_mse = mse[mask].mean()

    # Check for divide by zero in case MSE is zero
    if masked_mse == 0:
        return float('inf')
    
    # Calculate PSNR
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(masked_mse))
    return psnr_value
    
