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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def pytorch2opencv(pytorch_tensor):
    numpy_image = pytorch_tensor.detach().cpu().numpy()
    numpy_image = numpy_image.transpose(1,2,0)
    return numpy_image

def show_pytorch_image(pytorch_tensor, title=["Image"]):
    
    import cv2
    for idx, image in enumerate(pytorch_tensor):
        numpy_image = pytorch2opencv(image)

        cv2.imshow(title[idx], numpy_image)
    cv2.waitKey()
