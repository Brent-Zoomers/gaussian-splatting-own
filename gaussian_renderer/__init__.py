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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, eval_sg, eval_sg_env
from utils.general_utils import build_rotation

iteration = 0
# env_map = torch.ones((1,3,1)).cuda().float()

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pipe.irradiance_model == "sh":
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            if pipe.irradiance_model == "sg":
                sgs_view = pc.get_features_sg
                sg2rgb = eval_sg(sgs_view, dir_pp_normalized, pc.get_env_map)
                colors_precomp = torch.clamp_min(sg2rgb + 0.5, 0.0)
            if pipe.irradiance_model == "sg_env":
                sg_diff = pc.get_features_diff
                sg_spec = pc.get_features_spec
                # normals = torch.rand((pc.get_features.shape[0],3)).cuda().float()

                # Create cov to find eigenvectors -> smallest is normal

                rotation_quats = pc.get_rotation.clone()

                rotation_matrices = build_rotation(rotation_quats)

                scalings = pc.get_scaling

                smallest_axis_idx = scalings.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
                normals_ = rotation_matrices.gather(2, smallest_axis_idx)
                normals_ = -normals_.squeeze(dim=2)

                sgenv2rgb = eval_sg_env(sg_diff, sg_spec, pc.get_env_map, dir_pp_normalized, normals_)
                colors_precomp = torch.clamp_min(sgenv2rgb + 0.5, 0.0)
            
                
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # size is #Gx3

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, contr_per_pixel, ids_per_pixel = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # x_indices = torch.arange(100)
    # y_indices = torch.arange(100)

    # # Create a mesh grid of all combinations of x and y indices
    # mesh_grid = torch.meshgrid(x_indices, y_indices)

    # # Stack the x and y coordinates into a single tensor
    # pixel_combinations = torch.stack((mesh_grid[0].flatten(), mesh_grid[1].flatten()), dim=1)
    # W = int(viewpoint_camera.image_width)
    # y = torch.tensor([x[0]*W + x[1] for x in pixel_combinations])

    # # final_ids = torch.cat([ids_per_pixel[z.item()*20:z.item()*20+20] for z in y])
    
    flattened = ids_per_pixel[0:20]
    values, counts = flattened.unique(return_counts=True)

    # largest_values, largest_indices = torch.topk(counts, k=500)
    # print(torch.max(values), means3D.shape)
    rendered_image_largest, radii_, contr_per_pixel, ids_per_pixel = rasterizer(
        means3D = means3D[values],
        means2D = means2D[values],
        shs = shs,
        colors_precomp = colors_precomp[values],
        opacities = opacity[values],
        scales = scales[values],
        rotations = rotations[values],
        cov3D_precomp = cov3D_precomp)
    
    # smallest_values, largest_indices_ = torch.topk(counts, k=500, largest=False)

    # largest_indices = largest_indices_

    # rendered_image_smallest, radii_, contr_per_pixel, ids_per_pixel = rasterizer(
    #     means3D = means3D[largest_indices],
    #     means2D = means2D[largest_indices],
    #     shs = shs,
    #     colors_precomp = colors_precomp[largest_indices],
    #     opacities = opacity[largest_indices],
    #     scales = scales[largest_indices],
    #     rotations = rotations[largest_indices],
    #     cov3D_precomp = cov3D_precomp)

    from torchvision import transforms
    import cv2
    import numpy as np
    # # Create a transform to convert the tensor to a PIL image
    transform = transforms.ToPILImage()

    # # Convert the tensor to a PIL image
    img_l = transform(rendered_image_largest)
    # img_s = transform(rendered_image_smallest)

    # # Display the image (requires an external library like matplotlib or OpenCV)
    # img_l.show()  # Using matplotlib
    # # Alternatively, use OpenCV for display:
    # global iteration
    # iteration = iteration+1
    # if iteration % 500 == 0:
    #     cv2.imshow("Image_l", cv2.cvtColor(np.array(img_l), cv2.COLOR_BGR2RGB))
    #     cv2.waitKey(0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
