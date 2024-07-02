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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import copy
from voxels.voxel_grid import SparseVoxelGrid


def find_local_maxima(tensor):
    # Ensure the tensor is 2D
    assert tensor.ndim == 2, "The input tensor must be 2D"
    
    N = tensor.shape[0]
    
    # Pad the tensor to handle edge cases
    padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode='constant', value=float('-inf'))
    
    # Create shifts for comparison
    shifts = [
        (0, 1),   # right
        (0, -1),  # left
        (1, 0),   # down
        (-1, 0),  # up
        (1, 1),   # down-right
        (1, -1),  # down-left
        (-1, 1),  # up-right
        (-1, -1)  # up-left
    ]
    
    # Initialize the mask for local maxima
    maxima_mask = torch.ones_like(tensor, dtype=torch.bool)
    
    # Compare each element with its neighbors
    for shift in shifts:
        shifted_tensor = torch.roll(padded_tensor, shifts=shift, dims=(0, 1))[1:N+1, 1:N+1]
        maxima_mask &= tensor > shifted_tensor
    
    # Get the indices of local maxima
    local_maxima_indices = torch.nonzero(maxima_mask)
    
    return local_maxima_indices, tensor[local_maxima_indices[:, 0], local_maxima_indices[:, 1]]

""""
@input : Nx3
@output : Nx2

"""
def cartesian2polar(input_tensors):
    real_eigenvectors = torch.real(input_tensors)

    x = real_eigenvectors[...,0]
    y = real_eigenvectors[...,1]
    z = real_eigenvectors[...,2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # r = torch.sqrt(x**2 + y**2 + z**2) is one
    theta = torch.atan2(y, x)
    phi = torch.acos(z/r)

    polar_coordinates = torch.stack((phi.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1).squeeze(-1)

    return polar_coordinates

""""
@input : Nx2
@output : Nx3

"""
def polar2cartesian(input_tensors):
    phi = input_tensors[...,0]
    theta = input_tensors[...,1]

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=1)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(f'output/debug/{model_path}', name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(f'output/debug/{model_path}', name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_indices(scene, gaussians, dataset, indices, name=""):
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    mask = indices.squeeze()

    clone_opacity = gaussians._opacity.clone()
    clone_xyz = gaussians._xyz.clone()
    clone_scaling = gaussians._scaling.clone()
    clone_features_dc = gaussians._features_dc.clone()
    clone_features_rest = gaussians._features_rest.clone()
    clone_rotation = gaussians._rotation.clone()

    gaussians._opacity = gaussians._opacity[mask] 
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._rotation = gaussians._rotation[mask]

    render_set(dataset.model_path, name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

    gaussians._opacity = clone_opacity
    gaussians._xyz = clone_xyz
    gaussians._scaling = clone_scaling 
    gaussians._features_dc = clone_features_dc
    gaussians._features_rest = clone_features_rest
    gaussians._rotation = clone_rotation

"""
@input Nx3x3
@output N
"""
def calculate_real_eigenvectors(matrices):
    eigenvalues, eigenvectors = torch.linalg.eig(matrices)
    return eigenvalues, torch.real(eigenvectors)

def calculate_features(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):

    with torch.no_grad():
        # Find interesting voxels that can be used to determine features
        #####################################################################################
        
        # Initialize point cloud data
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        

        # Create voxel grid with N dimensions in each direction NxNxN
        voxel_dims = 80 # TODO Change from this to hardcoded.
        points = gaussians.get_xyz
        dvg = SparseVoxelGrid(points,  voxel_dims, True)

        # Precompute eigenvectors for all Gaussians
        actual_covs = gaussians.get_actual_covariance()
        _, real_eigenvectors = calculate_real_eigenvectors(actual_covs)
        
        # 2 to only take smallest eigenvector into consideration
        polar_coordinates = cartesian2polar(real_eigenvectors[:,2,:])
        phi = polar_coordinates[...,0]
        theta = polar_coordinates[...,1]

        # import test_plot_histogram
        # test_plot_histogram.plot_data(real_eigenvectors[:,2,:].cpu().numpy())


        ##################################################################
        create_descriptor(points[50000], points, phi, theta, real_eigenvectors)

        ##################################################################

        # Count density in voxel grid to determine interesting voxels
        occurances = dvg.get_all_occurances()
        interesting_voxels = (torch.max(occurances)*0.1 < occurances).nonzero()
        #####################################################################################       
        # Calculate occurances of direction of largest eigenvector
        indices=dvg.get_points_at_tensor(interesting_voxels[0])
        AMOUNT_BUCKETS = 72
        # get histogram for cell
        phi_bucketized = torch.bucketize(phi[indices].flatten(), torch.linspace(0, torch.pi, AMOUNT_BUCKETS).cuda())
        theta_bucketized = torch.bucketize(theta[indices].flatten(), torch.linspace(-torch.pi, torch.pi, AMOUNT_BUCKETS).cuda())
        bin_indices = theta_bucketized * AMOUNT_BUCKETS + phi_bucketized
        histogram = torch.bincount(bin_indices, minlength=AMOUNT_BUCKETS * AMOUNT_BUCKETS)

        lol = histogram.view((AMOUNT_BUCKETS, AMOUNT_BUCKETS)).nonzero()

        torch.unique(lol, dim=0)

        for id, voxel in enumerate(interesting_voxels):
            indices = dvg.get_points_at_tensor(voxel)
            render_indices(scene, gaussians, dataset, indices, "vox"+str(id))

        print("")




        # # For now size per feature is 18 floats but will increase later on.
        # features_tensor = torch.zeros((interesting_voxels.shape[0], 18))
        # voxel_data = []

        # for idx,voxel in enumerate(interesting_voxels):
        #     indices=dvg.get_points_at_tensor(voxel)
        #     AMOUNT_BUCKETS = 72
        #     # get histogram for cell
        #     phi_bucketized = torch.bucketize(phi[indices].flatten(), torch.linspace(0, torch.pi, AMOUNT_BUCKETS).cuda())
        #     theta_bucketized = torch.bucketize(theta[indices].flatten(), torch.linspace(-torch.pi, torch.pi, AMOUNT_BUCKETS).cuda())

        #     bin_indices = theta_bucketized * AMOUNT_BUCKETS + phi_bucketized
           
        #     histogram = torch.bincount(bin_indices, minlength=AMOUNT_BUCKETS * AMOUNT_BUCKETS)

        #     # Calculate descriptor per voxel
        #     max_idx = torch.argmax(histogram)

        #     mask = bin_indices == max_idx

        #     # only render splats that are in this bin
        #     # render_indices(scene, gaussians, dataset, indices.squeeze(), str(idx))
            
        #     # Find N largest directions
        #     # interesting_mask = torch.nonzero(histogram >= histogram[max_idx]*0.8)
        #     _, interesting_mask = torch.topk(histogram, 6)
        #     interesting_mask = interesting_mask.unsqueeze(1)

        #     # Convert directions back to xyz
            
        #     rev_phi = interesting_mask % AMOUNT_BUCKETS
        #     rev_theta = (interesting_mask-rev_phi) / AMOUNT_BUCKETS

        #     interesting_polar_cors = torch.cat((rev_phi, rev_theta),dim=1)

        #     interesting_cartesian_cors = polar2cartesian(interesting_polar_cors)

        #     # Also use color
        #     # gaussians.get_features[:,0,:]

        #     feature_vector = interesting_cartesian_cors.flatten()
        #     features_tensor[idx] = feature_vector
        #     voxel_data.append(voxel)

        # return features_tensor, voxel_data

def filter_points():
    pass




def create_descriptor(point_tensor, xyz, phis, thetas, real_eigenvectors):

    AMOUNT_BUCKETS = 72

    ## Orientation Assignment

    # Find N closest points
    distances = torch.cdist(point_tensor.unsqueeze(0), xyz).squeeze()

    # TODO Account for scale
    # Find all points within range
    mask = (distances < distances.min() + 0.1).nonzero().squeeze()

    import test_plot_histogram
    test_plot_histogram.plot_data(real_eigenvectors[:,2,:][mask].cpu().numpy())

    # Created weighted histogram
    phi_bucketized = torch.bucketize(phis[mask].flatten(), torch.linspace(0, torch.pi, AMOUNT_BUCKETS).cuda())
    theta_bucketized = torch.bucketize(thetas[mask].flatten(), torch.linspace(-torch.pi, torch.pi, AMOUNT_BUCKETS).cuda())
    bin_indices = theta_bucketized * AMOUNT_BUCKETS + phi_bucketized
    histogram = torch.bincount(bin_indices, minlength= AMOUNT_BUCKETS*AMOUNT_BUCKETS)

    max_indices = histogram.argmax()
    max_value = histogram[max_indices] * 0.8

    peaks = torch.where(histogram >= max_value)
    
    # For each peak create discriptor
    
    # Take points in grid around point and divide into 8 cubes, each of which will again be used to calculate the histogram

    x_edges = torch.linspace(point_tensor[0]-0.5, point_tensor[0]+0.5, 3)
    y_edges = torch.linspace(point_tensor[1]-0.5, point_tensor[1]+0.5, 3)
    z_edges = torch.linspace(point_tensor[2]-0.5, point_tensor[2]+0.5, 3)

    




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    m_args = model.extract(args)

    m_args.model_path = m_args.a
    f1,vd1 = calculate_features(m_args, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    # m_args.model_path = m_args.b
    # f2,vd2 = calculate_features(m_args, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    # distances = torch.cdist(f1, f2)

    # mask = (distances < distances.min() + 1).nonzero()

    # # _, mask = torch.topk(distances, k=10, largest=False) 


    # # TEST CODE
    # with torch.no_grad():

    #     for id, entry in enumerate(mask):
    #         m_args.model_path = m_args.a
    #         gaussians = GaussianModel(m_args.sh_degree)
    #         scene = Scene(m_args, gaussians, load_iteration=args.iteration, shuffle=False)

    #         points = gaussians.get_xyz
    #         # Create voxel grid with N dimensions in each direction NxNxN
    #         dvg = SparseVoxelGrid(points,  20, True)

    #         render_indices(scene, gaussians, m_args, dvg.get_points_at_tensor(vd1[entry[0]]).squeeze(), str(id))


    #         m_args.model_path = m_args.b
    #         gaussians = GaussianModel(m_args.sh_degree)
    #         scene = Scene(m_args, gaussians, load_iteration=args.iteration, shuffle=False)

    #         points = gaussians.get_xyz
    #         # Create voxel grid with N dimensions in each direction NxNxN
    #         dvg = SparseVoxelGrid(points,  20, True)

    #         render_indices(scene, gaussians, m_args, dvg.get_points_at_tensor(vd2[entry[1]]).squeeze(), str(id))






"""

Find dominant direction(s)
Around this direction create 8x8 grid and for each cell calculate histogram
Concatenate results to from feature descriptor


"""