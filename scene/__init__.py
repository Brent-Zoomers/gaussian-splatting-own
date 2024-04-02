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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import numpy as np
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]


        # min_values = torch.empty(0)
        # scale_factor = torch.empty(0)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply")) 
            # loaded_tensors = []
            # with open(f'{args.model_path}/tensors.txt', 'r') as file:
            #     lines = file.readlines()
            #     for line in lines:
            #         loaded_tensors.append(torch.tensor(np.array(eval(line.strip()), dtype=np.float32)).cuda())
            # min_values = loaded_tensors[0]
            # scale_factor = loaded_tensors[1]

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

            # with torch.no_grad():

            #     min_values = torch.min(gaussians._xyz, dim=0)[0]
            #     max_values = torch.max(gaussians._xyz, dim=0)[0]
            #     print(min_values)
            #     print(max_values)
                
            #     # Shift all dimensions to make them positive
            #     gaussians._xyz = gaussians._xyz - torch.tensor(min_values).cuda()
            #     max_vals = torch.max(gaussians._xyz, dim=0)[0]
            #     # Compute the scale factor to fit the point cloud within the unit box
            #     scale_factor = (1.0 / torch.max(max_vals)) * 150
            #     # Scale the point cloud to fit within the unit box
            #     gaussians._xyz = gaussians._xyz * scale_factor

            #     with open(f'{args.model_path}/tensors.txt', 'w') as file:
            #         file.write(min_values.__repr__() + '\n')
            #         file.write(scale_factor.__repr__() + '\n')    


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args) #, (-min_values.cpu().numpy(), scale_factor.cpu().numpy())
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args) #, (-min_values.cpu().numpy(), scale_factor.cpu().numpy())

         
        # print("Here")
        
        #     # Compute the maximum value along each dimension
        #     max_vals = torch.max(gaussians._xyz, dim=0)[0]
            
        #     # Compute the scale factor to fit the point cloud within the unit box
        #     scale_factor = (1.0 / torch.max(max_vals))
            
        #     # Scale the point cloud to fit within the unit box
        #     # gaussians._xyz = shifted_point_cloud * scale_factor

        #     for resolution_scale in resolution_scales:
        #         for cam in self.train_cameras[resolution_scale]:
        #             pass
        #             # cam.camera_center -= min_values
        #             # cam.translate(min_values.cpu().numpy())
        #             # cam.T += min_values.cpu().numpy()
        #             # cam.T *= scale_factor.cpu().numpy()

        #         for cam in self.test_cameras[resolution_scale]:
        #             pass
        #             # cam.camera_center -= min_values
        #             # cam.translate(min_values.cpu().numpy())
        #             # cam.T += min_values.cpu().numpy()
        #             # cam.T *= scale_factor.cpu().numpy()


            
        # print(self.train_cameras[1.0][0].T)       
                

    def save(self, iteration, passed_time=0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        results_path = os.path.join(self.model_path, "amount_gaussians.txt")
        with open(results_path, 'a') as file:
            # Append the data to the file
            file.write(str(iteration) + ":" + str(self.gaussians.get_xyz.shape) + " Gaussians and " + str(passed_time) + " passed time\n")

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]