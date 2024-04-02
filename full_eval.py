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
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle"]
mipnerf360_indoor_scenes = ["counter", "kitchen"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

sh_degree = [0, 3]
reg_constant = [1e-3, 1e-4, 0]
    
failed_runs = [
    ("drjohnson", 0.0001, 3)
  ]

# if not args.skip_training:
#     # for sh in sh_degree:
#     #     for rc in reg_constant:
#     for entry in failed_runs:
#         sh = entry[2]
#         rc = entry[1]
#         scene_n = entry[0]

#         common_args = f' --eval --test_iterations -1 --reg_constant {rc} --sh_degree {sh}'
#         path = f'full_eval_{sh}_{rc}'
#         for scene in mipnerf360_outdoor_scenes:
#             if scene == scene_n:
#                 source = args.mipnerf360 + "/" + scene
#                 os.system("python train.py -s " + source + " -i images_4 -m eval/" + path + "/" + scene + common_args)
#         for scene in mipnerf360_indoor_scenes:
#             if scene == scene_n:
#                 source = args.mipnerf360 + "/" + scene
#                 os.system("python train.py -s " + source + " -i images_2 -m eval/" + path + "/" + scene + common_args)
#         for scene in tanks_and_temples_scenes:
#             if scene == scene_n:
#                 source = args.tanksandtemples + "/" + scene
#                 os.system("python train.py -s " + source + " -m eval/" + path + "/" + scene + common_args)
#         for scene in deep_blending_scenes:
#             if scene == scene_n:
#                 source = args.deepblending + "/" + scene
#                 os.system("python train.py -s " + source + " -m eval/" + path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for sh in sh_degree:
        for rc in reg_constant:

    # for entry in failed_runs:
    #     sh = entry[2]
    #     rc = entry[1]
    #     scene_n = entry[0]
            path = f'full_eval_{sh}_{rc}'
            for scene, source in zip(all_scenes, all_sources):
                # os.system("python render.py --iteration 7000 -s " + source + " -m eval/" + path + "/" + scene + common_args)
                os.system("python render.py --iteration 30000 -s " + source + " -m eval/" + path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for sh in sh_degree:
        for rc in reg_constant:
            path = f'full_eval_{sh}_{rc}'
            for scene in all_scenes:
                scenes_string += "\"" + path + "/" + scene + "\" "

                os.system("python metrics.py -m eval/" + path + "/" + scene)
                
