import os

# Select two scenes, calculate features of both, and render one patch in red and the other in blue to same viewpoint


scene_1 = "output/random1"
scene_2 = "output/random2"

# Calculate features of each seperate scene

os.system(f'python test_3dgs_features.py -m {scene_1}')
os.system(f'python test_3dgs_features.py -m {scene_2}')

# Calculate features of both and match them using simple distance metric









