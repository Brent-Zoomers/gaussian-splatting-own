import os


#, #'combined_random', 'combined_random_masked'] #'random1', 'random2', 'combined_random'

# output_scenes = [f'top{x/10.0}percentcontributors_weighted_by_oc' for x in range(1,11)]
# output_scenes_2 = [f'top{x/10.0}percentcontributors' for x in range(1,11)] 
output_scenes = [f'mine_weighted_opacity/{x}' for x in range(1,6)] 


for scene in output_scenes:
    
    os.system(f'python render.py -m output/{scene}')
    os.system(f'python metrics.py -m output/{scene}')

