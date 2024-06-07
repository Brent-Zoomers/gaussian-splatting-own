import os



output_scenes = ['combined_random', 'combined_random_masked'] #'random1', 'random2', 'combined_random'



for scene in output_scenes:
    os.system(f'python render.py -m output/{scene}')
    os.system(f'python metrics.py -m output/{scene}')