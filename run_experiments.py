import os

scenes = ["truck", "train", "counter", "bicycle", "drjohnson", "kitchen"] 
iteration_options = [30_000]

for scene in scenes:
        os.system(f'python train.py -s datasets/{scene} --model_path final_eval/{scene} --iterations {30_000} --eval \
                  --reg_constant {1e-3} --beta_scaling_weight {1}')
        os.system(f'python render.py --model_path reg_run15k/{scene}')
        os.system(f'python metrics.py --model_path reg_run15k/{scene}')
        
# 20 * 3 * 2 = 60 * 15