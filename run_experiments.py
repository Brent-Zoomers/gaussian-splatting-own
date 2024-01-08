import os


# reg = [1e8, 1e9, 1e10, 1e6, 1e5]
iteration_options = [30_000]


for op3 in iteration_options:
    os.system(f'python train.py -s tandt_db/tandt/truck --model_path output/r_dynamicdiv10000_{op3} --iterations {op3} --eval')
    os.system(f'python render.py --model_path output/r_dynamicdiv10000_{op3}')
    os.system(f'python metrics.py --model_path output/r_dynamicdiv10000_{op3}')
        
# 20 * 3 * 2 = 60 * 15