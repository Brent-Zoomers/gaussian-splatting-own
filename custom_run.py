import os


dataset = 'tandt_db/tandt/truck'

# os.system(f'python train.py -s {dataset} -m output/inv_log_start{1e-3}_end{1e-6}_clamped --reg_constant {0} --eval')

# os.system(f'python render.py -m output/0')
# os.system(f'python metrics.py -m output/0')

for i in range(-10,-4, 1):
    j=i/2.0
    print(1 * 10**j)
    os.system(f'python train.py -s {dataset} -m output/v2_{1 * 10**j} --reg_constant {1 * 10**j} --eval')