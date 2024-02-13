import os


dataset = 'tandt_db/tandt/truck'




# os.system(f'python train.py -s {dataset} -m output/inv_log_start{1e-3}_end{1e-6}_clamped --reg_constant {0} --eval')

# os.system(f'python render.py -m output/0')
# os.system(f'python metrics.py -m output/0')

# os.system(f'python train.py -s {dataset} -m output/gt_shdegree_0 --reg_constant {0} --sh_degree 0 --eval ')
# os.system(f'python render.py -m output/gt_shdegree_0')
# os.system(f'python metrics.py -m output/gt_shdegree_0')

# os.system(f'python render.py -m output/0.001_shdegree_0')
# os.system(f'python metrics.py -m output/0.001_shdegree_0'
# )

# am_biggest = [2560358,2194662,1949544,1586624,1184790]

# for entry in am_biggest:
    #  print(entry)

os.system(f'python train.py -s -m output/degree0 --sh_degree 0 --eval')

# for i in range(-10, 1, 1):
#     # j=i/2.0
#     j=i
#     print(1 * 10**j)
#     # os.system(f'python train.py -s {dataset} -m output/v2_{1 * 10**j} --reg_constant {1 * 10**j} --sh_degree 3 --eval ')
#     os.system(f'python render_n_biggest.py -m output/{0} --amount_biggest ')