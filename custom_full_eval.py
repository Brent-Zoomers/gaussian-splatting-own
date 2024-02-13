
import os
sh_degree = [0, 3]
reg_constant = [1e-3, 1e-4]



dirs  = os.listdir("datasets")
print(dirs)
for dir_name in dirs:
    print(dir_name)

    for rc in reg_constant:
        print(rc)
        os.system(f'python train.py -s datasets/{dir_name} -m output/{dir_name}_{rc} --reg_constant {rc} --eval')
        os.system(f'python render.py -m output/{dir_name}_{rc}')
        os.system(f'python metrics.py -m output/{dir_name}_{rc}')
    exit()