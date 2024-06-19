import os

datasets = os.listdir("datasets")


for dataset in datasets:
    os.system(f'python train.py -s datasets/{dataset} -m output/datasets/{dataset} --eval')