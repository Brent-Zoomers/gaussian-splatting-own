import os
import shutil


datasets = os.listdir("datasets")


for dataset in datasets:
    # dataset_name = dataset.model_path.split("/")[-1]
    # output_folder_path = f'output/datasets/{dataset_name}'


    # shutil.copy(dataset.model_path+'/cameras.json', output_folder_path)
    # shutil.copy(dataset.model_path+'/cfg_args', output_folder_path)
    
    os.system(f'python filter_low_impact_splats.py -m output/datasets/{dataset}')
    for x in range(1,6):
        os.system(f'python render.py -m output/datasets/{dataset}_webv/{x}')
        os.system(f'python metrics.py -m output/datasets/{dataset}_webv/{x}')