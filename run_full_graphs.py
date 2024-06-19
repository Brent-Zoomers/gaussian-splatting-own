import os
import shutil
import matplotlib.pyplot as plt
import json


datasets = os.listdir("datasets")


for dataset in datasets:
    PSNR = []
    PSNR1 = []
    for x in range(1,6):
        
        current_metric = 'SSIM'
        try:
            file_name = f'output/datasets/{dataset}/{x}/results.json'
            f = open(file_name)
            data = json.load(f)
            PSNR.append(data['ours_30000'][current_metric])
        except:
            pass

        try:
            file_name = f'output/datasets/{dataset}_webv/{x}/results.json'
            f = open(file_name)
            data = json.load(f)
            PSNR1.append(data['ours_30000'][current_metric])
        except:
            pass
        
    plt.plot(PSNR, label=f'{dataset}')
    plt.plot(PSNR1, label=f'{dataset}_webv')
    print(PSNR)

    plt.legend()
    # plt.ylim(0.45,0.9)
    # plt.xticks(np.arange(0, 6, 1))
    plt.xlabel("Percentage splats")
    plt.ylabel(current_metric)
    plt.show()