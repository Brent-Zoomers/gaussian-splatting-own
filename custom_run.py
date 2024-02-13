import os

dataset = "truck_big"
skip_frame = [3]
skip_segment = [2,3,4]
densify_viewpoint_iterations = [2500]


os.system(f'python train.py \
                        --source_path {dataset} \
                        --images images_4 \
                        --stdev_threshold_skipped {3} \
                        --stdev_threshold_segment {4} \
                        --densify_viewpoint_iterations {2500} \
                        --model_path output/testv3{3}_{4} \
                        --eval')

# for skip_f in skip_frame:
#     for skip_s in skip_segment:
#         for densify_v in densify_viewpoint_iterations:
#             # print(densify_v, skip_f, skip_s)
#             # if skip_f == 1 and skip_s == 2 and
#             os.system(f'python train.py -s {dataset} -i images_4 \
#                       --stdev_threshold_skipped {skip_f} \
#                       --stdev_threshold_segment {skip_s} \
#                       --densify_viewpoint_iterations {int(densify_v)} \
#                       -m output/v3{skip_f}_{skip_s} \
#                       --eval') 
            
            #    
                   
# for skip_f in skip_frame:
#     for skip_s in skip_segment:
#         for densify_v in densify_viewpoint_iterations:
#             # print(densify_v, skip_f, skip_s)
#             # if skip_f == 1 and skip_s == 2 and
#             os.system(f'python render.py -m output/v3{skip_f}_{skip_s}') 
#             os.system(f'python metrics.py -m output/v3{skip_f}_{skip_s}') 


# import json
# import matplotlib.pyplot as plt
# PSNR = []
# SSIM = []
# LPIPS = []
# labels = []             
# for skip_f in skip_frame:
#     for skip_s in skip_segment:
#         for densify_v in densify_viewpoint_iterations:
#             if os.path.isdir(f'output/{skip_f}_{skip_s}_{densify_v}'):
#                 json_file = open(f'output/{skip_f}_{skip_s}_{densify_v}/results.json')
#                 stats = json.load(json_file)
#                 print(stats['ours_30000'])
#                 PSNR.append(stats['ours_30000']['PSNR'])
#                 LPIPS.append(stats['ours_30000']['LPIPS'])
#                 SSIM.append(stats['ours_30000']['SSIM'])
#                 labels.append(f'{skip_f}_{skip_s}_{densify_v}')


# plt.bar(labels, PSNR)
# plt.ylim(25)
# # Add labels and title
# plt.xlabel('labels')
# plt.ylabel('values')
# plt.title('Bar Chart Example')
# # Show plot
# plt.show()

# plt.bar(labels, LPIPS)

# # Add labels and title
# plt.xlabel('labels')
# plt.ylabel('values')
# plt.title('Bar Chart Example')
# # Show plot
# plt.show()



# plt.bar(labels, SSIM)
# # Add labels and title
# plt.xlabel('labels')
# plt.ylabel('values')
# plt.title('Bar Chart Example')
# # Show plot
# plt.show()



            # print(densify_v, skip_f, skip_s)
            # if skip_f == 1 and skip_s == 2 and
            # os.system(f'python render.py -m output/{skip_f}_{skip_s}') 
            # os.system(f'python metrics.py -m output/{skip_f}_{skip_s}_{densify_v}') 