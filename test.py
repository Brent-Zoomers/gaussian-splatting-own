import torch

ids_per_pixel = torch.tensor([0, 1, 1, 2, 2, 2]).cuda()

total_occ = torch.zeros(3, device='cuda', dtype=torch.long)
total_occ_lol = torch.zeros(3, device='cuda', dtype=torch.long)

for i in range(20):

    # set = torch.unique(ids_per_pixel)  # set = [0, 1, 2]
    total_occ.index_add_(0, ids_per_pixel, torch.ones_like(ids_per_pixel) * 1)

    total_occ_lol[ids_per_pixel] += 1


print(total_occ)
print(total_occ_lol)
