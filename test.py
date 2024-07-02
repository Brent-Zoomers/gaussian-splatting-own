import torch
import torch.nn.functional as F
def is_local_maximum(indices, grid):
    padded_tensor = F.pad(grid, (1, 1, 1, 1), mode='constant', value=float('-inf'))
    result = torch.zeros((2))
    
    for idx, i,j in enumerate(indices):
        result[idx] = (grid[i][j] < padded_tensor[i:i+3, j:j+3]).any()


        



random_init = torch.rand((100))

random_init_grid = random_init.view(10,10)

largest_v, largest_i = torch.topk(random_init, k=2)

i2 = largest_i % 10
i1 = (largest_i - i2) / 10

is_local_maximum(torch.cat((i1.int().unsqueeze(1), i2.int().unsqueeze(1)),dim=1), random_init_grid)