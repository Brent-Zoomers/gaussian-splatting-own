import torch
import numpy as np
from collections import defaultdict

class SparseVoxelGrid():
    def __init__(self, xyz, dims, store_indices=True):
        
        x_max, x_min = torch.max(xyz[...,0]),torch.min(xyz[...,0])
        y_max, y_min = torch.max(xyz[...,1]),torch.min(xyz[...,1])
        z_max, z_min = torch.max(xyz[...,2]),torch.min(xyz[...,2])

        x_range = torch.linspace(x_min, x_max, dims).cuda()
        x_masked = x_range.unsqueeze(1) >= xyz[...,0].unsqueeze(0).repeat(dims,1)
        x_indices = x_range.shape[0] - x_masked.count_nonzero(dim=0)

        y_range = torch.linspace(y_min, y_max, dims).cuda()
        y_masked = y_range.unsqueeze(1) >= xyz[...,1].unsqueeze(0).repeat(dims,1)
        y_indices = y_range.shape[0] - y_masked.count_nonzero(dim=0)

        z_range = torch.linspace(z_min, z_max, dims).cuda()
        z_masked = z_range.unsqueeze(1) >= xyz[...,2].unsqueeze(0).repeat(dims,1)
        z_indices = z_range.shape[0] - z_masked.count_nonzero(dim=0)

        stacked = torch.stack((x_indices, y_indices, z_indices))
        
        self.dim_per_point = stacked.permute(1,0)
        self.dims = dims
        # Convert indices to actual structure

        # unique_points, idx = torch.unique(dim_per_point, return_inverse=True, dim=0)

        # self.point_dict = {}

        # for id, value in enumerate(idx):
        #     key = unique_points[value]
        #     if key[0].item() not in self.point_dict:
        #         self.point_dict[key[0].item()] = {}
        #     if key[1].item() not in self.point_dict[key[0].item()]:
        #         self.point_dict[key[0].item()][key[1].item()] = {}
        #     if key[2].item() not in self.point_dict[key[0].item()][key[1].item()]:
        #         self.point_dict[key[0].item()][key[1].item()][key[2].item()] = []

        #     self.point_dict[key[0].item()][key[1].item()][key[2].item()].append(id)


        # f_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        

        # for id, value in enumerate(idx):
        #     key = unique_points[value]
        #     f_dict[key[0].item()][key[1].item()][key[2].item()].append(id)


        # print("DEBUG")

        # counter=0
        # for i in range(dims):
        #     exist_x = True
        #     if i not in self.point_dict:
        #         exist_x = False
        #     for j in range(dims):
        #         exist_y = exist_x
        #         if not exist_y or j not in self.point_dict[i]:
        #             exist_y = False
        #         for k in range(dims):
        #             exist_z = exist_y
        #             if not exist_z or k not in self.point_dict[i][j]:
        #                 print(i,j,k, "NOT")
        #             else:
        #                 print(i,j,k, "YES")
        #                 counter+= 1

        # print(counter / dims**3)

        # for entry in self.point_dict:
        #     for entry_1 in self.point_dict[entry]:
        #         for entry_2 in self.point_dict[entry][entry_1]:
        #             print(entry, entry_1, entry_2, len(self.point_dict[entry][entry_1][entry_2]))

        # print("DEBUG")
            
        
    def get_points_at_indices(self,x,y,z):  
        x = (self.dim_per_point == torch.tensor([x,y,z]).cuda()).all(dim=1)
        return torch.nonzero(x).squeeze(1)
    
    def get_points_at_tensor(self,tensor):  
        x = (self.dim_per_point == tensor.cuda()).all(dim=1)
        return torch.nonzero(x).squeeze(1)
    
    def get_all_occurances(self):
        occurances = torch.zeros((self.dims,self.dims,self.dims)).cuda().int()
        for i in range(self.dims):
            for j in range(self.dims):
                for k in range(self.dims):
                    occurances[i][j][k] = self.get_points_at_indices(i, j, k).shape[0]

        return occurances

# points = torch.rand((1000000,3)).repeat(2,1)
# dvg = SparseVoxelGrid(points, 10, True)