import torch.nn as nn
import torch.nn.functional as F
import torch

class SH(nn.Module):
    def __init__(self, batch_size, weights):
        super().__init__()
        # Setup SH functions
        self.weights = weights
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.coefficients = torch.tensor([0.282095,0.488602,0.488602,0.488602,1.092548,1.092548,0.315391,1.092548,0.546274], device="cuda")
        self.batch_size = batch_size
   
    # dir = Nx3
    
    def forward(self, direction):
        # N
        x, y, z = direction.unbind(1)

        # assumes r = 1
        # B x 9
        self.constants = self.weights * self.coefficients

        # B x 40
        result = torch.zeros((self.batch_size, x.shape[0]), device="cuda")

        self.constants = self.constants.unsqueeze(1)

        # band 0
        result += self.constants[:,:,0]

        #band 1
        result += self.constants[:,:,1] * y
        result += self.constants[:,:,2] * z
        result += self.constants[:,:,3] * x

        #band 2
        result += self.constants[:,:,4] * x*y
        result += self.constants[:,:,5] * x*z
        result += self.constants[:,:,6] * 3*(z*z)-1
        result += self.constants[:,:,7] * x*z
        result += self.constants[:,:,8] * x*x-y*y

        return result

# sh = SH(torch.rand(9))


# print(sh(torch.tensor([0.0,1.0,0.0])))