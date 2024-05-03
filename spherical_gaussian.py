import torch
import cv2
import numpy as np

class SphericalGaussian():

    def __init__(self, a, l, m):
        """
        Expects m to be unit vectors
        """
        self.a = a
        self.l = l
        self.m = m

    def get_value_in_dir(self, v):
        """
        Expects v to be unit vectors
        """
        dot_products = torch.sum(self.m * v, dim=2)

        return torch.sigmoid(self.a) * torch.exp(self.l * ((dot_products) - 1.0))
    
    def to_equirectangular(self, w, h):
        """"
        return HxWx1
        """
        theta = torch.linspace(0, 2*torch.pi, w).unsqueeze(1).expand(w,h).cuda()
        phi = torch.linspace(0, torch.pi, h).unsqueeze(0).expand(w,h).cuda()

        grid = torch.cat((theta.unsqueeze(2),phi.unsqueeze(2)), dim=2)

        thetas, phis = grid[...,0], grid[...,1]
        
        z = -torch.sin(phis) * torch.cos(thetas)
        x = -torch.sin(phis) * torch.sin(thetas)
        y = -torch.cos(phis)

        dirs = -torch.cat((x.unsqueeze(2),y.unsqueeze(2),z.unsqueeze(2)), dim=2)

        result = self.get_value_in_dir(dirs)
        return result.permute(1,0).unsqueeze(2)
  
# gaussian_parameters = [[0.5,50,0,0,-1],[0.5, 2000, 1,0,0],[1, 20, 1,1,0],[0.1, 0.1, 0,0,-1]]
# result = torch.zeros((500,800,1))

# for element in gaussian_parameters:
#     normmalized_dirs = torch.tensor(element[2:5]).float() / torch.tensor(element[2:5]).float().norm()
#     sg = SphericalGaussian(element[0], element[1], normmalized_dirs   )

#     result += sg.to_equirectangular(800, 500)

# np_result = result.numpy()
# cv2.imshow("result", np_result)
# cv2.waitKey()




