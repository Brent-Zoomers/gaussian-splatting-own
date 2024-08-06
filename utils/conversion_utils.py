import torch


# Formula to 3DGS
# XYZ -> -ZX-Y

# 3DGS to Formula
# XYZ -> Y-Z-X

# Assuming colmap coordinate system
# Nx2 to Nx3, required Phi,Theta
def convert_spherical_to_cartesian(input_data):
    phi, theta = input_data[...,0], input_data[...,1]
    x = torch.sin(phi) * torch.cos(theta)  #x
    y = torch.sin(phi) * torch.sin(theta)   #y
    z = torch.cos(phi)                     #z

    return torch.stack((y,-z, -x), dim=-1)

# Assuming colmap coordinate system
# Nx3 to Nx2, Requires XYZ in Colmap,3DGS coordinate system
def convert_cartesian_to_spherical(input_data):
    x,y,z = -input_data[...,2]+1e-10, input_data[...,0], -input_data[...,1]

    phi = torch.acos(z)
    theta = torch.acos(x / torch.sqrt(x**2 + y**2)) * y/torch.abs(y) # atan2 handles the quadrant issue

    return torch.stack((phi, theta), dim=-1)

w,h = 360, 360

theta = torch.linspace(-torch.pi, torch.pi, w).unsqueeze(1).expand(w,h)
phi = torch.linspace(0, torch.pi, h).unsqueeze(0).expand(w,h)

grid = torch.cat((phi.unsqueeze(2),theta.unsqueeze(2)), dim=2)
grid = grid.reshape((-1, 2))


cartesian = convert_spherical_to_cartesian(grid)
spherical = convert_cartesian_to_spherical(cartesian)

epsilon = 1e-5
torch.allclose(grid, spherical, atol=epsilon)
