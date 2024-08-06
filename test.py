import torch

# Formula to 3DGS
# XYZ -> -ZX-Y

# 3DGS to Formula
# XYZ -> Y-Z-X

# Assuming colmap coordinate system
# Nx2 to Nx3
def convert_spherical_to_cartesian(input_data):
    phi, theta = input_data[..., 0], input_data[..., 1]
    x = torch.sin(phi) * torch.cos(theta)  # x
    y = torch.sin(phi) * torch.sin(theta)  # y
    z = torch.cos(phi)                     # z

    return torch.stack((y, -z, -x), dim=-1)

# Assuming colmap coordinate system
# Nx3 to Nx2
def convert_cartesian_to_spherical(input_data):
    x, y, z = -input_data[..., 2], input_data[..., 0], -input_data[..., 1]

    phi = torch.acos(z)
    theta = torch.atan2(y, x)  # atan2 handles the quadrant issue

    return torch.stack((phi, theta), dim=-1)

# Define the grid
w, h = 360, 360

theta = torch.linspace(0, 2*torch.pi, w).unsqueeze(1).expand(w, h)
phi = torch.linspace(0, torch.pi, h).unsqueeze(0).expand(w, h)

grid = torch.cat((phi.unsqueeze(2), theta.unsqueeze(2)), dim=2)
grid = grid.reshape((-1, 2))

# Convert to Cartesian coordinates
cartesian = convert_spherical_to_cartesian(grid)
print("Cartesian coordinates:", cartesian)

# Convert back to Spherical coordinates
spherical = convert_cartesian_to_spherical(cartesian)
print("Spherical coordinates:", spherical)

# Check if the conversion is correct by comparing the original and converted spherical coordinates
# Allow for some numerical precision error
epsilon = 1e-5
if torch.allclose(grid, spherical, atol=epsilon):
    print("Conversion between spherical and Cartesian coordinates is consistent.")
else:
    print("Conversion error detected!")
    # Print some mismatches for debugging
    for i in range(grid.shape[0]):
        if not torch.allclose(grid[i], spherical[i], atol=epsilon):
            print(f"Original: {grid[i]}, Converted: {spherical[i]}")
            break
