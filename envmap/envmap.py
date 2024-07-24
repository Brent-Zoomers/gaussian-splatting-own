import torch

def store_envmap(file_name, spherical_gaussians):
    torch.save(spherical_gaussians, file_name)


def load_envmap(file_name):
    envmap = torch.load(file_name)
    return envmap

# store_envmap("test.pt", x)
# y = load_envmap("test.pt")

# print(y)