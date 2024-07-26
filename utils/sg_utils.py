import torch
# def gaussian_integral_of_product(self, g1:SphericalGaussian, g2:SphericalGaussian):
#         d_m = torch.norm(g1.lambda_*g1.mu_+g2.lambda_*g2.mu_)
#         lambda_m = g1.lambda_ + g2.lambda_
#         return 2*torch.pi*g1.alpha_*g2.alpha_ * ((torch.exp(d_m-lambda_m)-torch.exp(-d_m-lambda_m))/(d_m))


def calculate_colors(gaussian_data, normal_data, envmap_data):
    """
    gaussian_data: Nx3x2
    normal_data: Nx3
    envmap_data: Gx3x5

    Apply all Gs to all Ns
    """
    g_a = gaussian_data[...,0].unsqueeze(1)

    # g_a = g_l / (2 * torch.pi * (1.0 - torch.exp(-2*g_l)))

    em_a = envmap_data[...,3]
    g_l = gaussian_data[...,1].unsqueeze(1)
    em_l = envmap_data[...,4]
    g_m = normal_data.unsqueeze(1).repeat(1,3,1).unsqueeze(1)
    em_m = envmap_data[...,0:3]

    # g_a = g_l / (2 * torch.pi * (1.0 - torch.exp(-2*g_l)))

    g_m = g_m / torch.norm(g_m, dim=-1, keepdim=True)
    em_m = em_m / torch.norm(em_m, dim=-1, keepdim=True)


    d_m = torch.norm(g_l.unsqueeze(-1) * g_m + em_l.unsqueeze(-1) * em_m, dim=3)
    l_m = g_l + em_l
    alphas = g_a * em_a

    fraction = torch.exp(d_m - l_m) - torch.exp(-d_m-l_m) / (d_m + 1e-10)


    return torch.sum(2.0 * torch.pi * alphas * fraction, dim=1)
