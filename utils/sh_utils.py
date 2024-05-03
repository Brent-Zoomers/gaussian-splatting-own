#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

# class SphericalGaussian():
#     def __init__(self, alpha_, lambda_, mu_):
#         self.alpha_ = alpha_
#         self.lambda_ = lambda_
#         self.mu_ = mu_

#     def gaussian_integral_of_product(self, g2):
#             d_m = torch.norm(self.lambda_*self.mu_+g2.lambda_*g2.mu_)
#             lambda_m = self.lambda_ + g2.lambda_
#             return 2*torch.pi*self.alpha_*g2.alpha_ * ((torch.exp(d_m-lambda_m)-torch.exp(-d_m-lambda_m))/(d_m))

# ...x3
def dot_product(t1, t2):
    # r1 = torch.rand_like(t1)
    # r2 = torch.rand_like(t2)
    res = t1 * t2
    return torch.sum(res, dim=2)

# Calculate product of Spherical Gaussians
def calculate_product(g1, g2):
    pass

def calculate_inner_product(g1, g2):
    pass


def eval_sg_env(diff, spec, env_map, dirs, normals):
    """
    diff: Nx3x2 --> alpha
    spec: Nx3x2 -> alpha, lambda
    env_map: Mx7 --> alpha, lambda, dir
    dirs: Kx3
    normals: Nx3

    out: Nx3
    """
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    env_dirs = env_map[...,0:3] / torch.norm(env_map[...,0:3], dim=0)
    env_alphas = torch.sigmoid(env_map[...,3:6])
    env_lambdas = env_map[...,6]
    dirs = -dirs / torch.norm(dirs, dim=0)
    # Diffuse part

    # Point toward normal with lambda static and alpha taken from {diff}
   
    splat_alphas = torch.sigmoid(diff[...,0:3])
    splat_lambdas = torch.ones((splat_alphas.shape[0])).cuda().float() * 0.1

    expanded_splat_lambdas = splat_lambdas.unsqueeze(1).repeat(1,env_lambdas.shape[0])

    lambda_m = expanded_splat_lambdas + env_lambdas

    l1mulmu1 = (splat_lambdas.unsqueeze(1) * normals).unsqueeze(1).repeat(1,env_lambdas.shape[0],1)

    d_m = torch.norm(l1mulmu1 + env_lambdas.unsqueeze(1)*env_dirs, dim=2)

    expanded_alphas = splat_alphas.unsqueeze(1).repeat(1,env_lambdas.shape[0],1)

    fraction = (torch.exp(d_m - lambda_m) - torch.exp(-d_m-lambda_m)) / d_m
    result = 2*torch.pi*expanded_alphas*env_alphas * fraction.unsqueeze(2)

    # Specular part

    # spec_splat_alphas = spec[...,0:3]
    # spec_splat_lambdas = spec[...,3]

    # # Calc mu of lobe as mirror of input around normal

    # mirrored = dirs - 2* torch.sum(dirs*normals) * normals

    # spec_expanded_splat_lambdas = spec_splat_lambdas.unsqueeze(1).repeat(1,env_lambdas.shape[0])

    # spec_lambda_m = spec_expanded_splat_lambdas + env_lambdas

    # spec_l1mulmu1 = (spec_splat_lambdas.unsqueeze(1) * mirrored).unsqueeze(1).repeat(1,env_lambdas.shape[0],1)

    # spec_d_m = torch.norm(spec_l1mulmu1 + env_lambdas.unsqueeze(1)*env_dirs, dim=2)

    # spec_expanded_alphas = spec_splat_alphas.unsqueeze(1).repeat(1,env_lambdas.shape[0],1)

    # spec_fraction = (torch.exp(spec_d_m - spec_lambda_m) - torch.exp(-spec_d_m-spec_lambda_m)) / spec_d_m
    # spec_result = 2*torch.pi*spec_expanded_alphas*env_alphas * spec_fraction.unsqueeze(2)

    return torch.sigmoid(torch.sum(result, dim=1))

# N x 3 x 5
def eval_sg(params, dirs):
    
    alphas = params[...,0]
    lambdas = params[...,1]
    directions = params[...,2:5]

    directions_normalized = directions/directions.norm(dim=1, keepdim=True)

    rgb = alphas * torch.exp(lambdas * (dot_product(directions_normalized, dirs.unsqueeze(1))-1.0) )

    return rgb

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5