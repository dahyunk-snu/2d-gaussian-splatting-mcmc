from diff_surfel_rasterization import compute_relocation
import torch
import math

N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
for n in range(N_MAX):
    for k in range(n + 1):
        BINOMS[n, k] = math.comb(n, k)


# https://github.com/hwanhuh/diff-surfel-rasterization-MCMC.git
def compute_relocation_cuda(
    opacities,  # [N]
    scales,  # [N, 2]
    ratios,  # [N]
):
    """
    Computes new opacities and scales using the MCMC relocation kernel.

    Args:
        opacities (torch.Tensor): Array of opacities for each Gaussian splat.
        scales (torch.Tensor): Array of scales for each Gaussian splat.
        ratios (torch.Tensor): Array of ratios used in relocation computation.

    Returns:
        new_opacities (torch.Tensor): Updated opacities after relocation.
        new_scales (torch.Tensor): Updated scales after relocation.
    """
    N = opacities.shape[0]
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=N_MAX)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, BINOMS, N_MAX
    )
    return new_opacities, new_scales
