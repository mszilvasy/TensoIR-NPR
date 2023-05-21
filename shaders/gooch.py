import torch
from models.relight_utils import *
from shaders.shader import Shader


class Gooch(Shader):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.name = 'gooch'
        self.k_blue = torch.Tensor([0.0, 0.0, args.gooch_b]).to(device)
        self.k_yellow = torch.Tensor([args.gooch_y, args.gooch_y, 0.0]).to(device)
        self.alpha = args.gooch_alpha
        self.beta = args.gooch_beta
        self.specular_intensity = args.gooch_specular

    def __call__(self, mask, pos, depth, view, normal, albedo, roughness):

        gooch = torch.zeros_like(pos)
        l = super().light_dir(pos)
        cos = torch.sum(normal[mask] * l[mask], dim=-1, keepdim=True)  # [bs, 1]
        interp = (1.0 + cos) / 2.0  # [bs, 1]

        k_cool = self.k_blue[None, :] + self.alpha * albedo[mask]
        k_warm = self.k_yellow[None, :] + self.beta * albedo[mask]
        surface_gooch = self.light_rgb[None, :] * (interp * k_warm + (1.0 - interp) * k_cool)  # [bs, 3]
        if self.specular_intensity > 0.0:
            specular = super().blinn_phong_specular(view, normal, roughness, l)
            surface_gooch += self.specular_intensity * specular[mask]

        # Tone-mapping
        surface_gooch = torch.clamp(surface_gooch, min=0.0, max=1.0)
        # Colorspace transform
        if surface_gooch.shape[0] > 0:
            surface_gooch = linear2srgb_torch(surface_gooch)
        gooch[mask] = surface_gooch

        return gooch  # [bs, 3]
