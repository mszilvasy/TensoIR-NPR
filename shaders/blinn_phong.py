import torch
from models.relight_utils import *
from shaders.shader import Shader


class BlinnPhong(Shader):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.name = 'blinn_phong'
        self.ambience = args.blinn_phong_ambience
        self.diffuse_intensity = args.blinn_phong_diffuse
        self.specular_intensity = args.blinn_phong_specular

    def __call__(self, mask, pos, depth, view, normal, albedo, roughness):

        blinn_phong = torch.zeros_like(pos)
        l = super().light_dir(pos)
        ln = torch.clamp(torch.sum(l * normal, dim=-1, keepdim=True), min=0.0)  # [bs, 1]
        ambient, diffuse = self.ambience * albedo, (1.0 - self.ambience) * albedo
        lambertian = self.light_rgb[None, :] * diffuse * ln  # [bs, 3]
        specular = super().blinn_phong_specular(view, normal, roughness, l)
        surface_blinn_phong = ambient + self.diffuse_intensity * lambertian + self.specular_intensity * specular

        # Tone-mapping
        surface_blinn_phong = torch.clamp(surface_blinn_phong, min=0.0, max=1.0)
        # Colorspace transform
        if surface_blinn_phong.shape[0] > 0:
            surface_blinn_phong = linear2srgb_torch(surface_blinn_phong)
        blinn_phong[mask] = surface_blinn_phong[mask]

        return blinn_phong  # [bs, 3]
