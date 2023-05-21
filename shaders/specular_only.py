import torch
from models.relight_utils import *
from shaders.shader import Shader


class SpecularOnly(Shader):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.name = 'specular'

    def __call__(self, mask, pos, depth, view, normal, albedo, roughness):
        specular_only = torch.ones_like(pos) / 2.0
        specular_only[mask] = super().blinn_phong_specular(view, normal, roughness, super().light_dir(pos))[mask]
        specular_only = torch.clamp(specular_only, min=0.0, max=1.0)
        return specular_only  # [bs, 3]
