import torch
from models.relight_utils import *
from shaders.shader import Shader


class Toon(Shader):

    # Toon shader
    def __init__(self, args, device):
        super().__init__(args, device)
        self.name = 'toon'
        self.ambience = args.ambience
        self.cutoff = args.toon_cutoff
        self.has_edges = True

    def __call__(self, mask, pos, depth, view, normal, albedo, roughness):

        toon = torch.ones_like(pos)
        l = super().light_dir(pos)
        ln = torch.sum(l * normal, dim=-1, keepdim=True)  # [bs, 1]
        surface_toon = self.ambience * albedo
        illuminated = (ln > self.cutoff).squeeze(-1)
        surface_toon[illuminated] = albedo[illuminated]

        # Colorspace transform
        if surface_toon.shape[0] > 0:
            surface_toon = linear2srgb_torch(surface_toon)
        toon[mask] = surface_toon[mask]

        return toon  # [bs, 3]

    def draw_edges(self, rgb, depth_edges, normal_edges):
        rgb = torch.clamp(
            rgb - normal_edges.unsqueeze(-1).expand(-1, 3) - depth_edges.unsqueeze(-1).expand(-1, 3), min=0.0)
        return rgb
