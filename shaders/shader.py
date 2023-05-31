import torch

from models.relight_utils import *


class Shader:
    def __init__(self, args, device):
        self.light_type = args.light
        self.light_pos = torch.Tensor(args.light_pos).to(device)
        self.light_rgb = torch.Tensor(args.light_rgb).to(device)
        self.shininess = args.shininess

        if self.light_type == 'infinity':
            self.light_pos = safe_l2_normalize(self.light_pos, dim=-1)

    def light_dir(self, pos):
        if self.light_type == 'infinity':
            return self.light_pos[None, :]
        elif self.light_type == 'point':
            return safe_l2_normalize(self.light_pos[None, :] - pos, dim=-1)

    def blinn_phong_specular(self, view, normal, roughness, l):
        mask = torch.sum(normal * l, dim=-1) >= 0.0
        h = safe_l2_normalize(l + view, dim=-1)
        cos = torch.zeros_like(normal)
        cos[mask] = torch.clamp(torch.sum(normal * h, dim=-1, keepdim=True), min=0.0)[mask]  # [bs, 1]
        return self.light_rgb[None, :] * (1.0 - roughness) * (cos ** self.shininess)

    def draw_edges(self, rgb, depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask):
        return rgb
