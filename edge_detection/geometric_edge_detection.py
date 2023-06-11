import torch


def line_plane_intersection(o, d, p, n):
    ret = p - o  # [H, W, 3]
    ret = torch.sum(ret * n, dim=-1)  # [H, W]
    ret = ret / torch.sum(d * n, dim=-1)  # [H, W]
    return ret


def geometric(*args, **kwargs):
    # Edge detection based on surface inconsistencies between adjacent pixels
    # This works very poorly, and is not used in the paper
    H, W = kwargs['hw']
    rays = kwargs['rays'].reshape(H, W, 6)  # [H, W, 6]
    rays_o = rays[:, :, 0:3]  # [H, W, 3]
    rays_d = rays[:, :, 3:6]  # [H, W, 3]
    depth_map = kwargs['depth_map'].to(rays.device).reshape(H, W)  # [H, W]
    normal_map = kwargs['normal_map'].to(rays.device).reshape(H, W, 3)  # [H, W, 3]
    pos = kwargs['xyz_map'].to(rays.device).reshape(H, W, 3)  # [H, W, 3]
    mask = kwargs['mask'].to(rays.device)  # [H*W]

    left = (slice(None), slice(None, W-1))
    right = (slice(None), slice(1, None))
    up = (slice(None, H-1), slice(None))
    down = (slice(1, None), slice(None))

    plane_dist_left = line_plane_intersection(rays_o[left], rays_d[left], pos[right], normal_map[right])  # [H, W-1]
    plane_dist_right = line_plane_intersection(rays_o[right], rays_d[right], pos[left], normal_map[left])  # [H, W-1]
    plane_dist_up = line_plane_intersection(rays_o[up], rays_d[up], pos[down], normal_map[down])  # [H-1, W]
    plane_dist_down = line_plane_intersection(rays_o[down], rays_d[down], pos[up], normal_map[up])  # [H-1, W]

    displacement_left = plane_dist_left - depth_map[left]  # [H, W-1]
    displacement_right = plane_dist_right - depth_map[right]  # [H, W-1]
    displacement_up = plane_dist_up - depth_map[up]  # [H-1, W]
    displacement_down = plane_dist_down - depth_map[down]  # [H-1, W]

    inconsistency_x = torch.logical_or(displacement_left.abs() > 0.025, displacement_right.abs() > 0.025)  # [H, W-1]
    # inconsistency_x = torch.logical_and(displacement_left.sign() != displacement_right.sign(), inconsistency_x)
    inconsistency_y = torch.logical_or(displacement_up.abs() > 0.025, displacement_down.abs() > 0.025)  # [H-1, W]
    # inconsistency_y = torch.logical_and(displacement_up.sign() != displacement_down.sign(), inconsistency_y)

    depth_edge_map = torch.zeros_like(depth_map)  # [H, W]
    depth_edge_map[left] += displacement_left.abs()
    depth_edge_map[right] += displacement_right.abs()
    depth_edge_map[up] += displacement_up.abs()
    depth_edge_map[down] += displacement_down.abs()

    depth_edge_mask = torch.full((H, W), False, device=rays.device)  # [H, W]
    # depth_edge_mask[left] = torch.logical_and(inconsistency_x, depth_map[left] > depth_map[right])
    # depth_edge_mask[right] = torch.logical_and(inconsistency_x, depth_map[right] > depth_map[left])
    # depth_edge_mask[up] = torch.logical_and(inconsistency_y, depth_map[up] > depth_map[down])
    # depth_edge_mask[down] = torch.logical_and(inconsistency_y, depth_map[down] > depth_map[up])
    depth_edge_mask = depth_edge_map > 0.025
    depth_edge_map = depth_edge_map.clamp(max=1.0).view(-1).cpu()  # [H*W]
    depth_edge_mask = torch.logical_and(depth_edge_mask.view(-1), mask).cpu()  # [H*W]

    normal_change_x = torch.sum(normal_map[left] * normal_map[right], dim=-1)  # [H, W-1]
    normal_change_y = torch.sum(normal_map[up] * normal_map[down], dim=-1)  # [H-1, W]

    normal_edge_mask = torch.full((H, W), False, device=rays.device)  # [H, W]
    normal_edge_mask[left] = torch.logical_and(normal_change_x < 0.8, depth_map[left] >= depth_map[right])
    normal_edge_mask[right] = torch.logical_and(normal_change_x < 0.8, depth_map[right] >= depth_map[left])
    normal_edge_mask[up] = torch.logical_and(normal_change_y < 0.8, depth_map[up] >= depth_map[down])
    normal_edge_mask[down] = torch.logical_and(normal_change_y < 0.8, depth_map[down] >= depth_map[up])
    normal_edge_mask = torch.logical_and(normal_edge_mask.view(-1), mask).cpu()  # [H*W]

    #normal_edge_mask = torch.full((H*W,), False)  # [H*W]
    normal_edge_map = torch.zeros((H*W,))  # [H*W]

    return depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask
