import torch
import kornia.filters as K


def normalize_depth_map(depth_map, mask):
    depth_min = depth_map.min()
    depth_max = 2.0 * depth_map.max()
    ret = torch.full_like(depth_map, depth_max)
    ret[mask] = depth_map[mask]
    ret = (ret - depth_min) / (depth_max - depth_min)
    return ret


def modify_depth_map(depth_map, mask, scale):
    normalized_depth_map = normalize_depth_map(depth_map, mask)
    ret = 1.0 - normalized_depth_map
    ret = ret ** scale
    return ret


def modify_normal_map(normal_map, depth_map):
    return normal_map * depth_map.expand(-1, 3, -1, -1)


def normals(*args, **kwargs):
    # Edge detection based on normals
    H, W = kwargs['hw']
    normal_map = kwargs['normal_map']
    view_map = kwargs['view_map']
    mask = kwargs['mask']
    edge_map = torch.where(mask, torch.abs(torch.sum(normal_map * view_map, dim=-1)), torch.ones((H*W,)))  # [H*W]
    edge_mask = edge_map < args[0]
    return edge_map, edge_mask, torch.zeros_like(edge_map), torch.full_like(edge_mask, False)


def canny(*args, **kwargs):
    # Canny edge detection on depth and normals
    H, W = kwargs['hw']
    depth_map = kwargs['depth_map'].reshape(1, 1, H, W)
    normal_map = kwargs['normal_map'].permute(1, 0).reshape(1, 3, H, W)
    depth_map = modify_depth_map(depth_map, kwargs['mask'].reshape_as(depth_map), kwargs['scale'])
    normal_map = modify_normal_map(normal_map, depth_map)

    depth_edge_map, depth_edge_mask = K.canny(
        depth_map, low_threshold=args[0], high_threshold=args[1])  # [1, 1, H, W]
    depth_edge_map = depth_edge_map.view(-1)  # [H*W]
    depth_edge_mask = depth_edge_mask.bool().view(-1)  # [H*W]

    normal_edge_map, normal_edge_mask = K.canny(
        normal_map, low_threshold=args[2], high_threshold=args[3])  # [1, 1, H, W]
    normal_edge_map = normal_edge_map.view(-1)  # [H*W]
    normal_edge_mask = normal_edge_mask.bool().view(-1)  # [H*W]

    return depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask


def sobel(*args, **kwargs):
    # Sobel edge detection on depth and normals
    H, W = kwargs['hw']
    depth_map = kwargs['depth_map'].reshape(1, 1, H, W)
    normal_map = kwargs['normal_map'].permute(1, 0).reshape(1, 3, H, W)
    depth_map = modify_depth_map(depth_map, kwargs['mask'].reshape_as(depth_map), kwargs['scale'])
    normal_map = modify_normal_map(normal_map, depth_map)

    depth_edge_map = K.sobel(depth_map).view(-1)  # [H*W]
    depth_edge_mask = depth_edge_map > args[0]  # [H*W]
    normal_edge_map = K.sobel(normal_map).view(3, -1)  # [3, H*W]
    normal_edge_map = torch.sum(normal_edge_map, dim=0)  # [H*W]
    normal_edge_mask = normal_edge_map > args[1]  # [H*W]

    return depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask
