import torch
import kornia.filters as K


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
    depth_map = torch.where(kwargs['mask'].reshape(1, 1, H, W), depth_map, 2.0 * torch.max(depth_map))

    depth_edges_raw, depth_edges = K.canny(
        depth_map, low_threshold=args[0], high_threshold=args[1])  # [1, 1, H, W]
    depth_edges_raw = depth_edges_raw.view(-1)  # [H*W]
    depth_edges = depth_edges.view(-1)  # [H*W]

    normal_edges_raw, normal_edges = K.canny(
        normal_map, low_threshold=args[2], high_threshold=args[3])  # [1, 1, H, W]
    normal_edges_raw = normal_edges_raw.view(-1)  # [H*W]
    normal_edges = normal_edges.view(-1)  # [H*W]

    return depth_edges_raw, depth_edges, normal_edges_raw, normal_edges


def sobel(*args, **kwargs):
    # Sobel edge detection on depth and normals
    H, W = kwargs['hw']
    depth_map = kwargs['depth_map'].reshape(1, 1, H, W)
    normal_map = kwargs['normal_map'].permute(1, 0).reshape(1, 3, H, W)
    depth_map = torch.where(kwargs['mask'].reshape(1, 1, H, W), depth_map, torch.max(depth_map) + args[0])

    depth_edges_raw = K.sobel(depth_map).view(-1)  # [H*W]
    depth_edge_mask = depth_edges_raw > args[0]  # [H*W]
    depth_edges = torch.where(depth_edge_mask, torch.ones_like(depth_edges_raw), torch.zeros_like(depth_edges_raw))
    normal_edges_raw = K.sobel(normal_map).view(3, -1)  # [3, H*W]
    normal_edges_raw = torch.sum(normal_edges_raw, dim=0)  # [H*W]
    normal_edge_mask = normal_edges_raw > args[1]  # [H*W]
    normal_edges = torch.where(normal_edge_mask, torch.ones_like(normal_edges_raw), torch.zeros_like(normal_edges_raw))

    return depth_edges_raw, depth_edges, normal_edges_raw, normal_edges
