import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataLoader import dataset_dict
from models.tensoRF_rotated_lights import TensorVMSplit
from opt import config_parser
from utils import visualize_depth_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScanEdges(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kwargs):
        super().__init__(aabb, gridSize, device, **kwargs)

    def __call__(self, rays, wh,
                 decay=-1,
                 acc_threshold=0.8,  # positive: use acc thresholding, negative: use sigma thresholding
                 inclusion_threshold=0.1,
                 track_lines=False,
                 near_weight=1.0,
                 far_weight=1.0,
                 line_style=0,  # 0: euclidean, 1: manhattan
                 image_dir=None,
                 save_intermediates=False,
                 visualize_depth=None):
        """
        :param rays: [H*W, 6]
        :param wh: (W, H)
        :return: [H, W, 1]
        """
        save_results = image_dir is not None
        if save_results:
            video_path = os.path.join(image_dir, 'video')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(video_path, exist_ok=True)
            if save_intermediates:
                scan_path = os.path.join(image_dir, 'scan')
                edge_path = os.path.join(image_dir, 'edge')
                os.makedirs(scan_path, exist_ok=True)
                os.makedirs(edge_path, exist_ok=True)
        else:
            scan_path, edge_path, video_path = None, None, None
        scans, outlines = [], []

        W, H = wh
        red_torch = torch.zeros(H, W, 3).to(rays.device)
        red_torch[:, :, 0] = 1.0
        red_numpy = np.zeros((H, W, 3), dtype=np.uint8)
        red_numpy[:, :, 0] = 255
        green_torch = torch.zeros(H, W, 3).to(rays.device)
        green_torch[:, :, 1] = 1.0
        yellow_numpy = np.zeros((H, W, 3), dtype=np.uint8)
        yellow_numpy[:, :, 0:2] = 255
        orange_numpy = np.zeros((H, W, 3), dtype=np.uint8)
        orange_numpy[:, :, 0] = 255
        orange_numpy[:, :, 1] = 128

        if line_style == 0:
            line_style = 'euclidean'
        elif line_style == 1:
            line_style = 'manhattan'

        rays_o = rays[:, None, 0:3]  # [H*W, 1, 3]
        rays_d = rays[:, None, 3:6]  # [H*W, 1, 3]

        step = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = max(torch.minimum(rate_a, rate_b).amax(-1).min(), near)
        steps = t_min + torch.arange(self.nSamples).to(rays.device) * step

        kernel = torch.tensor([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], dtype=torch.float32).to(rays.device).unsqueeze(0).unsqueeze(0)
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        with torch.no_grad():
            acc = torch.zeros(H, W).to(rays.device)
            edge_depth = torch.zeros(H, W).to(rays.device)  # the depth at which edges are detected
            layer = torch.full((H, W), False).to(rays.device)
            surface = torch.full((H, W), False).to(rays.device)
            immature = torch.full((H, W), False).to(rays.device)  # immature outlines could become mature later
            mature = torch.full((H, W), False).to(rays.device)  # finalised silhouette edges

            if track_lines:
                adj = torch.full((H, W), -1, dtype=torch.int8)\
                    .to(rays.device)  # number of neighbours for each point on a line, or -1 if not on a line
                endpoints = torch.full((H, W), False).to(rays.device)  # whether a point is an endpoint of a line
                row_coords, col_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
                lines = torch.stack((row_coords, col_coords), dim=-1).to(rays.device)  # [H, W, 2]

                def union(u, v):
                    u, v = tuple(u), tuple(v)
                    lines[find(u)] = torch.tensor(find(v))

                def find(a):
                    b = a
                    while tuple(lines[b]) != b:
                        b = tuple(lines[b])
                    while tuple(lines[a]) != a:
                        c = tuple(lines[a])
                        lines[a] = torch.tensor(b)
                        a = c
                    return b

            for i in tqdm(range(len(steps)),
                          desc=f'Scanning outlines (thresholds: {acc_threshold}, {inclusion_threshold}, '
                               f'weights: {near_weight} to {far_weight})',
                          leave=save_results):
                pts = rays_o + steps[i] * rays_d

                mask = ((self.aabb[0] <= pts) & (pts <= self.aabb[1])).all(dim=-1)
                if self.alphaMask is not None:
                    alphas = self.alphaMask.sample_alpha(pts[mask])
                    alpha_mask = alphas > 0
                    mask_inv = ~mask
                    mask_inv[mask] |= (~alpha_mask)
                    mask = ~mask_inv

                sigma = torch.zeros(H*W, 1).to(rays.device)
                if mask.any():
                    pts = self.normalize_coord(pts)
                    sigma_feature = self.compute_densityfeature(pts[mask])
                    valid_sigma = self.feature2density(sigma_feature)
                    sigma[mask] = valid_sigma

                sigma = sigma.reshape(H, W)
                alpha = 1. - torch.exp(-sigma * step * self.distance_scale)
                acc += alpha * (1.0 - acc)

                if acc_threshold >= 0.0:
                    # thresholding based on accumulated opacity
                    layer &= alpha >= inclusion_threshold * acc
                    layer |= torch.logical_and(acc >= acc_threshold, ~surface)
                else:
                    # thresholding based on volume density
                    layer = sigma > inclusion_threshold

                surface |= layer
                immature[layer] = False

                outer = F.conv2d(layer.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0) > 0
                outer[surface] = False

                # surviving immature outlines which are no longer on the edge of the current layer are finalised
                fresh = torch.logical_and(immature, ~outer)
                edge_depth[torch.logical_and(outer, ~torch.logical_or(mature, immature))] = i

                if track_lines:

                    fresh_padded = F.pad(fresh, (1, 1, 1, 1), mode='constant', value=False)
                    final_padded = F.pad(torch.logical_or(mature, surface), (1, 1, 1, 1), mode='constant', value=False)
                    tips = torch.logical_and((adj == 0) | (adj == 1), ~endpoints)
                    for coord in tips.nonzero():
                        x, y = coord[0].item(), coord[1].item()
                        for (dx, dy) in dirs:
                            if fresh_padded[x+dx+1][y+dy+1]:
                                union(lines[x + dx][y + dy], lines[x][y])
                                adj[x][y] += 1
                        if adj[x][y] <= 1 and final_padded[x:x+3, y:y+3].count_nonzero() == 9:
                            endpoints[x][y] = True

                    update = torch.logical_or(tips, fresh)
                    update_padded = F.pad(update, (1, 1, 1, 1), mode='constant', value=False)
                    adj[fresh] = 0
                    for coord in fresh.nonzero():
                        x, y = coord[0].item(), coord[1].item()
                        for (dx, dy) in dirs:
                            if update_padded[x+dx+1][y+dy+1]:
                                union(lines[x][y], lines[x+dx][y+dy])
                                adj[x][y] += 1

                if decay >= 0:
                    fresh |= torch.logical_and(immature, edge_depth <= i - decay)

                mature |= fresh
                immature = outer
                immature[mature] = False

                if save_results:
                    scan = torch.where(
                        layer, torch.ones_like(acc), torch.where(
                            surface, torch.full_like(acc, 0.5), torch.zeros_like(acc)))
                    scan = scan.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]

                    # immature edges are red
                    if visualize_depth is not None:
                        # mature edges are colored according to their depth
                        edge_depth_numpy = edge_depth.unsqueeze(-1).cpu().numpy()
                        colored = visualize_depth_numpy(edge_depth_numpy, (0, len(steps) - 1), visualize_depth)[0]
                        outline = np.where(
                            immature.unsqueeze(-1).cpu().numpy(), red_numpy, np.where(
                                mature.unsqueeze(-1).cpu().numpy(), colored, (scan * 255).cpu().numpy()))  # [H, W, 3]
                    else:
                        # mature edges are green
                        outline = torch.where(
                            immature.unsqueeze(-1), red_torch, torch.where(
                                mature.unsqueeze(-1), green_torch, scan))  # [H, W, 3]
                        outline = (outline.cpu().numpy() * 255)

                    if track_lines:
                        # temporary tips are yellow, finalised endpoints are orange
                        tips = ((adj == 0) | (adj == 1)).cpu().numpy()
                        endpoints_mask = endpoints.cpu().numpy()
                        outline[tips] = yellow_numpy[tips]
                        outline[endpoints_mask] = orange_numpy[endpoints_mask]

                    outline = outline.astype('uint8')
                    scan = (scan.cpu().numpy() * 255).astype('uint8')
                    if save_intermediates:
                        imageio.imwrite(os.path.join(scan_path, f'{i}.png'), scan)
                        imageio.imwrite(os.path.join(edge_path, f'{i}.png'), outline)
                    scans.append(scan)
                    outlines.append(outline)

            mature |= immature
            edges_raw = torch.where(mature, torch.ones_like(edge_depth), torch.zeros_like(edge_depth))

            # thicken edges according to their weight
            if near_weight != 1.0 or far_weight != 1.0:
                # weight(x) = a + (b - a) * (x - low) / (high - low) = a + m * (x - low)
                edge_depth = t_min + edge_depth * step  # [H, W]
                low, high = t_min.item(), (t_min + step * self.nSamples).item()
                mult = (far_weight - near_weight) / (high - low)
                weight_map = torch.where(
                    mature, near_weight + mult * (edge_depth - low), torch.zeros_like(edge_depth))  # [H, W]

                edges = torch.zeros_like(weight_map)  # [H, W]
                edge_points = weight_map.nonzero()  # [N, 2]
                y, x = torch.meshgrid(
                    torch.arange(H, device=rays.device), torch.arange(W, device=rays.device), indexing='ij')  # [H, W]
                for point in edge_points:
                    c_y, c_x = point[0].item(), point[1].item()
                    if line_style == 'euclidean':
                        dist = torch.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
                    elif line_style == 'manhattan':
                        dist = torch.abs(x - c_x) + torch.abs(y - c_y)
                    edges += torch.clamp(weight_map[c_y, c_x] - dist, min=0.0)  # [H, W]
                edges = torch.clamp(edges, min=0.0, max=1.0)  # [H, W]

            else:
                edges = edges_raw

        if save_results:

            os.makedirs(video_path, exist_ok=True)
            imageio.mimsave(os.path.join(
                video_path,
                f'{args.frame_index} ({acc_threshold}, {inclusion_threshold}) scan.mp4'),
                np.stack(scans),
                fps=24,
                macro_block_size=1)
            imageio.mimsave(os.path.join(
                video_path,
                f'{args.frame_index} ({acc_threshold}, {inclusion_threshold}) edge.mp4'),
                np.stack(outlines),
                fps=24,
                macro_block_size=1)

            scan = torch.where(surface, torch.full_like(acc, 0.5), torch.ones_like(acc))  # [H, W]
            edges_raw_img = torch.clamp(scan - edges_raw, min=0.0).unsqueeze(-1).expand(-1, -1, 3)
            edges_raw_img = (edges_raw_img.cpu().numpy() * 255).astype('uint8')
            imageio.imwrite(os.path.join(
                args.geo_buffer_path,
                f'{args.frame_index} ({acc_threshold}, {inclusion_threshold}) raw.png'),
                edges_raw_img)
            edges_img = torch.clamp(scan - edges, min=0.0).unsqueeze(-1).expand(-1, -1, 3)
            edges_img = (edges_img.cpu().numpy() * 255).astype('uint8')
            imageio.imwrite(os.path.join(
                args.geo_buffer_path,
                f'{args.frame_index} ({acc_threshold}, {inclusion_threshold}) '
                f'{near_weight} to {far_weight}, {line_style}.png'),
                edges_img)

        return edges_raw, edges


if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    dataset = dataset_dict[args.dataset_name]

    if args.pose == 'render' and args.dataset_name != 'tankstemple':
        args.pose = 'test'

    if args.dataset_name == 'tensoIR_simple' or args.dataset_name == 'tankstemple':
        dataset = dataset(
            args.datadir,
            split=args.pose,
            random_test=False,
            downsample=args.downsample_test,
            light_names=[],
            light_rotation=args.light_rotation
        )
    else:
        dataset = dataset(
            args.datadir,
            args.hdrdir,
            split='test',
            random_test=False,
            downsample=args.downsample_test,
            light_names=[],
            light_rotation=args.light_rotation
        )

    frame = dataset[args.frame_index]
    rays = frame['rays'].to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    model = ScanEdges(**kwargs)
    model.load(ckpt)

    if len(args.edge_detection_args) > 0:
        for edge_detection_args in args.edge_detection_args:
            edge_args = eval(edge_detection_args)
            model(
                rays, dataset.img_wh,
                decay=-1,
                acc_threshold=edge_args[0],
                inclusion_threshold=edge_args[1],
                track_lines=False,
                near_weight=edge_args[2],
                far_weight=edge_args[3],
                line_style=edge_args[4] if len(edge_args) > 4 else 'euclidean',
                image_dir=args.geo_buffer_path,
                save_intermediates=False,
                visualize_depth=None)#cv2.COLORMAP_SUMMER)
    else:
        model(
            rays, dataset.img_wh,
            decay=-1,
            track_lines=False,
            image_dir=args.geo_buffer_path,
            save_intermediates=False,
            visualize_depth=None)
