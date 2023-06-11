import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataLoader import dataset_dict
from models.tensoRF_rotated_lights import TensorVMSplit, raw2alpha
from opt import config_parser
from utils import visualize_depth_numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScanEdges(TensorVMSplit):
    def __init__(self, aabb, gridSize, device,
                 acc_threshold=0.5, alpha_threshold=0.35, strength_threshold=0.1,
                 **kwargs):
        super().__init__(aabb, gridSize, device, **kwargs)
        self.acc_threshold = acc_threshold
        self.alpha_threshold = alpha_threshold
        self.strength_threshold = strength_threshold

    def __call__(self, rays, hw,
                 track_lines=-1, depth_weight=1.0, image_dir=None, save_intermediates=False, visualize_depth=None):
        """
        :param rays: [H*W, 6]
        :param hw: (H, W)
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

        H, W = hw
        red_torch = torch.zeros(H, W, 3).to(rays.device)
        red_torch[:, :, 0] = 1.0
        red_numpy = np.zeros((H, W, 3), dtype=np.uint8)
        red_numpy[:, :, 0] = 255
        green = torch.zeros(H, W, 3).to(rays.device)
        green[:, :, 1] = 1.0

        rays_o = rays[:, None, 0:3]  # [H*W, 1, 3]
        rays_d = rays[:, None, 3:6]  # [H*W, 1, 3]

        step = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = max(torch.minimum(rate_a, rate_b).amax(-1).min(), near)
        steps = t_min + torch.arange(self.nSamples).to(rays.device) * step
        minmax = (t_min.item(), (t_min + step * self.nSamples).item())

        with torch.no_grad():
            acc = torch.zeros(H, W).to(rays.device)
            edge_depth = torch.zeros(H, W).to(rays.device)  # the depth at which edges are detected
            layer = torch.full((H, W), False).to(rays.device)
            surface = torch.full((H, W), False).to(rays.device)
            immature = torch.full((H, W), False).to(rays.device)  # immature outlines could become mature later
            mature = torch.full((H, W), False).to(rays.device)  # finalised silhouette edges
            for i in tqdm(range(len(steps)), desc='Scanning outlines', leave=save_results):
                depth = steps[i]
                pts = rays_o + depth * rays_d

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

                alpha, _, _ = raw2alpha(sigma, step * self.distance_scale)
                alpha = alpha.reshape(H, W)
                acc += alpha * (1.0 - acc)
                acc_mask = acc >= self.acc_threshold
                layer_mask = torch.logical_and(acc_mask, alpha > self.alpha_threshold)
                layer &= layer_mask
                layer |= torch.logical_and(acc_mask, ~surface)
                surface |= layer

                immature[layer] = False
                outer = torch.logical_or(
                    torch.logical_or(F.pad(layer, (2, 0, 1, 1), value=False), F.pad(layer, (0, 2, 1, 1), value=False)),
                    torch.logical_or(F.pad(layer, (1, 1, 2, 0), value=False), F.pad(layer, (1, 1, 0, 2), value=False))
                )  # [H+2, W+2]
                outer = outer[1:-1, 1:-1]  # [H, W]

                # surviving immature outlines which are no longer on the edge of the current layer are finalised
                mature |= torch.logical_and(immature, ~outer)
                edge_depth[torch.logical_and(outer, ~torch.logical_or(mature, immature))] = depth
                immature = outer
                immature[torch.logical_or(surface, mature)] = False

                if save_results:
                    scan = torch.where(
                        layer, torch.ones_like(acc), torch.where(
                            surface, torch.full_like(acc, 0.5), torch.zeros_like(acc)))
                    scan = scan.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
                    if visualize_depth is not None:
                        edge_depth_numpy = edge_depth.unsqueeze(-1).cpu().numpy()
                        colored = visualize_depth_numpy(edge_depth_numpy, minmax, visualize_depth)[0]
                        outline = np.where(
                            immature.unsqueeze(-1).cpu().numpy(), red_numpy, np.where(
                                mature.unsqueeze(-1).cpu().numpy(), colored, (scan * 255).cpu().numpy()))  # [H, W, 3]
                        outline = outline.astype('uint8')
                    else:
                        outline = torch.where(
                            immature.unsqueeze(-1), red_torch, torch.where(
                                mature.unsqueeze(-1), green, scan))  # [H, W, 3]
                        outline = (outline.cpu().numpy() * 255).astype('uint8')

                    scan = (scan.cpu().numpy() * 255).astype('uint8')
                    if save_intermediates:
                        imageio.imwrite(os.path.join(scan_path, f'{i}.png'), scan)
                        imageio.imwrite(os.path.join(edge_path, f'{i}.png'), outline)
                    scans.append(scan)
                    outlines.append(outline)

            mature |= immature

            # weight(n) = (k - 1)(n - y)/(x - y) + 1 = m(n - y) + 1
            if depth_weight != 1.0:
                mult = (depth_weight - 1.0) / (minmax[0] - minmax[1])
                edge_map = torch.where(
                    mature, mult * (edge_depth - minmax[1]) + 1.0, torch.zeros_like(edge_depth))  # [H, W]
                max_weight = edge_map.max().item()
                pad = int(np.ceil(max_weight) - 1)
                edge_map = F.pad(edge_map, (pad, pad, pad, pad), value=0.0)  # [H+2*pad, W+2*pad]
                edge_mask = edge_map > 0.0  # [H+2*pad, W+2*pad]

                for i in range(0, pad):
                    excess = torch.clamp(edge_map - 1.0, min=0.0)
                    edge_map -= excess
                    excess /= 4.0
                    for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        edge_map[~edge_mask] += torch.roll(excess, shift, dims=(0, 1))[~edge_mask]
                    edge_mask = edge_map > 0.0

                    if save_results and visualize_depth is not None:
                        scan = torch.where(surface, torch.full_like(acc, 0.5), torch.zeros_like(acc))  # [H, W]
                        scan = (scan.unsqueeze(-1).expand(-1, -1, 3) * 255).cpu().numpy()  # [H, W, 3]
                        edge_map_numpy = edge_map[pad:-pad, pad:-pad].unsqueeze(-1).cpu().numpy()
                        colored = visualize_depth_numpy(edge_map_numpy, (1.0, depth_weight), visualize_depth)[0]
                        outline = np.where(
                            edge_mask[pad:-pad, pad:-pad].unsqueeze(-1).cpu().numpy(), colored, scan)  # [H, W, 3]
                        outline = outline.astype('uint8')
                        if save_intermediates:
                            imageio.imwrite(os.path.join(edge_path, f'{len(outlines)}_expand.png'), outline)
                        outlines.append(outline)

                edge_map = edge_map[pad:-pad, pad:-pad]  # [H, W]
                mature = edge_map >= self.strength_threshold  # [H, W]
            else:
                edge_map = torch.where(mature, torch.ones_like(edge_depth), torch.zeros_like(edge_depth))  # [H, W]

        if save_results:
            os.makedirs(video_path, exist_ok=True)
            imageio.mimsave(os.path.join(video_path, 'scan.mp4'), np.stack(scans), fps=24, macro_block_size=1)
            imageio.mimsave(os.path.join(video_path, 'edge.mp4'), np.stack(outlines), fps=24, macro_block_size=1)

        return mature, edge_map


if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(os.path.abspath(args.geo_buffer_path)))

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    # The following args are not defined in opt.py
    args.frame_index = 10

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

    res_mask, res_map = model(
        rays, dataset.img_wh,
        track_lines=2,
        depth_weight=1.0,
        image_dir=args.geo_buffer_path,
        save_intermediates=False,
        visualize_depth=None)#cv2.COLORMAP_SUMMER)
    mask_img = torch.where(res_mask, torch.zeros_like(res_mask), torch.ones_like(res_mask))
    mask_img = mask_img.unsqueeze(-1).expand(-1, -1, 3)
    mask_img = (mask_img.cpu().numpy() * 255).astype('uint8')
    imageio.imwrite(os.path.join(args.geo_buffer_path, f'{args.frame_index}_mask.png'), mask_img)
    map_img = torch.where(res_mask, 1.0 - res_map, torch.ones_like(res_map))
    map_img = (map_img.cpu().numpy() * 255).astype('uint8')
    imageio.imwrite(os.path.join(args.geo_buffer_path, f'{args.frame_index}_map.png'), map_img)
