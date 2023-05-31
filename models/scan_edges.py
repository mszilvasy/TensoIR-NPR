import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataLoader import dataset_dict
from models.tensoRF_rotated_lights import TensorVMSplit, raw2alpha
from opt import config_parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScanEdges(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kwargs):
        super().__init__(aabb, gridSize, device, **kwargs)

    def __call__(self, rays, hw, image_dir=None):
        """
        :param rays: [H*W, 6]
        :return: [H, W, 1]
        """
        save_images = image_dir is not None
        if save_images:
            scan_path = os.path.join(image_dir, 'scan')
            edge_path = os.path.join(image_dir, 'edge')
            video_path = os.path.join(image_dir, 'video')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(scan_path, exist_ok=True)
            os.makedirs(edge_path, exist_ok=True)
        else:
            scan_path, edge_path, video_path = None, None, None
        scans, outlines = [], []

        H, W = hw
        red = torch.zeros(H, W, 3).to(rays.device)
        red[:, :, 0] = 1.0
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

        with torch.no_grad():
            acc = torch.zeros(H, W).to(rays.device)
            layer = torch.full((H, W), False).to(rays.device)
            surface = torch.full((H, W), False).to(rays.device)
            immature = torch.full((H, W), False).to(rays.device)  # immature outlines could become mature later
            mature = torch.full((H, W), False).to(rays.device)  # finalised silhouette edges
            for i in tqdm(range(len(steps))):
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

                _, weight, _ = raw2alpha(sigma, step * self.distance_scale)
                weight = weight.reshape(H, W)
                weight_mask = weight > self.rayMarch_weight_thres  # [H, W]
                acc[weight_mask] += weight[weight_mask]

                layer_mask = torch.logical_and(weight_mask, acc >= 1.0)  # [H, W]
                layer &= layer_mask
                layer |= torch.logical_and(layer_mask, ~surface)
                surface |= layer

                immature[layer] = False
                outer = torch.logical_or(
                    torch.logical_or(F.pad(layer, (2, 0, 1, 1), value=False), F.pad(layer, (0, 2, 1, 1), value=False)),
                    torch.logical_or(F.pad(layer, (1, 1, 2, 0), value=False), F.pad(layer, (1, 1, 0, 2), value=False))
                )  # [H+2, W+2]
                outer = outer[1:-1, 1:-1]  # [H, W]

                # surviving immature outlines which are no longer on the edge of the current layer are finalised
                mature |= torch.logical_and(immature, ~outer)
                immature = outer
                immature[torch.logical_or(surface, mature)] = False

                if save_images:
                    scan = torch.where(
                        layer, torch.ones_like(acc), torch.where(
                            surface, torch.full_like(acc, 0.5), torch.zeros_like(acc)))
                    scan = scan.unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
                    outline = torch.where(
                        immature.unsqueeze(-1), red, torch.where(
                            mature.unsqueeze(-1), green, scan))  # [H, W, 3]

                    scan = (scan.cpu().numpy() * 255).astype('uint8')
                    outline = (outline.cpu().numpy() * 255).astype('uint8')
                    imageio.imwrite(os.path.join(scan_path, f'{i}.png'), scan)
                    imageio.imwrite(os.path.join(edge_path, f'{i}.png'), outline)
                    scans.append(scan)
                    outlines.append(outline)

            mature |= immature

        if save_images:
            os.makedirs(video_path, exist_ok=True)
            imageio.mimsave(os.path.join(video_path, 'scan.mp4'), np.stack(scans), fps=24, macro_block_size=1)
            imageio.mimsave(os.path.join(video_path, 'edge.mp4'), np.stack(outlines), fps=24, macro_block_size=1)

        return mature


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

    res = model(rays, dataset.img_wh, image_dir=args.geo_buffer_path)
    img = torch.where(res, torch.zeros_like(res), torch.ones_like(res))
    img = img.unsqueeze(-1).expand(-1, -1, 3)
    img = (img.cpu().numpy() * 255).astype('uint8')
    imageio.imwrite(os.path.join(args.geo_buffer_path, f'{args.frame_index}_final.png'), img)
