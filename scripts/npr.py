
import os
from tqdm import tqdm
import imageio
import numpy as np

from models.scan_edges import ScanEdges
from opt import config_parser
import torch
import torch.nn as nn
import kornia.filters as K

from shaders import shader_dict
from utils import visualize_depth_numpy, bilateral_filter
# ----------------------------------------
# use this if loaded checkpoint is generate from single-light or rotated multi-light setting 
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask

# # use this if loaded checkpoint is generate from general multi-light setting 
# from models.tensoRF_general_multi_lights import TensorVMSplit, AlphaGridMask
# ----------------------------------------
from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = GGX_specular
from utils import rgb_ssim, rgb_lpips
from models.relight_utils import Environment_Light
from renderer import compute_rescale_ratio

@torch.no_grad()
def npr(dataset, args):

    if not os.path.exists(args.ckpt):
        print('the checkpoint path for tensoIR does not exists!!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)

    scan_edges = ScanEdges(**kwargs)
    scan_edges.load(ckpt)

    W, H = dataset.img_wh
    near_far = dataset.near_far
    
    rgb_frames_list = []
    optimized_normal_list = []

    aligned_albedo_list = []
    roughness_list = []

    xyz_list = []
    x_list, y_list, z_list = [], [], []

    edge_map_list, edge_mask_list = [], []

    shader_list = [shader_dict[shader](args, device) for shader in args.shaders]
    shader_lists, shader_debug_lists = [], []

    #### 
    light_rotation_idx = 0
    ####

    light_pos = torch.Tensor(args.light_pos).to(device)
    light_debug = args.light == 'point' and len(args.light_debug) == 3
    light_debug_color = torch.Tensor(args.light_debug).cpu() if light_debug else None

    for idx in tqdm(range(len(dataset))):

        cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(device)  # [H*W, 6]
        # gt_normal = item['normals'].squeeze(0).cpu() # [H*W, 3]s
        # gt_rgb = item['rgbs'].squeeze(0).reshape(len(light_name_list), H, W, 3).cpu()  # [N, H, W, 3]

        light_idx = torch.zeros((frame_rays.shape[0], 1), dtype=torch.int).to(device).fill_(light_rotation_idx)

        rgb_map, depth_map, normal_map, albedo_map, roughness_map, fresnel_map, normals_diff_map, acc_map, \
            normals_orientation_loss_map, xyz_map, view_map = [], [], [], [], [], [], [], [], [], [], []
        depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask = [], [], [], []
        shader_maps, shader_debug_maps = [], []
        light_debug_mask = []

        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]), args.batch_size)  # choose the first light idx
        for chunk_idx in chunk_idxs:
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, \
                    fresnel_chunk, acc_chunk, *temp \
                    = tensoIR(frame_rays[chunk_idx], light_idx[chunk_idx], is_train=False, white_bg=True, ndc_ray=False, N_samples=-1)

            # gt_albedo_chunk = gt_albedo[chunk_idx] # use GT to debug
            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(-1) * rays_d_chunk  # [bs, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask]  # [surface_point_num, 3]

            masked_normal_chunk = normal_chunk[acc_chunk_mask]  # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask]  # [surface_point_num, 3]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask]  # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask]  # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask]  # [surface_point_num, 1]
            masked_xyz_chunk = surface_xyz_chunk[acc_chunk_mask]

            normal_chunk = safe_l2_normalize(normal_chunk, dim=-1)
            view_chunk = -rays_d_chunk  # [bs, 3]
            view_chunk = safe_l2_normalize(view_chunk, dim=-1)  # [bs, 3]

            shader_chunks = torch.stack([shader(
                acc_chunk_mask, surface_xyz_chunk, depth_chunk, view_chunk, normal_chunk, albedo_chunk, roughness_chunk
            ) for shader in shader_list], dim=0).to(device)  # [shader_num, bs, 3]

            rgb_map.append(rgb_chunk.cpu().detach())
            depth_map.append(depth_chunk.cpu().detach())
            acc_map.append(acc_chunk.cpu().detach())
            normal_map.append(normal_chunk.cpu().detach())
            albedo_map.append(albedo_chunk.cpu().detach())
            roughness_map.append(roughness_chunk.cpu().detach())
            xyz_map.append(surface_xyz_chunk.cpu().detach())
            view_map.append(view_chunk.cpu().detach())
            shader_maps.append(shader_chunks.cpu().detach())

            if light_debug:
                o = rays_o_chunk - light_pos[None, :]  # [bs, 3]
                a = torch.sum(rays_d_chunk * rays_d_chunk, dim=-1, keepdim=True)  # [bs, 1]
                b = 2.0 * torch.sum(o * rays_d_chunk, dim=-1, keepdim=True)  # [bs, 1]
                c = torch.sum(o * o, dim=-1, keepdim=True) - args.debug_light_size  # [bs, 1]
                d = b * b - 4.0 * a * c  # [bs, 1]

                sphere_mask = (d >= 0.0)
                d[sphere_mask] = torch.sqrt(d[sphere_mask])
                t0 = (-b + d) / (2.0 * a)  # [bs, 1]
                t1 = (-b - d) / (2.0 * a)  # [bs, 1]
                t = -torch.ones_like(d)
                t[sphere_mask] = torch.minimum(t0, t1)[sphere_mask]  # [bs, 1]

                light_debug_chunk = torch.logical_and(
                    torch.ge(t, 0.0),
                    torch.logical_or(
                        torch.lt(t, depth_chunk.unsqueeze(-1)),
                        torch.logical_not(acc_chunk_mask.unsqueeze(-1))
                    )
                )
                light_debug_mask.append(light_debug_chunk.cpu().detach())

        rgb_map = torch.cat(rgb_map, dim=0)
        depth_map = torch.cat(depth_map, dim=0)
        acc_map = torch.cat(acc_map, dim=0)
        normal_map = torch.cat(normal_map, dim=0)
        acc_map_mask = (acc_map > args.acc_mask_threshold)
        albedo_map = torch.cat(albedo_map, dim=0)
        roughness_map = torch.cat(roughness_map, dim=0)
        xyz_map = torch.cat(xyz_map, dim=0)
        view_map = torch.cat(view_map, dim=0)
        shader_maps = torch.cat(shader_maps, dim=1)

        # Edge detection
        if args.edge_detection != 'none':

            depth_min = depth_map.min()
            depth_max = 2.0 * depth_map.max()
            normalized_depth_map = torch.full_like(depth_map, depth_max)
            normalized_depth_map[acc_map_mask] = depth_map[acc_map_mask]
            normalized_depth_map = (normalized_depth_map - depth_min) / (depth_max - depth_min)
            normalized_depth_map = normalized_depth_map.reshape(1, 1, H, W)

            modified_depth_map = 1.0 - normalized_depth_map
            modified_depth_map = modified_depth_map ** args.edge_detection_depth_modifier
            modified_normal_map = normal_map.permute(1, 0).reshape(1, 3, H, W)
            modified_normal_map = modified_normal_map * modified_depth_map.expand(-1, 3, -1, -1)

            if args.edge_detection == 'scan':
                depth_edge_mask = scan_edges(frame_rays, (W, H)).reshape(H*W).cpu()  # [H*W]
                depth_edge_map = torch.where(depth_edge_mask, torch.zeros((H*W,)), torch.ones((H*W,)))  # [H*W]
                normal_edge_mask = torch.full((H * W,), False)  # [H*W]
                normal_edge_map = torch.zeros((H * W,))  # [H*W]

            elif args.edge_detection == 'normals':
                depth_edge_map = torch.where(
                    acc_map_mask, torch.abs(torch.sum(view_map * normal_map, dim=-1)), torch.ones((H*W,)))  # [H*W]
                depth_edge_mask = (depth_edge_map < args.edge_detection_args[0])  # [H*W]
                normal_edge_map = torch.zeros((H*W,))  # [H*W]
                normal_edge_mask = torch.full((H*W,), False)  # [H*W]

            elif args.edge_detection == 'canny':
                depth_edge_map, depth_edge_mask = K.canny(
                    modified_depth_map,
                    low_threshold=args.edge_detection_args[0],
                    high_threshold=args.edge_detection_args[1]
                )  # [1, 1, H, W]
                depth_edge_map = depth_edge_map.view(-1)  # [H*W]
                depth_edge_mask = depth_edge_mask.bool().view(-1)  # [H*W]

                normal_edge_map, normal_edge_mask = K.canny(
                    modified_normal_map,
                    low_threshold=args.edge_detection_args[2],
                    high_threshold=args.edge_detection_args[3]
                )  # [1, 1, H, W]
                normal_edge_map = normal_edge_map.view(-1)  # [H*W]
                normal_edge_mask = normal_edge_mask.bool().view(-1)  # [H*W]

            elif args.edge_detection == 'sobel':
                depth_edge_map = K.sobel(modified_depth_map).view(-1)  # [H*W]
                depth_edge_mask = depth_edge_map > args.edge_detection_args[0]  # [H*W]

                normal_edge_map = K.sobel(modified_normal_map).view(3, -1)  # [3, H*W]
                normal_edge_map = torch.sum(normal_edge_map, dim=0)  # [H*W]
                normal_edge_mask = normal_edge_map > args.edge_detection_args[1]  # [H*W]

            # Draw edges
            for i in range(len(shader_list)):
                shader_maps[i] = shader_list[i].draw_edges(
                    shader_maps[i], depth_edge_map, depth_edge_mask, normal_edge_map, normal_edge_mask
                )

        # Draw renders
        if light_debug:
            shader_debug_maps = shader_maps.clone()
            light_debug_mask = torch.cat(light_debug_mask, dim=0).expand_as(shader_debug_maps)
            shader_debug_maps[light_debug_mask] = light_debug_color.expand_as(shader_debug_maps)[light_debug_mask]

            shaders_debug_list = []
            for i in range(shader_debug_maps.shape[0]):
                shader_debug_map = shader_debug_maps[i]
                shader_debug_map = (shader_debug_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
                shaders_debug_list.append(shader_debug_map)
                imageio.imwrite(os.path.join(cur_dir_path, f'{shader_list[i].name}_debug.png'), shader_debug_map)
            shader_debug_lists.append(shaders_debug_list)

        shaders_list = []
        for i in range(shader_maps.shape[0]):
            shader_map = shader_maps[i]  # [bs, 3]
            shader_map = (shader_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            shaders_list.append(shader_map)
            imageio.imwrite(os.path.join(cur_dir_path, f'{shader_list[i].name}.png'), shader_map)
        shader_lists.append(shaders_list)

        rgb_map = (rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
        rgb_frames_list.append(rgb_map)
        depth_map, _ = visualize_depth_numpy(depth_map.reshape(H, W, 1).numpy(), near_far)
        acc_map = (acc_map.reshape(H, W, 1).numpy() * 255).astype('uint8')

        if 'albedo' in item:
            gt_albedo = item['albedo'].squeeze(0).to(device)  # [H*W, 3]
        else:
            gt_albedo = albedo_map
        if 'rgbs_mask' in item:
            gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu()  # [H*W]
        else:
            gt_mask = acc_map_mask

        if args.if_save_rgb:
            imageio.imwrite(os.path.join(cur_dir_path, 'rgb.png'), rgb_map)
        if args.if_save_depth:
            depth_map = np.concatenate([depth_map, acc_map], axis=2)
            # depth_map, _ = visualize_depth_numpy(normalized_depth_map.reshape(H, W, 1).numpy(), near_far)
            imageio.imwrite(os.path.join(cur_dir_path, 'depth.png'), depth_map)
        if args.if_save_acc:
            imageio.imwrite(os.path.join(cur_dir_path, 'acc.png'), acc_map.repeat(3, axis=-1))
        if args.if_save_albedo:
            gt_albedo_reshaped = gt_albedo.reshape(H, W, 3).cpu()
            albedo_map = albedo_map.reshape(H, W, 3)
            # three channels rescale
            gt_albedo_mask = gt_mask.reshape(H, W)
            ratio_value, _ = (gt_albedo_reshaped[gt_albedo_mask] / albedo_map[gt_albedo_mask].clamp(min=1e-6)).median(dim=0)
            # ratio_value = gt_albedo_reshaped[gt_albedo_mask].median(dim=0)[0] / albedo_map[gt_albedo_mask].median(dim=0)[0]
            albedo_map[gt_albedo_mask] = (ratio_value * albedo_map[gt_albedo_mask]).clamp(min=0.0, max=1.0)

            albedo_map_to_save = (albedo_map * 255).numpy().astype('uint8')
            albedo_map_to_save = np.concatenate([albedo_map_to_save, acc_map], axis=2).astype('uint8')
            imageio.imwrite(os.path.join(cur_dir_path, 'albedo.png'), albedo_map_to_save)
            if args.if_save_albedo_gamma_corrected:
                to_save_albedo = (albedo_map ** (1/2.2) * 255).numpy().astype('uint8')
                to_save_albedo = np.concatenate([to_save_albedo, acc_map], axis=2)
                # gamma correction
                imageio.imwrite(os.path.join(cur_dir_path, 'albedo_gamma_corrected.png'), to_save_albedo)

            # save GT gamma corrected albedo
            gt_albedo_reshaped = (gt_albedo_reshaped ** (1/2.2) * 255).numpy().astype('uint8')
            gt_albedo_reshaped = np.concatenate([gt_albedo_reshaped, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'gt_albedo_gamma_corrected.png'), gt_albedo_reshaped)

            aligned_albedo_list.append(((albedo_map ** (1.0/2.2)) * 255).numpy().astype('uint8'))

            roughness_map = roughness_map.reshape(H, W, 1)
            # expand to three channels
            roughness_map = (roughness_map.expand(-1, -1, 3) * 255)
            roughness_map = np.concatenate([roughness_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'roughness.png'), (roughness_map).astype('uint8'))
            roughness_list.append(roughness_map.astype('uint8'))
        if args.if_render_normal:
            normal_rgb_map = normal_map * 0.5 + 0.5
            normal_rgb_map = (normal_rgb_map.reshape(H, W, 3).numpy() * 255).astype('uint8')
            normal_rgb_map = np.concatenate([normal_rgb_map, acc_map], axis=2)
            imageio.imwrite(os.path.join(cur_dir_path, 'normal.png'), normal_rgb_map)
        if args.if_save_xys:
            xyz_map = (torch.clamp(xyz_map, min=0.0, max=1.0).reshape(H, W, 3).numpy() * 255).astype('uint8')
            xyz_map = np.concatenate([xyz_map, acc_map], axis=2)
            x_map, y_map, z_map = xyz_map[:, :, 0:1], xyz_map[:, :, 1:2], xyz_map[:, :, 2:3]  # [H, W, 1]
            x_map, y_map, z_map = np.repeat(x_map, 3, axis=2), np.repeat(y_map, 3, axis=2), np.repeat(z_map, 3, axis=2)
            x_map, y_map, z_map = np.concatenate([x_map, acc_map], axis=2), np.concatenate([y_map, acc_map], axis=2), \
                np.concatenate([z_map, acc_map], axis=2)  # [H, W, 4]

            imageio.imwrite(os.path.join(cur_dir_path, 'xyz.png'), xyz_map)
            imageio.imwrite(os.path.join(cur_dir_path, 'x_pos.png'), x_map)
            imageio.imwrite(os.path.join(cur_dir_path, 'y_pos.png'), y_map)
            imageio.imwrite(os.path.join(cur_dir_path, 'z_pos.png'), z_map)

            xyz_list.append(xyz_map)
            x_list.append(x_map)
            y_list.append(y_map)
            z_list.append(z_map)
        if args.if_save_edges and args.edge_detection != 'none':
            edge_map = torch.full((H*W,), 0.5)
            edge_map += normal_edge_map / 2.0
            edge_map -= depth_edge_map / 2.0
            edge_map = (edge_map.reshape(H, W, 1).expand(-1, -1, 3).numpy() * 255).astype('uint8')
            imageio.imwrite(os.path.join(cur_dir_path, 'edge map.png'), edge_map)
            edge_map_list.append(edge_map)

            edge_mask = torch.full((H*W, 3), 0.5)
            edge_mask[normal_edge_mask] = torch.ones_like(edge_mask)[normal_edge_mask]
            edge_mask[depth_edge_mask] = torch.zeros_like(edge_mask)[depth_edge_mask]
            edge_mask = (edge_mask.reshape(H, W, 3).numpy() * 255).astype('uint8')
            imageio.imwrite(os.path.join(cur_dir_path, 'edge mask.png'), edge_mask)
            edge_mask_list.append(edge_mask)

    video_path = os.path.join(args.geo_buffer_path, 'video')
    os.makedirs(video_path, exist_ok=True)

    if args.if_save_rgb_video:
        imageio.mimsave(os.path.join(video_path, 'rgb_video.mp4'), np.stack(rgb_frames_list), fps=24, macro_block_size=1)

    if args.if_render_normal:
        for render_idx in range(len(dataset)):
            cur_dir_path = os.path.join(args.geo_buffer_path, f'{dataset.split}_{render_idx:0>3d}')
            normal_map = imageio.v2.imread(os.path.join(cur_dir_path, 'normal.png'))
            normal_mask = (normal_map[..., -1] / 255) > args.acc_mask_threshold 
            normal_map = normal_map[..., :3] * (normal_mask[..., None] / 255.0) + 255 * (1 - normal_mask[..., None] / 255.0)

            optimized_normal_list.append(normal_map.astype('uint8'))

        imageio.mimsave(os.path.join(video_path, 'render_normal_video.mp4'), np.stack(optimized_normal_list), fps=24, macro_block_size=1)

    if args.if_save_albedo:
        imageio.mimsave(os.path.join(video_path, 'aligned_albedo_video.mp4'), np.stack(aligned_albedo_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'roughness_video.mp4'), np.stack(roughness_list), fps=24, macro_block_size=1)

    if args.if_save_xys:
        imageio.mimsave(os.path.join(video_path, 'xyz.mp4'), np.stack(xyz_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'x_pos.mp4'), np.stack(x_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'y_pos.mp4'), np.stack(y_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'z_pos.mp4'), np.stack(z_list), fps=24, macro_block_size=1)

    if args.if_save_edges and args.edge_detection != 'none':
        imageio.mimsave(os.path.join(video_path, 'edge_map.mp4'), np.stack(edge_map_list), fps=24, macro_block_size=1)
        imageio.mimsave(os.path.join(video_path, 'edge_mask.mp4'), np.stack(edge_mask_list), fps=24, macro_block_size=1)

    if args.render_video:
        for i in range(len(shader_list)):
            imageio.mimsave(os.path.join(video_path, f'{shader_list[i].name}.mp4'),
                            np.stack([shader[i] for shader in shader_lists]), fps=24, macro_block_size=1)
            if light_debug:
                imageio.mimsave(os.path.join(video_path, f'{shader_list[i].name}_debug.mp4'),
                                np.stack(shader_debug[i] for shader_debug in shader_debug_lists), fps=24, macro_block_size=1)


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
    args.if_save_rgb = True
    args.if_save_depth = True
    args.if_save_acc = True
    args.if_save_rgb_video = False
    args.if_save_relight_rgb = True
    args.if_save_albedo = True
    args.if_save_albedo_gamma_corrected = True
    args.if_save_xys = True
    args.if_save_edges = True
    args.debug_light_size = 0.005
    args.acc_mask_threshold = 0.5
    args.if_render_normal = True
    args.vis_equation = 'nerv'
    args.render_video = True

    dataset = dataset_dict[args.dataset_name]

    if args.pose == 'render' and args.dataset_name != 'tankstemple':
        args.pose = 'test'

    if args.dataset_name == 'tensoIR_simple' or args.dataset_name == 'tankstemple':
        test_dataset = dataset(
            args.datadir,
            split=args.pose,
            random_test=False,
            downsample=args.downsample_test,
            light_names=[],
            light_rotation=args.light_rotation
        )
    else:
        test_dataset = dataset(
            args.datadir,
            args.hdrdir,
            split='test',
            random_test=False,
            downsample=args.downsample_test,
            light_names=[],
            light_rotation=args.light_rotation
        )

    npr(test_dataset, args)

    