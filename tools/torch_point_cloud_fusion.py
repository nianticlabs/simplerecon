"""
    This is borrowed from https://github.com/alexrich021/3dvnet/blob/main/mv3d/eval/pointcloudfusion_custom.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

IMG_BATCH = 100  # modify to scale pc fusion implementation to your GPU

def process_depth(ref_depth, ref_image, src_depths, src_images, ref_P, src_Ps, ref_K, src_Ks, z_thresh=0.1,
                  n_consistent_thresh=3):
    n_src_imgs = src_depths.shape[0]
    h, w = int(ref_depth.shape[0]), int(ref_depth.shape[1])
    n_pts = h * w

    src_Ks = src_Ks.cuda()
    src_Ps = src_Ps.cuda()
    ref_K = ref_K.cuda()
    ref_P = ref_P.cuda()
    ref_depth = ref_depth.cuda()

    ref_K_inv = torch.inverse(ref_K)
    src_Ks_inv = torch.inverse(src_Ks)
    ref_P_inv = torch.inverse(ref_P)

    pts_x = np.linspace(0, w - 1, w)
    pts_y = np.linspace(0, h - 1, h)
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts = torch.from_numpy(np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)).float().cuda()
    pts = ref_P_inv[:3, :3] @ (ref_K_inv @ (pts * ref_depth.unsqueeze(0)).view(3, n_pts))\
          + ref_P_inv[:3, 3, None]

    n_batches = (n_src_imgs - 1) // IMG_BATCH + 1
    n_valid = 0.
    pts_sample_all = []
    valid_per_src_all = []
    for b in range(n_batches):
        idx_start = b * IMG_BATCH
        idx_end = min((b + 1) * IMG_BATCH, n_src_imgs)
        src_Ps_batch = src_Ps[idx_start: idx_end]
        src_Ks_batch = src_Ks[idx_start: idx_end]
        src_Ks_inv_batch = src_Ks_inv[idx_start: idx_end]
        src_depths_batch = src_depths[idx_start: idx_end].cuda()

        n_batch_imgs = idx_end - idx_start
        pts_reproj = torch.bmm(src_Ps_batch[:, :3, :3],
                               pts.unsqueeze(0).repeat(n_batch_imgs, 1, 1)) + src_Ps_batch[:, :3, 3, None]
        pts_reproj = torch.bmm(src_Ks_batch, pts_reproj)
        z_reproj = pts_reproj[:, 2]
        pts_reproj = pts_reproj / z_reproj.unsqueeze(1)

        valid_z = (z_reproj > 1e-4)
        valid_x = (pts_reproj[:, 0] >= 0.) & (pts_reproj[:, 0] <= float(w - 1))
        valid_y = (pts_reproj[:, 1] >= 0.) & (pts_reproj[:, 1] <= float(h - 1))

        grid = torch.clone(pts_reproj[:, :2]).transpose(2, 1).view(n_batch_imgs, n_pts, 1, 2)
        grid[..., 0] = (grid[..., 0] / float(w - 1)) * 2 - 1.0  # normalize to [-1, 1]
        grid[..., 1] = (grid[..., 1] / float(h - 1)) * 2 - 1.0  # normalize to [-1, 1]
        z_sample = F.grid_sample(src_depths_batch.unsqueeze(1), grid, mode='nearest', align_corners=True,
                                 padding_mode='zeros')
        z_sample = z_sample.squeeze(1).squeeze(-1)

        z_diff = torch.abs(z_reproj - z_sample)
        valid_disp = z_diff < z_thresh

        valid_per_src = (valid_disp & valid_x & valid_y & valid_z)
        n_valid += torch.sum(valid_per_src.int(), dim=0)

        # back project sampled pts for later averaging
        pts_sample = torch.bmm(src_Ks_inv_batch, pts_reproj * z_sample.unsqueeze(1))
        pts_sample = torch.bmm(src_Ps_batch[:, :3, :3].transpose(2, 1),
                               pts_sample - src_Ps_batch[:, :3, 3, None])
        pts_sample_all.append(pts_sample)
        valid_per_src_all.append(valid_per_src)
    pts_sample_all = torch.cat(pts_sample_all, dim=0)
    valid_per_src_all = torch.cat(valid_per_src_all, dim=0)

    valid = n_valid >= n_consistent_thresh

    # average sampled points amongst consistent views
    pts_avg = pts
    for i in range(n_src_imgs):
        pts_sample_i = pts_sample_all[i]
        invalid_idx = torch.isnan(pts_sample_i)  # filter out NaNs from div/0 due to grid sample zero padding
        pts_sample_i[invalid_idx] = 0.
        valid_i = valid_per_src_all[i] & ~torch.any(invalid_idx, dim=0)
        pts_avg += pts_sample_i * valid_i.float().unsqueeze(0)
    pts_avg = pts_avg / (n_valid + 1).float().unsqueeze(0).expand(3, n_pts)

    pts_filtered = pts_avg.transpose(1, 0)[valid].cpu().numpy()
    valid = valid.view(ref_depth.shape[-2:])
    rgb_filtered = ref_image[valid].view(-1, 3).cpu().numpy()

    return pts_filtered, rgb_filtered, valid.cpu().numpy()


def process_scene(depth_preds, images, poses, K, z_thresh, n_consistent_thresh):
    n_imgs = depth_preds.shape[0]
    fused_pts = []
    fused_rgb = []
    all_idx = torch.arange(n_imgs)
    all_valid = []
    for ref_idx in tqdm.tqdm(range(n_imgs)):
        src_idx = all_idx != ref_idx
        pts, rgb, valid = process_depth(depth_preds[ref_idx], images[ref_idx], depth_preds[src_idx], images[src_idx],
                                        poses[ref_idx], poses[src_idx], K[ref_idx], K[src_idx], z_thresh,
                                        n_consistent_thresh)
        fused_pts.append(pts)
        fused_rgb.append(rgb)
        all_valid.append(valid)
    fused_pts = np.concatenate(fused_pts, axis=0)
    fused_rgb = np.concatenate(fused_rgb, axis=0)
    all_valid = np.stack(all_valid, axis=0)

    return fused_pts, fused_rgb, all_valid
