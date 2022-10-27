import os

import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
from PIL import Image

from utils.generic_utils import reverse_imagenet_normalize


def colormap_image(
                image_1hw,
                mask_1hw=None, 
                invalid_color=(0.0, 0, 0.0), 
                flip=True,
                vmin=None,
                vmax=None, 
                return_vminvmax=False,
                colormap="turbo",
            ):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args: 
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels. 
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the 
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
                            plt.cm.get_cmap(colormap)(
                                                torch.linspace(0, 1, 256)
                                            )[:, :3]
                        ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
                                        ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw

def save_viz_video_frames(frame_list, path, fps=30):
    """
    Saves a video file of numpy RGB frames in frame_list.
    """
    clip = mpy.ImageSequenceClip(frame_list, fps=fps)
    clip.write_videofile(path, verbose=False, logger=None)

    return


def quick_viz_export(
            output_path,
            outputs, 
            cur_data,
            batch_ind, 
            valid_mask_b,
            batch_size):
    """ Helper function for quickly exporting depth maps during inference. """

    if valid_mask_b.sum() == 0:
        batch_vmin = 0.0
        batch_vmax = 5.0
    else:
        batch_vmin = cur_data["full_res_depth_b1hw"][valid_mask_b].min()
        batch_vmax = cur_data["full_res_depth_b1hw"][valid_mask_b].max()

    if batch_vmax == batch_vmin:
        batch_vmin = 0.0
        batch_vmax = 5.0

    for elem_ind in range(outputs["depth_pred_s0_b1hw"].shape[0]):
        if "frame_id_string" in cur_data:
            frame_id = cur_data["frame_id_string"][elem_ind]
        else:
            frame_id = (batch_ind * batch_size) + elem_ind
            frame_id = f"{str(frame_id):6d}"
        
        # check for valid depths from dataloader
        if valid_mask_b[elem_ind].sum() == 0:
            sample_vmin = 0.0
            sample_vmax = 0.0
        else:
            # these will be the same when the depth map is all ones.
            sample_vmin = cur_data["full_res_depth_b1hw"][elem_ind][valid_mask_b[elem_ind]].min()
            sample_vmax = cur_data["full_res_depth_b1hw"][elem_ind][valid_mask_b[elem_ind]].max()

        # if no meaningful gt depth in dataloader, don't viz gt and 
        # set vmin/max to default
        if sample_vmax != sample_vmin:
            full_res_depth_1hw = cur_data["full_res_depth_b1hw"][elem_ind]

            full_res_depth_3hw = colormap_image(
                                        full_res_depth_1hw, 
                                        vmin=batch_vmin, vmax=batch_vmax
                                    )

            full_res_depth_hw3 = np.uint8(
                                full_res_depth_3hw.permute(1,2,0
                            ).cpu().detach().numpy() * 255
                        )
            Image.fromarray(full_res_depth_hw3).save(
                                        os.path.join(output_path, 
                                        f"{frame_id}_gt_depth.png")
                                    )

        lowest_cost_3hw = colormap_image(
                                    outputs["lowest_cost_bhw"][elem_ind].unsqueeze(0), 
                                    vmin=batch_vmin, vmax=batch_vmax
                                )
        pil_image = Image.fromarray(
                            np.uint8(
                                lowest_cost_3hw.permute(1,2,0
                                    ).cpu().detach().numpy() * 255)
                        )
        pil_image.save(os.path.join(output_path, 
                                f"{frame_id}_lowest_cost_pred.png"))

        depth_3hw = colormap_image(
                            outputs["depth_pred_s0_b1hw"][elem_ind], 
                            vmin=batch_vmin, vmax=batch_vmax)
        pil_image = Image.fromarray(
                            np.uint8(depth_3hw.permute(1,2,0
                                    ).cpu().detach().numpy() * 255)
                        )

        pil_image.save(os.path.join(output_path, f"{frame_id}_pred_depth.png"))

        main_color_3hw = cur_data["high_res_color_b3hw"][elem_ind]
        main_color_3hw = reverse_imagenet_normalize(main_color_3hw)
        pil_image = Image.fromarray(
                            np.uint8(main_color_3hw.permute(1,2,0
                                    ).cpu().detach().numpy() * 255)
                        )
        pil_image.save(os.path.join(output_path, f"{frame_id}_color.png"))