""" 
    Loops through precomputed saved depth maps and associated RGB and groundtruth images, 
    and generates side by side viusalization comparisons. Saves the video under depth_videos for the specific
    test folder.

    This script will optionally use smoothed min/max limits from the gt to colormap predictions. These can 
    be computed by running visualization_scripts/generate_gt_min_max_cache.py 
    
    Example command:
        python ./visualization_scripts/visualize_scene_depth_output.py --name HERO_MODEL \
            --output_base_path OUTPUT_PATH \
            --data_config configs/data/scannet_dense.yaml;
"""

import sys

sys.path.append("/".join(sys.path[0].split("/")[:-1])) 

import os
import pickle
from pathlib import Path

import numpy as np
import options
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataset_utils import get_dataset
from utils.generic_utils import reverse_imagenet_normalize
from utils.geometry_utils import NormalGenerator
from utils.visualization_utils import colormap_image, save_viz_video_frames


def main(opts):

    print("Setting batch size to 1.")
    opts.batch_size = 1

    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    # path for min and max values for gt depth maps.
    gt_viz_path = os.path.join(opts.output_base_path, "gt_min_max", opts.dataset)

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, opts.name, 
                                        opts.dataset, opts.frame_tuple_type)

    # path where cached depth maps are
    depth_output_dir = os.path.join(results_path, "depths")

    # path where we store output videos.
    viz_output_folder_name = "depth_videos"
    viz_output_dir = os.path.join(results_path, "viz", viz_output_folder_name)
    print(viz_output_dir)
    Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"".center(80, "#"))
    print(f" Computing Visualizations.")
    print(f"Output directory: {viz_output_dir} ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")

    # gap between image panels
    buffer_gap = 4
    # otuput resolution for each panel
    large_format_size = [480, 640]

    # normals computer
    compute_normals_full = NormalGenerator(
                            large_format_size[0], large_format_size[1]).cuda()

    src_format_size = [69, 91]

    # depth prediction normals computer
    pred_format_size = [192, 256]
    compute_normals = NormalGenerator(
                            pred_format_size[0], pred_format_size[1]).cuda()

    with torch.inference_mode():
        for scan in tqdm(scans):
        
            # load gt_min_max if it exists, else just use the deault at 0m and 5m.
            gt_min_max_path = os.path.join(gt_viz_path, f"{scan}.txt")
            if os.path.exists(gt_min_max_path):
                with open(gt_min_max_path, 'r') as handle:
                    limits = handle.readline()
                    limits = [float(limit) for limit in limits.strip().split(" ")]
            else:
                print(f"".center(80, "#"))
                print(f" No GT min/max found. Using min=0m and max=5m ".center(80, "#"))
                print(f"".center(80, "#"))
                print("")
                limits = [0,5]

            vmin, vmax = limits

            # load dataset with specific scan
            dataset = dataset_class(
                                opts.dataset_path,
                                split=opts.split,
                                mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                                limit_to_scan_id=scan,
                                include_full_res_depth=True,
                                tuple_info_file_location=opts.tuple_info_file_location,
                                num_images_in_tuple=None,
                                shuffle_tuple=opts.shuffle_tuple,
                                include_high_res_color=True,
                                pass_frame_id=True,
                                include_full_depth_K=True,
                                skip_to_frame=opts.skip_to_frame,
                            )

            dataloader = torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=opts.batch_size, 
                                                shuffle=False, 
                                                num_workers=opts.num_workers, 
                                                drop_last=False,
                                            )
        
            # loop over every element in dataset
            output_image_list=[]
            for _, batch in enumerate(tqdm(dataloader)):
                cur_data, src_data = batch

                invK_full_depth_b44 = cur_data["invK_full_depth_b44"].cuda()
                invK_s0_b44 = cur_data["invK_s0_b44"].cuda()

                frame_id = cur_data["frame_id_string"][0]
                pickled_path = os.path.join(depth_output_dir, scan, f"{frame_id}.pickle")

                # try to load result
                try:
                    with open(pickled_path, 'rb') as handle:
                        outputs = pickle.load(handle)
                except:
                    print(f"Error! Couldn't find {pickled_path}!")
                    continue

                ############################## RGB ##############################
                # reference image
                main_color_3hw = F.interpolate(
                                    cur_data["high_res_color_b3hw"].cuda(), 
                                    size=(large_format_size[0],
                                                        large_format_size[1]), 
                                    mode="bilinear", 
                                    align_corners=False,
                                )[0]
                main_color_3hw = reverse_imagenet_normalize(main_color_3hw)

                # source images
                src_images_k3hw = reverse_imagenet_normalize(torch.tensor(src_data['image_b3hw'][0].cuda()))
                lower_image_panel_k3hw = F.interpolate(
                                    src_images_k3hw, 
                                    size=(src_format_size[0], 
                                                        src_format_size[1]), 
                                    mode="bilinear", 
                                    align_corners=False,
                                )

                lower_image_panel_13hw_list = lower_image_panel_k3hw.split(split_size=1, dim=0)
                lower_image_panel_3hw = torch.cat(
                                        lower_image_panel_13hw_list, 
                                        dim=3,
                                    ).squeeze()

                ############################## Our depth and normals ##############################
                depth_pred_s0_b1hw = outputs["depth_pred_s0_b1hw"].cuda()
                our_depth_3hw = colormap_image(
                                    depth_pred_s0_b1hw.squeeze(0), 
                                    vmin=vmin,
                                    vmax=vmax,
                                )

                normals_b3hw = compute_normals(depth_pred_s0_b1hw, invK_s0_b44)
                our_normals_3hw = 0.5 * (1 + normals_b3hw).squeeze(0)

                # lowest cost guest from the cost volume
                lowest_cost_bhw = outputs["lowest_cost_bhw"].cuda()
                if opts.mask_pred_depth:
                    lowest_cost_bhw[outputs["overall_mask_bhw"].cuda()] = 0
                lowest_cost_3hw = colormap_image(
                                    lowest_cost_bhw,
                                    vmin=vmin,
                                    vmax=vmax,
                                )

                ############################## gt depth ##############################
                gt_depth_b1hw = cur_data["full_res_depth_b1hw"].cuda()[0][None]

                if gt_depth_b1hw.shape[-1] != large_format_size[1]:
                    gt_depth_b1hw = F.interpolate(
                            gt_depth_b1hw,
                            size=(large_format_size[0], large_format_size[1]), 
                            mode="nearest",
                        )

                gt_mask_b1hw = (
                                    (gt_depth_b1hw > 1e-3) & 
                                    (gt_depth_b1hw < 10)
                                ).float() 

                gt_depth_3hw = colormap_image(
                                    gt_depth_b1hw.squeeze(0), 
                                    gt_mask_b1hw.squeeze(0), 
                                    vmin=vmin, 
                                    vmax=vmax,
                                )

                normals_b3hw = compute_normals_full(
                                            gt_depth_b1hw, invK_full_depth_b44)
                gt_normals_3hw = 0.5 * (1 + normals_b3hw).squeeze(0)
                gt_normals_3hw = torch.nan_to_num(gt_normals_3hw)

                ############################## assemble ##############################

                # gt column
                target_height = large_format_size[0]
                horizontal_buffer = torch.ones(
                                            gt_depth_3hw.shape[0], 
                                            buffer_gap, 
                                            gt_depth_3hw.shape[2]
                                        ).cuda()
                gt_column = [gt_depth_3hw, horizontal_buffer, gt_normals_3hw]
                gt_column_np = [element.permute(1,2,0).cpu().detach().numpy() 
                                                * 255 for element in gt_column]
                gt_column_np = np.uint8(np.concatenate(gt_column_np, axis=0))

                # our column
                target_width = large_format_size[1]
                our_depth_3hw = F.interpolate(
                                    our_depth_3hw.unsqueeze(0), 
                                    size=(target_height, target_width), 
                                    mode="nearest",
                                ).squeeze()
                our_normals_3hw = F.interpolate(
                                    our_normals_3hw.unsqueeze(0), 
                                    size=(target_height, target_width), 
                                    mode="nearest",
                                ).squeeze()
                our_lowest_cost_3hw = F.interpolate(
                                    lowest_cost_3hw.unsqueeze(0), 
                                    size=(target_height, target_width),
                                    mode="nearest",
                                ).squeeze()

                horizontal_buffer = torch.ones(
                                        our_depth_3hw.shape[0],
                                        buffer_gap,
                                        our_depth_3hw.shape[2],
                                    ).cuda()
                our_column = [our_depth_3hw, horizontal_buffer, our_normals_3hw]
                our_column_np = [element.permute(1,2,0).cpu().detach().numpy() 
                                                * 255 for element in our_column]
                our_column_np = np.uint8(np.concatenate(our_column_np, axis=0))

                # assemble results
                horizontal_buffer = np.ones((target_height*2 + buffer_gap, buffer_gap, 3)) * 255
                
                results_list = [our_column_np, horizontal_buffer, gt_column_np]  
                results_panel_np = np.uint8(np.concatenate(results_list, axis=1))

                # shape image and src images
                pad_val = main_color_3hw.shape[2] - lower_image_panel_3hw.shape[2]
                lower_image_panel_3hw = F.pad(
                                            lower_image_panel_3hw, 
                                            (0,pad_val,0,0,0,0),
                                            value=0.0,
                                        )
                main_color_3hw[:, 
                                main_color_3hw.shape[1] 
                                    - lower_image_panel_3hw.shape[1]:,
                                :] = lower_image_panel_3hw


                # append image to lowest cost
                horizontal_buffer = torch.ones(
                                        main_color_3hw.shape[0], 
                                        buffer_gap, 
                                        main_color_3hw.shape[2]
                                    ).cuda()
                color_panel = torch.cat(
                                [
                                    our_lowest_cost_3hw, 
                                    horizontal_buffer,
                                    main_color_3hw
                                ], dim = 1)      
                color_panel_np = np.uint8(
                                        color_panel.permute(1,2,0
                                                ).cpu().detach().numpy() * 255)

                assert color_panel_np.shape[0] == results_panel_np.shape[0]

                # final assemble
                vertical_buffer = np.ones((color_panel_np.shape[0], buffer_gap, 3)) * 255
                final_image_list = [color_panel_np, vertical_buffer, results_panel_np]
                final_image_np = np.concatenate(final_image_list, axis=1)

                # pad for optimal compression
                pad_height = (16 - (final_image_np.shape[0] % 16) 
                                    if final_image_np.shape[0] % 16 !=0 else 0)
                pad_width = (16 - (final_image_np.shape[1] % 16) 
                                    if final_image_np.shape[1] % 16 !=0 else 0)
                final_image_np = np.uint8(
                                    np.pad(
                                        final_image_np, 
                                        pad_width=(
                                            (0, pad_height), 
                                            (0, pad_width), 
                                            (0, 0),
                                        ), 
                                        constant_values=255,
                                    )
                                )

                # export image
                output_image_list.append(final_image_np)

            fps = (opts.standard_fps if opts.skip_frames is None 
                                else round(opts.standard_fps/opts.skip_frames))
            save_viz_video_frames(
                                output_image_list, 
                                os.path.join(viz_output_dir, f"{scan}.mp4"),
                                fps=fps,
                            )

if __name__ == '__main__':
    # don't need grad for test.
    torch.set_grad_enabled(False)

    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    main(opts)
