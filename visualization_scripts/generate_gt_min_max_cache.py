""" 
    Loops through an MVS dataset, and extracts smoothed max and min values 
    across a scene. Saves those values to disk. These limits can then be used 
    for visualization.

    Example command:
        python visualization_scripts/generate_gt_min_max_cache.py \
            --output_base_path OUTPUT_PATH \
            --data_config configs/data/scannet_dense.yaml;

"""

import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1])) 
import os
from pathlib import Path

import numpy as np
import options
import scipy
import torch
from tqdm import tqdm
from utils.dataset_utils import get_dataset


def main(opts):

    print("Setting batch size to 1.")
    opts.batch_size = 1

    # by default, we skip every 12 frames. We're looking for a rough average, 
    # so this saves a lot of time.
    if opts.skip_frames is None:
        print("Setting skip_frames size to 12.")
        opts.skip_frames = 12

    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    # will save limits per scene here
    gt_viz_path = os.path.join(opts.output_base_path, "gt_min_max", opts.dataset)
    Path(gt_viz_path).mkdir(parents=True, exist_ok=True)

    print(f"".center(80, "#"))
    print(f" Computing GT min/max.")
    print(f" Output directory: {gt_viz_path} ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")

    with torch.inference_mode():
        for scan in tqdm(scans):

            # set up dataset with current scan
            dataset = dataset_class(
                        opts.dataset_path,
                        split=opts.split,
                        mv_tuple_file_suffix=opts.mv_tuple_file_suffix,
                        limit_to_scan_id=scan,
                        include_full_res_depth=False,
                        tuple_info_file_location=opts.tuple_info_file_location,
                        num_images_in_tuple=None,
                        shuffle_tuple=opts.shuffle_tuple,
                        include_high_res_color=False,
                        pass_frame_id=True,
                        include_full_depth_K=False,
                        skip_frames=opts.skip_frames,
                        skip_to_frame=opts.skip_to_frame,
                    )

            dataloader = torch.utils.data.DataLoader(
                                            dataset, 
                                            batch_size=opts.batch_size, 
                                            shuffle=False, 
                                            num_workers=opts.num_workers, 
                                            drop_last=False,
                                        )

            # set inits
            vmin = torch.inf
            vmax = 0
            mins = []
            maxs = []
            for _, batch in enumerate(tqdm(dataloader)):
                cur_data, _ = batch
                
                depth = cur_data["depth_b1hw"].cuda()[
                                                cur_data["mask_b_b1hw"].cuda()]
                # get values at 98% and 2%
                maxs.append(torch.quantile(depth, 0.98).squeeze().cpu())
                mins.append(torch.quantile(depth, 0.02).squeeze().cpu())
            
            # gaussian filter all values to remove any outliers, then take 
            # min/max
            maxs = scipy.ndimage.gaussian_filter1d(np.array(maxs), sigma=1)
            vmax = np.max(maxs)

            mins = scipy.ndimage.gaussian_filter1d(np.array(mins), sigma=1)
            vmin = np.min(mins)

            # print and save limits to file.
            limits = [vmin, vmax]
            print(scan, limits)

            gt_min_max_path = os.path.join(gt_viz_path, f"{scan}.txt")
            with open(gt_min_max_path, 'w') as handle:
                handle.write(f"{vmin} {vmax}")

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
