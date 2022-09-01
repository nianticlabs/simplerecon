
"""Script for precomputing and storing a list of valid frames per scan. A valid 
    frame is defined as one that has an existing RGB frame, an existing depth 
    map, and a valid pose. 

    Run like so for test (default):
    
    python ./scripts/precompute_valid_frames.py 
        --data_config configs/data/scannet_default_test.yaml
        --num_workers 16 
        --split test
    
    where scannet_default_test.yaml looks like: 
        !!python/object:options.Options
        dataset_path: SCANNET_PATH/
        tuple_info_file_location: SCANNET_PATH/tuples
        dataset_scan_split_file: SCANNET_PATH/scannetv2_test.txt
        dataset: scannet
        mv_tuple_file_suffix: _eight_view_deepvmvs.txt
        num_images_in_tuple: 8
        frame_tuple_type: default

    For validation, use scannet_default_val.yaml, and for train use 
    scannet_default_train.yaml

    It will save a valid_frames.txt file where the dataset class defines.

"""

import os
import sys

sys.path.append("/".join(sys.path[0].split("/")[:-1]))
import random
import string
from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import options
from utils.dataset_utils import get_dataset

def process_scan(opts_temp_filepath, scan, count, progress):
    """
    Precomputes a scan's valid frames by calling the dataset's appropriate 
    function.

    Args:
        opts_temp_filepath: filepath for an options config file.
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass 
                multiprocessing.Manager().Value('i', 0)
            for this.

    """
    item_list = []

    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(config_filepaths=opts_temp_filepath, 
                                                    ignore_cl_args=True)
    opts = option_handler.options

    # get dataset
    dataset_class, _ = get_dataset(
                            opts.dataset, 
                            opts.dataset_scan_split_file, 
                            opts.single_debug_scan_id,
                            verbose=False,
                        )

    ds = dataset_class(
                    dataset_path=opts.dataset_path, 
                    mv_tuple_file_suffix=None,
                    split=opts.split,
                    tuple_info_file_location=opts.tuple_info_file_location,
                    pass_frame_id=True,
                    verbose_init=False,
                )

    _ = ds.get_valid_frame_ids(opts.split, scan)
    
    progress.value += 1
    print(f"Completed scan {scan}, {progress.value} of total {count}.")

    return item_list

def multi_process_scans(opts_temp_filepath, opts, scans):
    """
    Multiprocessing helper for crawl_subprocess_long and crawl_subprocess_long.

    Precomputes a scan's valid frames by calling the dataset's appropriate 
    function.

    Args:
        opts_temp_filepath: filepath for an options config file.
        opts: options dataclass.
        scans: scans to multiprocess.
    """
    pool = Pool(opts.num_workers)
    manager = Manager()

    count = len(scans)
    progress = manager.Value('i', 0)

    item_list = []

    for scan_item_list in pool.imap_unordered(
                                    partial(
                                        process_scan,
                                        opts_temp_filepath,
                                        count=count,
                                        progress=progress
                                    ),
                                    scans,
                                ):
        item_list.extend(scan_item_list)

    return item_list

if __name__ == '__main__':

    
    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(ignore_cl_args=False)
    option_handler.pretty_print_options()
    opts = option_handler.options
    opts_temp_filepath = os.path.join(
        os.path.expanduser("~"),
        "tmp/", 
        ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            + ".yaml",
    )
    option_handler.save_options_as_yaml(opts_temp_filepath, opts)

    np.random.seed(42)
    random.seed(42)

    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    # get dataset
    dataset_class, scan_names = get_dataset(
                                    opts.dataset, 
                                    opts.dataset_scan_split_file, 
                                    opts.single_debug_scan_id,
                                )

    item_list = []

    Path(opts.tuple_info_file_location).mkdir(exist_ok=True, parents=True)
    split_filename = f"{opts.split}{opts.mv_tuple_file_suffix}"
    split_filepath = os.path.join(opts.tuple_info_file_location, split_filename)
    print(f"Processing valid frames.\n")

    if opts.single_debug_scan_id is not None:
        item_list = process_scan(
                        opts_temp_filepath, 
                        opts.single_debug_scan_id,
                        0, 
                        Manager().Value('i', 0),
                    )
    else:
        item_list = multi_process_scans(opts_temp_filepath, opts, scan_names)

    with open(split_filepath, "w") as f:
        for line in item_list:
            f.write(line + "\n")

    print(f"Complete")
