"""Script for generating DeeoVideoMVS multiview lists in the split folder 
    indicated. It will export these frame tuples in this format line by line in 
    the output file: 

    scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    where frame_id_0 is the reference image.

    Run like so for generating a list of train tuples of eight frames (default):
    
    python ./data_scripts/generate_train_tuples.py 
        --data_config configs/data/scannet_default_train.yaml
        --num_workers 16 

    where scannet_default_train.yaml looks like: 
        !!python/object:options.Options
        dataset_path: SCANNET_PATH/
        tuple_info_file_location: $tuples_directory$
        dataset_scan_split_file: $train_split_list_location$
        dataset: scannet
        mv_tuple_file_suffix: _eight_view_deepvmvs.txt
        num_images_in_tuple: 8
        frame_tuple_type: default
        split: train

    For val, use configs/data/scannet_default_val.yaml.

    It will output a tuples file with a tuple list at:
    {tuple_info_file_location}/{split}{mv_split_filesuffix}

    This file uses the defaults for frame distances from DVMVS.

    This module also borrows the main tuple generation function from
    the DeepVideoMVS repo https://github.com/ardaduz/deep-video-mvs 
"""

import copy
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
from tools.keyframe_buffer import DVMVS_Config, is_valid_pair
from utils.dataset_utils import get_dataset


def gather_pairs_train(poses, used_pairs, 
                    is_backward, initial_pose_dist_min, initial_pose_dist_max):
    sequence_length = len(poses)
    while_range = range(0, sequence_length)

    pose_dist_min = copy.deepcopy(initial_pose_dist_min)
    pose_dist_max = copy.deepcopy(initial_pose_dist_max)

    used_measurement_indices = set()

    # Gather pairs
    check_future = False
    pairs = []

    if is_backward:
        i = sequence_length - 1
        step = -1
        first_limit = 5
        second_limit = sequence_length - 5
    else:
        i = 0
        step = 1
        first_limit = sequence_length - 5
        second_limit = 5

    loosening_counter = 0
    while i in while_range:
        pair = (i, -1)

        if check_future:
            for j in range(i + step, first_limit, step):
                if (j not in used_measurement_indices 
                                                and (i, j) not in used_pairs):

                    valid = is_valid_pair(poses[i], poses[j], 
                                        pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break
        else:
            for j in range(i - step, second_limit, -step):
                if j not in used_measurement_indices and (i, j) not in used_pairs:
                    valid = is_valid_pair(poses[i], poses[j], 
                                            pose_dist_min, pose_dist_max)

                    if valid:
                        pair = (i, j)
                        pairs.append(pair)
                        used_pairs.add(pair)
                        used_pairs.add((pair[1], pair[0]))
                        
                        used_measurement_indices.add(j)
                        pose_dist_min = copy.deepcopy(initial_pose_dist_min)
                        pose_dist_max = copy.deepcopy(initial_pose_dist_max)
                        i += step
                        check_future = False
                        loosening_counter = 0
                        break

        if pair[1] == -1:
            if check_future:
                pose_dist_min = pose_dist_min / 1.1
                pose_dist_max = pose_dist_max * 1.1
                check_future = False
                loosening_counter += 1
                if loosening_counter > 1:
                    i += step
                    loosening_counter = 0
            else:
                check_future = True
        else:
            check_future = False

    return pairs

def crawl_subprocess_short(opts_temp_filepath, scan, count, progress):
    """
    Returns a list of DVMVS train tuples according to the options at 
    opts_temp_filepath for two frame tuples.

    Args:
        opts_temp_filepath: filepath for an options config file.
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass 
                multiprocessing.Manager().Value('i', 0)
            for this.

    Returns:
        item_list: a list of strings where each string is the cocnatenated 
            scan id and frame_ids for every tuple.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    """
    scan_item_list = []

    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(config_filepaths=opts_temp_filepath, 
                                            ignore_cl_args=True)
    opts = option_handler.options

    # get dataset
    dataset_class, _ = get_dataset(opts.dataset, 
                                opts.dataset_scan_split_file, 
                                opts.single_debug_scan_id, verbose=False)

    ds = dataset_class(dataset_path=opts.dataset_path, 
                    mv_tuple_file_suffix=None,
                    split=opts.split,
                    tuple_info_file_location=opts.tuple_info_file_location,
                    pass_frame_id=True,
                    verbose_init=False)

    valid_frames = ds.get_valid_frame_ids(opts.split, scan)

    frame_ind_to_frame_id = {}
    for frame_ind, frame_line in enumerate(valid_frames):
        frame_ind_to_frame_id[frame_ind] = frame_line.strip().split(" ")[1]

    poses = []
    for frame_ind in range(len(valid_frames)):
        frame_id = frame_ind_to_frame_id[frame_ind]
        world_T_cam_44, _ = ds.load_pose(scan.rstrip("\n"), frame_id)
        poses.append(world_T_cam_44)

    samples = []
    used_pairs = set()

    for multiplier in [(1.0, False), (0.666, True), (1.5, False)]:
        pairs = gather_pairs_train(poses, used_pairs,
                                is_backward=multiplier[1],
                                initial_pose_dist_min=(multiplier[0] * 
                                    DVMVS_Config.train_minimum_pose_distance),
                                initial_pose_dist_max=(multiplier[0] * 
                                    DVMVS_Config.train_maximum_pose_distance))

        for pair in pairs:
            i, j = pair
            sample = {'scan': scan,
                      'indices': [i, j]}
            samples.append(sample)

    for sample in samples:
        chosen_frame_ids = [ds.idx_to_scan_id_frame_id(frame_ind)[1] 
                                            for frame_ind in sample["indices"]]

        cat_ids = ' '.join([str(frame_id) for frame_id in chosen_frame_ids])
        scan_item_list.append(f"{scan} {cat_ids}")

    progress.value += 1
    print(f"Completed scan {scan}, {progress.value} of total {count}\r")

    return scan_item_list

def crawl_subprocess_long(opts_temp_filepath, scan, count, progress):
    """
    Returns a list of DVMVS train tuples according to the options at 
    opts_temp_filepath for tuples longer than two frames.

    Args:
        opts_temp_filepath: filepath for an options config file.
        scan: scan to operate on.
        count: total count of multi process scans.
        progress: a Pool() progress value for tracking progress. For debugging
            you can pass 
                multiprocessing.Manager().Value('i', 0)
            for this.

    Returns:
        item_list: a list of strings where each string is the cocnatenated 
            scan id and frame_ids for every tuple.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1

    """
    scan_item_list = []

    # load options file
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options(
                                    config_filepaths=opts_temp_filepath, 
                                    ignore_cl_args=True,
                                )
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

    valid_frames = ds.get_valid_frame_ids(opts.split, scan)

    frame_ind_to_frame_id = {}
    for frame_ind, frame_line in enumerate(valid_frames):
        frame_ind_to_frame_id[frame_ind] = frame_line.strip().split(" ")[1]

    poses = []
    for frame_ind in range(len(valid_frames)):
        frame_id = frame_ind_to_frame_id[frame_ind]
        world_T_cam_44, _ = ds.load_pose(scan.rstrip("\n"), frame_id)
        poses.append(world_T_cam_44)

    subsequence_length = opts.num_images_in_tuple
    sequence_length = len(poses)

    used_pairs = set()

    usage_threshold = 1
    used_nodes = dict()
    for i in range(sequence_length):
        used_nodes[i] = 0

    calculated_step = DVMVS_Config.train_crawl_step
    samples = []
    for offset, multiplier, is_backward in [(0 % calculated_step, 1.0, False),
                                            (1 % calculated_step, 0.666, True),
                                            (2 % calculated_step, 1.5, False),
                                            (3 % calculated_step, 0.8, True),
                                            (4 % calculated_step, 1.25, False),
                                            (5 % calculated_step, 1.0, True),
                                            (6 % calculated_step, 0.666, False),
                                            (7 % calculated_step, 1.5, True),
                                            (8 % calculated_step, 0.8, False),
                                            (9 % calculated_step, 1.25, True)]:

        if is_backward:
            start = sequence_length - 1 - offset
            step = -calculated_step
            limit = subsequence_length
        else:
            start = offset
            step = calculated_step
            limit = sequence_length - subsequence_length + 1

        for i in range(start, limit, step):
            if used_nodes[i] > usage_threshold:
                continue

            sample = {'scan': scan,
                      'indices': [i]}

            previous_index = i
            valid_counter = 1
            any_counter = 1
            reached_sequence_limit = False
            while valid_counter < subsequence_length:

                if is_backward:
                    j = i - any_counter
                    reached_sequence_limit = j < 0
                else:
                    j = i + any_counter
                    reached_sequence_limit = j >= sequence_length

                if not reached_sequence_limit:
                    current_index = j

                    check1 = used_nodes[current_index] <= usage_threshold
                    check2 = (previous_index, current_index) not in used_pairs
                    check3 = is_valid_pair(
                                poses[previous_index],
                                poses[current_index],
                                (multiplier * 
                                DVMVS_Config.train_minimum_pose_distance),
                                (multiplier * 
                                DVMVS_Config.train_maximum_pose_distance),
                                t_norm_threshold=(multiplier * 
                                DVMVS_Config.train_minimum_pose_distance * 0.5))

                    if check1 and check2 and check3:
                        sample['indices'].append(current_index)
                        previous_index = copy.deepcopy(current_index)
                        valid_counter += 1
                    any_counter += 1
                else:
                    break

            if not reached_sequence_limit:
                previous_node = sample['indices'][0]
                used_nodes[previous_node] += 1
                for current_node in sample['indices'][1:]:
                    used_nodes[current_node] += 1
                    used_pairs.add((previous_node, current_node))
                    used_pairs.add((current_node, previous_node))
                    previous_node = copy.deepcopy(current_node)

                samples.append(sample)

    for sample in samples:
        chosen_frame_ids = [frame_ind_to_frame_id[frame_ind] for 
                                                frame_ind in sample["indices"]]

        cat_ids = ' '.join([str(frame_id) for frame_id in chosen_frame_ids])
        scan_item_list.append(f"{scan} {cat_ids}")

    progress.value += 1
    print(f"Completed scan {scan}, {progress.value} of total {count}\r")

    return scan_item_list

def crawl(opts_temp_filepath, opts, scans):
    """
    Multiprocessing helper for crawl_subprocess_long and crawl_subprocess_long.

    Returns a list of train tuples according to the options at 
    opts_temp_filepath.

    Args:
        opts_temp_filepath: filepath for an options config file.
        opts: options dataclass.
        scans: scans to multiprocess.

    Returns:
        item_list: a list of strings where each string is the cocnatenated 
            scan id and frame_ids for every tuple for every scan in scans.
                scan_id frame_id_0 frame_id_1 ... frame_id_N-1
    
    """
    pool = Pool(opts.num_workers)
    manager = Manager()

    count = len(scans)
    progress = manager.Value('i', 0)

    item_list = []

    crawler = (crawl_subprocess_short if opts.num_images_in_tuple == 2 
                                    else crawl_subprocess_long)

    for scan_item_list in pool.imap_unordered(
                                    partial(
                                        crawler,
                                        opts_temp_filepath,
                                        count=count,
                                        progress=progress), 
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
    
    Path(os.path.join(os.path.expanduser("~"), "tmp/")).mkdir(
                                    parents=True, exist_ok=True
                                )
    
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
    dataset_class, scan_names = get_dataset(opts.dataset, 
                                    opts.dataset_scan_split_file, 
                                    opts.single_debug_scan_id, verbose=False)

    Path(opts.tuple_info_file_location).mkdir(exist_ok=True, parents=True)
    split_filename = f"{opts.split}{opts.mv_tuple_file_suffix}"
    split_filepath = os.path.join(opts.tuple_info_file_location, split_filename)
    print(f"Saving to {split_filepath}")

    item_list = []
    if opts.single_debug_scan_id is not None:
        crawler = (crawl_subprocess_short if opts.num_images_in_tuple == 2 
                                else crawl_subprocess_long)
        item_list = crawler(
                        opts_temp_filepath, 
                        opts.single_debug_scan_id, 
                        0, Manager().Value('i', 0),
                    )
    else:
        item_list = crawl(opts_temp_filepath, opts, scan_names)

    random.shuffle(item_list)

    with open(split_filepath, "w") as f:
        for line in item_list:
            f.write(line + "\n")
    print(f"Saved to {split_filepath}")

