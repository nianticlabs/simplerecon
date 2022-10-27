import os
import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))
from datasets.arkit_dataset import process_data
import options

""" Download scans and extract each to a folder with their name like so:
    dataset_path
        scans 
            neucon_demodata_b5f1
                ...
            ....
"""

option_handler = options.OptionsHandler()
option_handler.parse_and_merge_options()
opts = option_handler.options


if opts.dataset_scan_split_file is not None:
    f = open(opts.dataset_scan_split_file, "r")
    scans = f.readlines()
    scans = [scan.strip() for scan in scans]
    f.close()
elif opts.single_debug_scan_id is not None:
    scans = [opts.single_debug_scan_id]
else:
    print("No valid scans pointers.")

for scan in scans:
    path_dir = os.path.join(opts.dataset_path, "scans", scan)
    process_data(path_dir)