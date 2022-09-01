from datasets.colmap_dataset import ColmapDataset
from datasets.arkit_dataset import ARKitDataset
from datasets.scannet_dataset import ScannetDataset
from datasets.seven_scenes_dataset import SevenScenesDataset
from datasets.vdr_dataset import VDRDataset
from datasets.scanniverse_dataset import ScanniverseDataset

def get_dataset(dataset_name, 
                split_filepath,
                single_debug_scan_id=None, 
                verbose=True):
    """ Helper function for passing back the right dataset class, and helps with
        itentifying the scans in a split file.
    
        dataset_name: a string pointing to the right dataset name, allowed names
            are:
                - scannet
                - arkit: arkit format as obtained and processed by NeuralRecon
                - vdr
                - scanniverse
                - colmap: colmap text format.
                - 7scenes: processed and undistorted seven scenes.
        split_filepath: a path to a text file that contains a list of scans that
            will be passed back as a list called scans.
        single_debug_scan_id: if not None will override the split file and will 
            be passed back in scans as the only item.
        verbose: if True will print the dataset name and number of scans.

        Returns:
            dataset_class: A handle to the right dataset class for use in 
                creating objects of that class.
            scans: a lit of scans in the split file.
    """
    if dataset_name == "scannet":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]
        
        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScannetDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ScanNet Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")


    elif dataset_name == "arkit":
        
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ARKitDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" ARKit Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "vdr":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]


        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]
        
        dataset_class = VDRDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" VDR Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "scanniverse":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ScanniverseDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" Scanniverse Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "colmap":

        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = ColmapDataset
        if verbose:
            print(f"".center(80, "#"))
            print(f" Colmap Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    elif dataset_name == "7scenes":
        
        with open(split_filepath) as file:
            scans = file.readlines()
            scans = [scan.strip() for scan in scans]

        if single_debug_scan_id is not None:
            scans = [single_debug_scan_id]

        dataset_class = SevenScenesDataset

        if verbose:
            print(f"".center(80, "#"))
            print(f" 7Scenes Dataset, number of scans: {len(scans)} ".center(80, "#"))
            print(f"".center(80, "#"))
            print("")

    else:
        raise ValueError(f"Not a recognized dataset: {dataset_name}")

    return dataset_class, scans
