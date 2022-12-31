import logging
import os
import re
import numpy as np
import PIL.Image as pil
import torch
from scipy.spatial.transform import Rotation as R
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from utils.geometry_utils import rotx

logger = logging.getLogger(__name__)

class ScanniverseDataset(GenericMVSDataset):
    """ 
    Reads a Scanniverse scan folder.
    
    self.capture_metadata is a dictionary indexed with a scan's id and is 
    populated with a scan's frame information when a frame is loaded for the 
    first time from that scan.

    This class does not load depth, instead returns dummy data.

    Inherits from GenericMVSDataset and implements missing methods.
    """

    def __init__(
            self, 
            dataset_path,
            split,
            mv_tuple_file_suffix,
            include_full_res_depth=False,
            limit_to_scan_id=None,
            num_images_in_tuple=None,
            color_transform=transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            tuple_info_file_location=None,
            image_height=384,
            image_width=512,
            high_res_image_width=640,
            high_res_image_height=480,
            image_depth_ratio=2,
            shuffle_tuple=False,
            include_full_depth_K=False,
            include_high_res_color=False,
            pass_frame_id=False,
            skip_frames=None,
            skip_to_frame=None,
            verbose_init=True,
            min_valid_depth=1e-3,
            max_valid_depth=10,
            native_depth_width=640,
            native_depth_height=480,
        ):
        super().__init__(
                dataset_path=dataset_path,
                split=split, mv_tuple_file_suffix=mv_tuple_file_suffix, 
                include_full_res_depth=include_full_res_depth, 
                limit_to_scan_id=limit_to_scan_id,
                num_images_in_tuple=num_images_in_tuple, 
                color_transform=color_transform, 
                tuple_info_file_location=tuple_info_file_location, 
                image_height=image_height, image_width=image_width, 
                high_res_image_width=high_res_image_width, 
                high_res_image_height=high_res_image_height, 
                image_depth_ratio=image_depth_ratio, shuffle_tuple=shuffle_tuple, 
                include_full_depth_K=include_full_depth_K, 
                include_high_res_color=include_high_res_color, 
                pass_frame_id=pass_frame_id, skip_frames=skip_frames, 
                skip_to_frame=skip_to_frame, verbose_init=verbose_init,
                native_depth_width=native_depth_width,
                native_depth_height=native_depth_height,
            )

        """
        Args:
            dataset_path: base path to the dataaset directory.
            split: the dataset split.
            mv_tuple_file_suffix: a suffix for the tuple file's name. The 
                tuple filename searched for wil be 
                {split}{mv_tuple_file_suffix}.
            tuple_info_file_location: location to search for a tuple file, if 
                None provided, will search in the dataset directory under 
                'tuples'.
            limit_to_scan_id: limit loaded tuples to one scan's frames.
            num_images_in_tuple: optional integer to limit tuples to this number
                of images.
            image_height, image_width: size images should be loaded at/resized 
                to. 
            include_high_res_color: should the dataset pass back higher 
                resolution images.
            high_res_image_height, high_res_image_width: resolution images 
                should be resized if we're passing back higher resolution 
                images.
            image_depth_ratio: returned gt depth maps "depth_b1hw" will be of 
                size (image_height, image_width)/image_depth_ratio.
            include_full_res_depth: if true will return depth maps from the 
                dataset at the highest resolution available.
            color_transform: optional color transform that applies when split is
                "train".
            shuffle_tuple: by default source images will be ordered according to 
                overall pose distance to the reference image. When this flag is
                true, source images will be shuffled. Only used for ablation.
            pass_frame_id: if we should return the frame_id as part of the item 
                dict
            skip_frames: if not none, will stride the tuple list by this value.
                Useful for only fusing every 'skip_frames' frame when fusing 
                depth.
            verbose_init: if True will let the init print details on the 
                initialization.
            native_depth_width, native_depth_height: the defaults here match the 
                scan processing functions used in this module. We've used fixed
                hardcoded values for the native depth resolution for loading 
                correct intrinsics, but this can definitely be automated.
            image_resampling_mode: resampling method for resizing images.
        
        """

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        self.capture_metadata = {}

        # many images will end up being upscaled.
        self.disable_resize_warning = True
        self.image_resampling_mode = pil.BICUBIC

    @staticmethod
    def get_sub_folder_dir(split):
        return "scans"

    def load_capture_metadata(self, scan_id):
        """ Reads a scanniverse scan file and loads metadata for that scan into
            self.capture_metadata

            It does this by parsing a metadata file that contains frame 
            RGB information, intrinsics, and poses for each frame.

            Metadata for each scan is cached in the dictionary 
            self.capture_metadata.

            Args:
                scan_id: a scan_id whose metadata will be read.
        """
        if scan_id in self.capture_metadata:
            return

        with open(os.path.join(self.dataset_path, 
            self.get_sub_folder_dir(self.split), scan_id, "frames.txt")) as f:
            data_lines = f.read()

        # find start strings for all frames
        frame_index_list = re.finditer("frames \{", data_lines)
        frame_index_list = [loc.start(0) for loc in frame_index_list]

        # matches new line char
        parent_end_brace_index_list = re.finditer("\n\}", data_lines) 
        parent_end_brace_index_list = [loc.start(0) + 1 for loc in 
                                                parent_end_brace_index_list]

        # find line delimiters locations for all frame index start 
        # locations.
        frame_strings = []
        for frame_info_loc in frame_index_list:
            # find the index of the closest end location after each new line at 
            # frame_info_loc
            end_brace_loc_ind = list(end_brace_loc > frame_info_loc for 
                end_brace_loc in parent_end_brace_index_list).index(True)

            # fetch the string end location
            end_brace_loc = parent_end_brace_index_list[end_brace_loc_ind]

            # get this frame's data using found start and end locations
            frame_strings.append(data_lines[frame_info_loc:end_brace_loc+1])

        # loop over frame strings, and extract information
        frames = {}
        num_0s = 0
        for frame_ind, frame_string in enumerate(frame_strings):
            frame_lines = frame_string.split("\n")
            frame_info = {}
            
            # first frame doesn't have an identifier for 0th frame, so loop 
            # until we find it, if not then 0 is the right index.
            frame_info["id"] = 0
            for line_index, line in enumerate(frame_lines):
                if "id:" in line:
                    frame_info["id"] = line.split(" ")[-1].strip()

            if frame_info["id"] == 0:
                num_0s+=1

            # get intrinsics
            frame_info["intrinsics"] = {}

            # loop until we find the camera line, then parse intrinsics.
            for line_index, line in enumerate(frame_lines):
                if "camera" in line:
                    frame_info["intrinsics"]["width"] = (
                        int(frame_lines[line_index+1].split(" ")[-1].strip()))
                    frame_info["intrinsics"]["height"] = (
                        int(frame_lines[line_index+2].split(" ")[-1].strip()))
                    frame_info["intrinsics"]["f"] = (
                        float(frame_lines[line_index+3].split(" ")[-1].strip()))
                    frame_info["intrinsics"]["px"] = (
                        float(frame_lines[line_index+4].split(" ")[-1].strip()))
                    frame_info["intrinsics"]["py"] = (
                        float(frame_lines[line_index+5].split(" ")[-1].strip()))

            # loop until we find each extrinsics line, then parse intrinsics.
            frame_info["extrinsics"] = {}
            for line_index, line in enumerate(frame_lines):
                if "rotation:" in line:
                    quadR = re.search('\[(.+?)\]', 
                                            frame_lines[line_index]).group(0)
                    quadR = quadR.replace("[","").replace("]","").split(",")
                    frame_info["extrinsics"]["quadR"] = [float(val) for 
                                                                val in quadR]

                if "translation:" in line:
                    t = re.search('\[(.+?)\]', 
                                            frame_lines[line_index]).group(0)
                    t = t.replace("[","").replace("]","").split(",")
                    frame_info["extrinsics"]["T"] = [float(val) for val in t]
            
            # some images have high resolution versions, mark those.
            frame_info["large_image"] = False
            for line_index, line in enumerate(frame_lines):
                if "is_large_image:" in line:
                    if "true" in line:
                        frame_info["large_image"] = True

            frames[str(frame_ind)] = frame_info

        if num_0s != 1:
            print(f"WARNING: {scan_id} has more than one detected 0th frame!")

        self.capture_metadata[scan_id] = frames

    def get_frame_id_string(self, frame_id):
        """ Returns an id string for this frame_id that's unique to this frame
            within the scan.

            This string is what this dataset uses as a reference to store files 
            on disk.
        """
        return frame_id

    def get_valid_frame_path(self, split, scan):
        """ returns the filepath of a file that contains valid frame ids for a 
            scan. """
        scan_dir = os.path.join(self.dataset_path, 
                            self.get_sub_folder_dir(split), scan)

        return os.path.join(scan_dir, "valid_frames.txt")

    def get_valid_frame_ids(self, split, scan, store_computed=True):
        """ Either loads or computes the ids of valid frames in the dataset for
            a scan.
            
            A valid frame is one that has an existing RGB frame and existing 
            pose file where the pose isn't inf, -inf, or nan.

            Args:
                split: the data split (train/val/test)
                scan: the name of the scan
                store_computed: store the valid_frame file where we'd expect to
                see the file in the scan folder. get_valid_frame_path defines
                where this file is expected to be. If the file can't be saved,
                a warning will be printed and the exception reason printed.

            Returns:
                valid_frames: a list of strings with info on valid frames. 
                Each string is a concat of the scan_id and the frame_id.
        """
        scan = scan.rstrip("\n")
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with 
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            print(f"Compuiting valid frames for scene {scan}.")
            # find out which frames have valid poses 

            # load metadata for this scan
            self.load_capture_metadata(scan)

            # find all valid frames by checking for pose and RGB
            bad_file_count = 0
            dist_to_last_valid_frame = 0
            valid_frames = []
            for frame_id in list(self.capture_metadata[scan].keys()):
                world_T_cam_44, _ = self.load_pose(scan, frame_id)


                if not os.path.exists(self.get_color_filepath(scan, frame_id)):
                    bad_file_count+=1
                    dist_to_last_valid_frame+=1
                    continue

                if (np.isnan(np.sum(world_T_cam_44)) or 
                    np.isinf(np.sum(world_T_cam_44)) or 
                    np.isneginf(np.sum(world_T_cam_44))
                ):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue

                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files out "
                f"of {len(list(self.capture_metadata[scan].keys()))}.")

            # store computed if we're being asked, but wrapped inside a try 
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, 'w') as f:
                        f.write('\n'.join(valid_frames) + '\n')
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, "
                        f"cause:")
                    print(e)

        return valid_frames

    def get_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's color file at the dataset's 
            configured RGB resolution.

            Scanniverse stores frames with padded integers in filenames.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        
        # check if the high resolution version is available.
        color_path = os.path.join(
                        self.dataset_path, 
                        self.get_sub_folder_dir(self.split), 
                        scan_id, 
                        "imgl", 
                        f"{int(frame_id):05d}.jpg"
                    )
        
        if os.path.exists(color_path):
            return color_path
        
        # high res isn't available, so load in normal res.
        return os.path.join(
                        self.dataset_path, 
                        self.get_sub_folder_dir(self.split),
                        scan_id,
                        "img",
                        f"{int(frame_id):05d}.jpg",
                    )

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's higher res color file at the 
            dataset's configured high RGB resolution.

            Scanniverse stores frames with padded integers in filenames.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the high res 
                size required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        return self.get_color_filepath(scan_id, frame_id)

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            This function is not implemented for Scanniverse

        """

        return ""

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the native 
            resolution in the dataset.

            This function is not implemented for Scanniverse
        """
        return ""

    def load_pose(self, scan_id, frame_id):
        """ Loads a frame's pose file.

            Will first check if the metadata for the scan this frame belongs to 
            is loaded. If not, it'll load that file into memory.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                world_T_cam (numpy array): matrix for transforming from the 
                    camera to the world (pose).
                cam_T_world (numpy array): matrix for transforming from the 
                    world to the camera (extrinsics).

        """
        self.load_capture_metadata(scan_id)
        
        frame_metadata = self.capture_metadata[scan_id][str(frame_id)]

        quad_R = frame_metadata["extrinsics"]["quadR"]
        world_T_cam = torch.eye(4)
        world_T_cam[:3,:3] = torch.tensor(R.from_quat(quad_R).as_matrix())
        world_T_cam[:3,3] = torch.tensor(frame_metadata["extrinsics"]["T"])

        world_T_cam = world_T_cam.numpy()
        
        rot_mat = world_T_cam[:3,:3]
        trans = world_T_cam[:3,3]

        rot_mat = rotx(np.pi / 2) @ rot_mat
        trans = rotx(np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans
        
        world_T_cam = world_T_cam
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def load_intrinsics(self, scan_id, frame_id, flip=None):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.

            Will first check if the metadata for the scan this frame belongs to 
            is loaded. If not, it'll load that file into memory.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
                flip: unused

            Returns:
                output_dict: A dict with
                    - K_s{i}_b44 (intrinsics) and invK_s{i}_b44 
                    (backprojection) where i in [0,1,2,3,4]. i=0 provides
                    intrinsics at the scale for depth_b1hw. 
                    - K_full_depth_b44 and invK_full_depth_b44 provides 
                    intrinsics for the maximum available depth resolution.
                    Only provided when include_full_res_depth is true. 
            
        """

        output_dict = {}

        self.load_capture_metadata(scan_id)

        frame_metadata = self.capture_metadata[scan_id][str(frame_id)]

        # intrinsics here are for RGB
        f = frame_metadata["intrinsics"]["f"]
        py = frame_metadata["intrinsics"]["py"]
        px = frame_metadata["intrinsics"]["px"]
        int_width = frame_metadata["intrinsics"]["width"]
        int_height = frame_metadata["intrinsics"]["height"]
    
        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(f)
        K[1, 1] = float(f)
        K[0, 2] = float(px)
        K[1, 2] = float(py)

        # scale to native depth resolution
        if self.include_full_depth_K:   
            full_K = K.clone()

            full_K[0] *= (self.native_depth_width/int_width) 
            full_K[1] *= (self.native_depth_height/int_height) 

            output_dict[f"K_full_depth_b44"] = full_K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.tensor(
                                                        np.linalg.inv(full_K))

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= (self.depth_width/int_width) 
        K[1] *= (self.depth_height/int_height)

        # Get the intrinsics of all the scales
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the resolution the dataset is configured for.

            This function is not implemented for Scanniverse
        """
        depth = torch.ones((1, self.depth_height, 
                                self.depth_width), dtype=torch.float32)
        mask = torch.ones((1, self.depth_height, 
                                self.depth_width), dtype=torch.float32)
        mask_b = torch.ones((1, self.depth_height, 
                                self.depth_width), dtype=torch.bool)

        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the native resolution the dataset provides.

            This function is not implemented for Scanniverse
        """
        
        full_res_depth = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.float32)
        full_res_mask = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.float32)
        full_res_mask_b = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.bool)

        return full_res_depth, full_res_mask, full_res_mask_b