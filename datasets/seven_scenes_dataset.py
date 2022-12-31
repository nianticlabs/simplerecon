from doctest import debug_script
from genericpath import exists
import os
import numpy as np
import PIL.Image as pil
import torch
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from utils.generic_utils import (readlines, read_image_file)
from utils.geometry_utils import rotx
from pathlib import Path
class SevenScenesDataset(GenericMVSDataset):
    """ 
    MVS 7Scenes Dataset class for SimpleRecon.
    
    Inherits from GenericMVSDataset and implements missing methods. See 
    GenericMVSDataset for how tuples work. 

    This dataset expects 7Scenes to be in the following format:

    dataset_path
        chess
            seq-01
                frame-000000.pose.txt
                frame-000000.color.png
                frame-000000.depth.proj.png (full res depth, stored scale *1000)
                ...
            seq-02
                ...
        office
            ...
        ...

    frame-000261.pose.txt should contain pose in the form:
    9.9935108e-001	 -1.5576084e-002	  3.1508941e-002	 -1.2323361e-001
    9.2375092e-003	  9.8130137e-001	  1.9211653e-001	 -1.1206967e+000
    -3.3912845e-002	 -1.9170459e-001	  9.8083067e-001	 -9.8870575e-001
    0.0000000e+000	  0.0000000e+000	  0.0000000e+000	  1.0000000e+000

    Intrinsics are hardcoded in load_intrinsics.

    A downloade dataset should be pre-processed using 
        data_scripts/7scenes_preprocessing.py

    NOTE: This dataset will place NaNs where gt depth maps are invalid.
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
            image_resampling_mode: resampling method for resizing images.
        
        """

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        self.image_resampling_mode = pil.BICUBIC

    @staticmethod
    def get_sub_folder_dir(split):
        """ Where scans are for each split. """
        return ""

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
            
            A valid frame is one that has an existing RGB frame, an existing 
            depth file, and existing pose file where the pose isn't inf, -inf, 
            or nan.

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

        valid_frame_path =  self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames 
            # with valid data.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:
            # find out which frames have valid poses 
            print(f"Compuiting valid frames for scene {scan}.")
            
            scan_dir = os.path.join(self.dataset_path, 
                            self.get_sub_folder_dir(split), scan)

            #get scannet directories
            all_frame_ids = [x[2] for x in os.walk(scan_dir)][0]
            all_frame_ids = [x.strip("frame-").strip(".pose.txt")
                                    for x in all_frame_ids if ".pose.txt" in x]

            all_frame_ids.sort()

            scan_sub_dir = "/".join(scan_dir.split("/")[-2:])

            dist_to_last_valid_frame = 0
            bad_file_count = 0
            valid_frames = []
            for frame_id in all_frame_ids:

                color_filepath = os.path.join(scan_dir, 
                                            f"frame-{frame_id}.color.png")
                if not os.path.isfile(color_filepath):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue
                
                depth_filepath = os.path.join(scan_dir, 
                                            f"frame-{frame_id}.depth.png")
                if not os.path.isfile(depth_filepath):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue

                pose_filepath = os.path.join(scan_dir, 
                                            f"frame-{frame_id}.pose.txt")
                world_T_cam_44 = np.genfromtxt(pose_filepath)

                if (np.isnan(np.sum(world_T_cam_44)) or 
                    np.isinf(np.sum(world_T_cam_44)) or 
                    np.isneginf(np.sum(world_T_cam_44))
                ):
                    dist_to_last_valid_frame+=1
                    bad_file_count+=1
                    continue
                    
                valid_frames.append(f"{scan_sub_dir} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files out "
                f"of {len(all_frame_ids)}.")

            # store computed if we're being asked, but wrapped inside a try 
            # incase this directory is read only.
            if store_computed:
                # store those files to valid_frames.txt
                try:
                    with open(valid_frame_path, "w") as f:
                        f.write('\n'.join(valid_frames) + '\n')
                except Exception as e:
                    print(f"Couldn't save valid_frames at {valid_frame_path}, "
                    f"cause:")
                    print(e)

        return valid_frames

    def get_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's color file at the dataset's 
            configured RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)

        cached_resized_path = os.path.join(scene_path, 
                        f"frame-{frame_id}.color.{self.image_width}.png")

        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        return os.path.join(scene_path, 
                        f"frame-{frame_id}.color.png")

    def get_high_res_color_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's higher res color file at the 
            dataset's configured high RGB resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the high res 
                size required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """

        scene_path = os.path.join(self.scenes_path, scan_id)

        cached_resized_path = os.path.join(scene_path, 
                f"frame-{frame_id}.color.{self.high_res_image_height}.png")
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        return os.path.join(scene_path, 
                        f"frame-{frame_id}.color.png")

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for a precached depth file at the size 
                required.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)

        cached_resized_path = os.path.join(scene_path, 
                f"frame-{frame_id}.depth.proj.{self.depth_width}.png")

        return cached_resized_path

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the native 
            resolution in the dataset.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth file at the size 
                required, or if that doesn't exist, the full size depth frame 
                from the dataset.

        """
        scene_path = os.path.join(self.scenes_path, scan_id)

        return os.path.join(scene_path, 
                        f"frame-{frame_id}.depth.proj.png")

    def get_pose_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for pose information.

        """

        scene_path = os.path.join(self.scenes_path, scan_id)
        return os.path.join(scene_path, f"frame-{frame_id}.pose.txt")

    def load_intrinsics(self, scan_id=None, frame_id=None, flip=None):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame. Not needed for ScanNet as images 
                share intrinsics across a scene.
                flip: unused.

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


        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(525)
        K[1, 1] = float(525)
        K[0, 2] = float(320)
        K[1, 2] = float(240)

        if self.include_full_depth_K:
            # at 640 by 480
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = np.linalg.inv(K)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / 640
        K[1] *= self.depth_height / 480

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

            Internally, if the loaded depth map isn't at the target resolution,
            the depth map will be resized on-the-fly to meet that resolution.

            NOTE: This function will place NaNs where depth maps are invalid.

            Args:
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                depth: depth map at the right resolution. Will contain NaNs 
                    where depth values are invalid.
                mask: a float validity mask for the depth maps. (1.0 where depth
                is valid).
                mask_b: like mask but boolean.
        """
        depth_filepath = self.get_cached_depth_filepath(scan_id, frame_id)

        if not os.path.exists(depth_filepath):
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)

        depth = read_image_file(
                            depth_filepath,
                            height=self.depth_height,
                            width=self.depth_width,
                            value_scale_factor=1e-3,
                            resampling_mode=pil.NEAREST,
                        )

        # Get the float valid mask
        mask_b = ((depth > self.min_valid_depth) 
                                & (depth < self.max_valid_depth))
        mask = mask_b.float()

        # set invalids to nan
        depth[~mask_b] = torch.tensor(np.nan)
        
        return depth, mask, mask_b

    def load_full_res_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the native resolution the dataset provides.

            NOTE: This function will place NaNs where depth maps are invalid.

            Args:
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
                
            Returns:
                full_res_depth: depth map at the right resolution. Will contain 
                    NaNs where depth values are invalid.
                full_res_mask: a float validity mask for the depth maps. (1.0 
                where depth is valid).
                full_res_mask_b: like mask but boolean.
        """
        full_res_depth_filepath = self.get_full_res_depth_filepath(
                                                    scan_id, frame_id)
        
        # Load depth
        full_res_depth = read_image_file(full_res_depth_filepath, 
                                                value_scale_factor=1e-3)

        # Get the float valid mask
        full_res_mask_b = ((full_res_depth > self.min_valid_depth) 
                                & (full_res_depth < self.max_valid_depth))
        full_res_mask = full_res_mask_b.float()

        # set invalids to nan
        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

    def load_pose(self, scan_id, frame_id):
        """ Loads a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                world_T_cam (numpy array): matrix for transforming from the 
                    camera to the world (pose).
                cam_T_world (numpy array): matrix for transforming from the 
                    world to the camera (extrinsics).
        """
        scene_path = os.path.join(self.scenes_path, scan_id)
        sensor_data_dir = os.path.join(scene_path)
        pose_path = os.path.join(sensor_data_dir, f"frame-{frame_id}.pose.txt")

        world_T_cam = np.genfromtxt(pose_path).astype(np.float32)
        
        
        rot_mat = world_T_cam[:3,:3]
        trans = world_T_cam[:3,3]

        rot_mat = rotx(np.pi / 2) @ rot_mat
        trans = rotx(np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans
        
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world
