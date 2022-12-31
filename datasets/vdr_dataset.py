import functools
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from utils.geometry_utils import rotx
from utils.generic_utils import read_image_file
import PIL.Image as pil

logger = logging.getLogger(__name__)

class VDRDataset(GenericMVSDataset):
    """ 
    Reads a VDR scan folder.
    
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
            native_depth_width=256,
            native_depth_height=192,
        ):
        super().__init__(dataset_path=dataset_path,
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

        self.capture_metadata = {}

        self.image_resampling_mode=pil.BICUBIC

    @staticmethod
    def get_sub_folder_dir(split):
        return "scans"

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
        valid_frame_path = self.get_valid_frame_path(split, scan)

        if os.path.exists(valid_frame_path):
            # valid frame file exists, read that to find the ids of frames with 
            # valid poses.
            with open(valid_frame_path) as f:
                valid_frames = f.readlines()
        else:

            print(f"Compuiting valid frames for scene {scan}.")
            # find out which frames have valid poses 

            # load scan metadata
            self.load_capture_metadata(scan)
            color_file_count = len(self.capture_metadata[scan])

            valid_frames = []
            dist_to_last_valid_frame = 0
            bad_file_count = 0
            for frame_ind in range(len(self.capture_metadata[scan])):
                world_T_cam_44, _ = self.load_pose(scan, frame_ind)
                if (np.isnan(np.sum(world_T_cam_44)) or 
                    np.isinf(np.sum(world_T_cam_44)) or 
                    np.isneginf(np.sum(world_T_cam_44))
                ):
                    bad_file_count+=1
                    dist_to_last_valid_frame+=1
                    continue

                valid_frames.append(f"{scan} {frame_ind} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files out of "
                    f"{color_file_count}.")

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

        self.load_capture_metadata(scan_id)
        frame_metadata = self.capture_metadata[scan_id][int(frame_id)]
        

        world_T_cam = torch.tensor(frame_metadata['pose4x4'], 
                                            dtype=torch.float32).view(4, 4).T
        gl_to_cv = torch.FloatTensor([[1, -1, -1, 1], [-1, 1, 1, -1], 
                                    [-1, 1, 1, -1], [1, 1, 1, 1]])
        world_T_cam *= gl_to_cv
        world_T_cam = world_T_cam.numpy()
        rot_mat = world_T_cam[:3,:3]
        trans = world_T_cam[:3,3]

        rot_mat = rotx(-np.pi / 2) @ rot_mat
        trans = rotx(-np.pi / 2) @ trans

        world_T_cam[:3, :3] = rot_mat
        world_T_cam[:3, 3] = trans
        
        world_T_cam = world_T_cam
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def load_intrinsics(self, scan_id, frame_id, flip=None):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame. Not needed for ScanNet as images 
                share intrinsics across a scene.
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
        frame_metadata = self.capture_metadata[scan_id][int(frame_id)]

        image_width, image_height = frame_metadata['resolution']
        
        fx, fy, cx, cy, _ = frame_metadata['intrinsics']

        K = torch.eye(4, dtype=torch.float32)
        K[0, 0] = float(fx)
        K[1, 1] = float(fy)
        K[0, 2] = float(cx)
        K[1, 2] = float(cy)

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:   
            full_K = K.clone()

            full_K[0] *= (self.native_depth_width/image_width) 
            full_K[1] *= (self.native_depth_height/image_height) 

            output_dict[f"K_full_depth_b44"] = full_K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.linalg.inv(full_K)

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= (self.depth_width/image_width) 
        K[1] *= (self.depth_height/image_height)

        # Get the intrinsics of all the scales
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

    def load_capture_metadata(self, scan_id):
        """ Reads a vdr scan file and loads metadata for that scan into
            self.capture_metadata

            It does this by loading a metadata json file that contains frame 
            RGB information, intrinsics, and poses for each frame.

            Metadata for each scan is cached in the dictionary 
            self.capture_metadata.

            Args:
                scan_id: a scan_id whose metadata will be read.
        """
        if scan_id in self.capture_metadata:
            return

        metadata_path = os.path.join(
                            self.dataset_path, 
                            self.get_sub_folder_dir(self.split), 
                            scan_id, 
                            "capture.json",
                        )

        with open(metadata_path) as f:
            capture_metadata = json.load(f)

        self.capture_metadata[scan_id] = capture_metadata["frames"]

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth file at the size 
                required, or if that doesn't exist, the full size depth frame 
                from the dataset.

        """
        cached_resized_path = os.path.join(
                                    self.dataset_path, 
                                    self.get_sub_folder_dir(self.split),
                                    scan_id,
                                    f"depth.{self.depth_width}_{frame_id}.bin"
                                )

        # check if we have cached resized depth on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path, False
        
        # instead return the default image
        return os.path.join(
                        self.dataset_path,
                        self.get_sub_folder_dir(self.split),
                        scan_id,
                        f"depth_{frame_id}.bin",
                    )

    def get_cached_confidence_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth confidence file at the 
            dataset's configured depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth confidence file at the 
                size required, or if that doesn't exist, the full size depth 
                frame from the dataset.

        """
        cached_resized_path = os.path.join(
                                    self.dataset_path, 
                                    self.get_sub_folder_dir(self.split),
                                    scan_id,
                                    f"depthConfidence.{self.depth_width}_"
                                        f"{frame_id}.bin",
                                )

        # check if we have cached resized depth on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        return os.path.join(
                        self.dataset_path,
                        self.get_sub_folder_dir(self.split),
                        scan_id, 
                        f"depthConfidence_{frame_id}.bin",
                    )

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

        return os.path.join(
                        self.dataset_path, 
                        self.get_sub_folder_dir(self.split),
                        scan_id,
                        f"depth_{frame_id}.bin",
                    )

    def get_full_res_confidence_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth confidence file at the 
            dataset's maximum depth resolution.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached depth confidence file at the 
                size required, or if that doesn't exist, the full size depth 
                frame from the dataset.

        """
        return os.path.join(
                        self.dataset_path, 
                        self.get_sub_folder_dir(self.split),
                        scan_id,
                        f"depthConfidence_{frame_id}.bin",
                    )

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

        full_res_depth = torch.from_numpy(
                                np.fromfile(
                                    full_res_depth_filepath, dtype=np.float32
                                ).reshape(-1, self.native_depth_width)
                            ).unsqueeze(0)

        confidence_filepath = self.get_full_res_confidence_filepath(scan_id,
                                                                    frame_id)

        conf = torch.from_numpy(
                    np.fromfile(
                        confidence_filepath, dtype=np.uint8
                        ).reshape(-1, self.native_depth_width)).unsqueeze(0)
        
        full_res_mask_b = conf != 0
        full_res_mask = full_res_mask_b.float()

        full_res_depth[~full_res_mask_b] = torch.tensor(np.nan)

        return full_res_depth, full_res_mask, full_res_mask_b

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

        if os.path.exists(depth_filepath):
            depth = torch.from_numpy(
                        np.fromfile(depth_filepath, dtype=np.float32
                                ).reshape(-1, self.depth_width)
                    ).unsqueeze(0)
        else:
            depth_filepath = self.get_full_res_depth_filepath(scan_id, frame_id)
            depth = torch.from_numpy(
                        np.fromfile(depth_filepath, dtype=np.float32
                                ).reshape(-1, self.native_depth_width)
                    ).unsqueeze(0)

            depth = F.interpolate(
                            depth, 
                            size=(self.depth_height, self.depth_width), 
                            mode="nearest",
                        )


        confidence_filepath = self.get_cached_confidence_filepath(scan_id,
                                                            frame_id)

        if os.path.exists(confidence_filepath):
            conf = torch.from_numpy(
                        np.fromfile(confidence_filepath, dtype=np.uint8
                            ).reshape(-1, self.depth_width)
                        ).unsqueeze(0)
        else:
            confidence_filepath = self.get_full_res_confidence_filepath(scan_id, 
                                                                    frame_id)
            conf = torch.from_numpy(
                        np.fromfile(confidence_filepath, dtype=np.uint8
                            ).reshape(-1, self.native_depth_width)
                        ).unsqueeze(0)

            conf = F.interpolate(
                            conf,
                            size=(self.depth_height, self.depth_width),
                            mode="nearest"
                        )

        mask_b = conf != 0
        mask = mask_b.float()

        depth[~mask_b] = torch.tensor(np.nan)

        return depth, mask, mask_b

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
        scene_path = os.path.join(self.dataset_path,
                                self.get_sub_folder_dir(self.split), scan_id)

        cached_resized_path = os.path.join(scene_path, 
                                    f"frame.{self.image_width}_{frame_id}.jpg")

        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        return os.path.join(scene_path, 
                        f"frame_{frame_id}.jpg")

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

        scene_path = os.path.join(self.dataset_path,
                                self.get_sub_folder_dir(self.split), scan_id)

        cached_resized_path = os.path.join(scene_path, 
                        f"frame.{self.high_res_image_height}_{frame_id}.jpg")

        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
        
        # instead return the default image
        return os.path.join(scene_path, 
                        f"frame_{frame_id}.jpg")
