import json
import logging
import os

import numpy as np
import torch
from datasets.generic_mvs_dataset import GenericMVSDataset
from torchvision import transforms
from tqdm import tqdm
from utils.generic_utils import read_image_file
from utils.geometry_utils import rotx
import PIL.Image as pil

logger = logging.getLogger(__name__)

class ARKitDataset(GenericMVSDataset):
    """ 
    Reads dataset data processed with NeuralRecon ARKit processing scripts.

    This module file contains files for pre-processing such scenes.
    
    This class does not load depth, instead returns dummy data.

    Inherits from GenericMVSDataset and implements missing methods.

    NOTE: This dataset will place NaNs where gt depth maps are invalid.
    """
    def __init__(self,
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
                native_depth_height=native_depth_height
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
            min_valid_depth, max_valid_depth: values to generate a validity mask
                for depth maps.
            image_resampling_mode: resampling method for resizing images.
        
        """

        self.min_valid_depth = min_valid_depth
        self.max_valid_depth = max_valid_depth

        self.image_resampling_mode = pil.BICUBIC

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
            scan_dir = os.path.join(self.dataset_path, 
                                        self.get_sub_folder_dir(split), scan)

            all_frame_ids = [x[2] for x 
                                in os.walk(os.path.join(scan_dir, "images"))][0]
            all_frame_ids = [x.strip(".png") for x 
                                in all_frame_ids if ".png" in x]

            all_frame_ids.sort()

            valid_frames = []

            bad_file_count = 0
            dist_to_last_valid_frame = 0
            for frame_id in all_frame_ids:

                try:
                    color_path = os.path.join(self.scenes_path, scan, "images")
                    color_filename = os.path.join(color_path, f"{frame_id}.png")
                    
                    # load intrinsics
                    _ = self.load_intrinsics(scan, frame_id)

                    # Load image
                    _ = read_image_file(color_filename, 
                            height=self.image_height, width=self.image_width)

                    # check pose
                    world_T_cam, _ = self.load_pose(scan, frame_id)
                    if (np.isnan(np.sum(world_T_cam)) or 
                        np.isinf(np.sum(world_T_cam)) or 
                        np.isneginf(np.sum(world_T_cam))
                    ):
                        dist_to_last_valid_frame+=1
                        bad_file_count+=1
                        continue

                except:
                    if not os.path.isfile(color_filename):
                        dist_to_last_valid_frame+=1
                        bad_file_count+=1
                        continue

                valid_frames.append(f"{scan} {frame_id} {dist_to_last_valid_frame}")
                dist_to_last_valid_frame = 0

            print(f"Scene {scan} has {bad_file_count} bad frame files "
                f"out of {len(all_frame_ids)}.")

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

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Either the filepath for a precached RGB file at the size 
                required, or if that doesn't exist, the full size RGB frame 
                from the dataset.

        """
        color_path = os.path.join(self.scenes_path, scan_id, "images")

        cached_resized_path = os.path.join(color_path, 
                        f"{frame_id}_{self.image_width}.png")
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
    
        # instead return the default image
        return os.path.join(color_path, f"{frame_id}.png")

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
        color_path = os.path.join(self.scenes_path, scan_id, "images")

        cached_resized_path = os.path.join(color_path, 
                        f"{frame_id}_{self.high_res_image_height}.png")
        # check if we have cached resized images on disk first
        if os.path.exists(cached_resized_path):
            return cached_resized_path
    
        # instead return the default image
        return os.path.join(color_path, f"{frame_id}.png")

    def get_cached_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the dataset's 
            configured depth resolution.

            This function is not implemented for ARKit

        """
        return ""

    def get_full_res_depth_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's depth file at the native 
            resolution in the dataset.

            This function is not implemented for ARKit
        """
        return ""

    def get_pose_filepath(self, scan_id, frame_id):
        """ returns the filepath for a frame's pose file.

            Args: 
                scan_id: the scan this file belongs to.
                frame_id: id for the frame.
            
            Returns:
                Filepath for pose information.

        """
        return os.path.join(self.scenes_path, scan_id, "poses", 
                                f"{frame_id}.txt")

    def load_target_size_depth_and_mask(self, scan_id, frame_id):
        """ Loads a depth map at the resolution the dataset is configured for.

            This function is not implemented for ARKit
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

            This function is not implemented for ARKit
        """
        
        full_res_depth = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.float32)
        full_res_mask = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.float32)
        full_res_mask_b = torch.ones((1, self.native_depth_height, 
                                self.native_depth_width), dtype=torch.bool)

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

        poses_path = self.get_pose_filepath(scan_id, frame_id)

        world_T_cam = np.genfromtxt(poses_path).astype(np.float32)
        cam_T_world = np.linalg.inv(world_T_cam)

        return world_T_cam, cam_T_world

    def load_intrinsics(self, scan_id, frame_id, flip=None):
        """ Loads intrinsics, computes scaled intrinsics, and returns a dict 
            with intrinsics matrices for a frame at multiple scales.

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

        intrinsics_path = os.path.join(self.scenes_path, scan_id, "intrinsics")
        intrinsics_filename = os.path.join(intrinsics_path, f"{frame_id}.txt")

        # intrinsics already scaled at preprocess
        K = torch.eye(4, dtype=torch.float32)
        K[0:3, 0:3] = torch.tensor(
                        np.genfromtxt(intrinsics_filename).astype(np.float32))

        # optionally include the intrinsics matrix for the full res depth map.
        if self.include_full_depth_K:
            output_dict[f"K_full_depth_b44"] = K.clone()
            output_dict[f"invK_full_depth_b44"] = torch.tensor(np.linalg.inv(K))

        # scale intrinsics to the dataset's configured depth resolution.
        K[0] *= self.depth_width / self.native_depth_width
        K[1] *= self.depth_height / self.native_depth_height

        # Get the intrinsics of all scales at various resolutions.
        for i in range(5):
            K_scaled = K.clone()
            K_scaled[:2] /= 2 ** i
            invK_scaled = np.linalg.inv(K_scaled)
            output_dict[f"K_s{i}_b44"] = K_scaled
            output_dict[f"invK_s{i}_b44"] = invK_scaled

        return output_dict

def process_data(data_path, data_source='ARKit', 
                ori_size=(1920, 1440), size=(640, 480)):
    # save image
    print('Extract images from video...')
    video_path = os.path.join(data_path, 'Frames.m4v')
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    extract_frames(video_path, out_folder=image_path, size=size)

    # load intrin and extrin
    print('Load intrinsics and extrinsics')
    sync_intrinsics_and_poses(
                    os.path.join(data_path, 'Frames.txt'), 
                    os.path.join(data_path, 'ARposes.txt'),
                    os.path.join(data_path, 'SyncedPoses.txt'),
                )

    path_dict = path_parser(data_path, data_source=data_source)
    cam_intrinsic_dict = load_camera_intrinsic(
                                path_dict['cam_intrinsic'], 
                                data_source=data_source,
                        )

    for k, v in tqdm(cam_intrinsic_dict.items(), 
                    desc='Processing camera intrinsics...'):
        cam_intrinsic_dict[k]['K'][0, :] /= (ori_size[0] / size[0])
        cam_intrinsic_dict[k]['K'][1, :] /= (ori_size[1] / size[1])
    cam_pose_dict = load_camera_pose(
        path_dict['camera_pose'], data_source=data_source)

    # save_intrinsics_extrinsics
    if not os.path.exists(os.path.join(data_path, 'poses')):
        os.mkdir(os.path.join(data_path, 'poses'))
    for k, v in tqdm(cam_pose_dict.items(), 
                                desc='Saving camera extrinsics...'):
        np.savetxt(os.path.join(data_path, 'poses', 
                                        '{}.txt'.format(k)), v, delimiter=' ')

    if not os.path.exists(os.path.join(data_path, 'intrinsics')):
        os.mkdir(os.path.join(data_path, 'intrinsics'))
    for k, v in tqdm(cam_intrinsic_dict.items(), 
                                desc='Saving camera intrinsics...'):
        np.savetxt(os.path.join(data_path, 'intrinsics', '{}.txt'.format(k)), 
                                                        v['K'], delimiter=' ')

def path_parser(root_path, data_source='TagBA'):
    full_path = root_path
    path_dict = dict()
    if data_source == 'TagBA':
        path_dict['camera_pose'] = os.path.join(
            full_path, 'TagBA', 'CameraTrajectory-BA.txt')
        path_dict['cam_intrinsic'] = os.path.join(
            full_path, 'camera_intrinsics.json')
    elif data_source == 'ARKit':
        path_dict['camera_pose'] = os.path.join(full_path, 'SyncedPoses.txt')
        path_dict['cam_intrinsic'] = os.path.join(full_path, 'Frames.txt')
    elif data_source == 'SenseAR':
        path_dict['camera_pose'] = os.path.join(full_path, 'frame_pose.csv')
        path_dict['cam_intrinsic'] = os.path.join(
            full_path, 'device_parameter.txt')
    return path_dict

def load_camera_pose(cam_pose_dir, use_homogenous=True, data_source='TagBA'):
    if cam_pose_dir is not None and os.path.isfile(cam_pose_dir):
        pass
    else:
        raise FileNotFoundError("Given camera pose dir:{} not found"
                                .format(cam_pose_dir))

    from transforms3d.quaternions import quat2mat

    pose_dict = dict()

    def process(line_data_list):
        line_data = np.array(line_data_list, dtype=float)
        fid = line_data_list[0]
        trans = line_data[1:4]
        quat = line_data[4:]
        rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
        if data_source == 'ARKit':
            rot_mat = rot_mat.dot(np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]))
            rot_mat = rotx(np.pi / 2) @ rot_mat
            trans = rotx(np.pi / 2) @ trans
        trans_mat = np.zeros([3, 4])
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans
        if use_homogenous:
            trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
        pose_dict[fid] = trans_mat

    print(f"data source: {data_source}")
    if data_source == 'TagBA' or data_source == 'ARKit':
        with open(cam_pose_dir, "r") as f:
            cam_pose_lines = f.readlines()
        for cam_line in cam_pose_lines:
            line_data_list = cam_line.split(" ")
            if len(line_data_list) == 0:
                continue
            process(line_data_list)
    elif data_source == 'SenseAR':
        import csv
        with open(cam_pose_dir, 'r') as f:
            reader = csv.reader(f, delimiter=' ', quotechar='|')
            for line_data_list in reader:
                if len(line_data_list) is not 8:
                    continue
                process(line_data_list)

    return pose_dict

def load_camera_intrinsic(cam_file, data_source='TagBA'):
    """Load camera parameter from file"""
    assert os.path.isfile(
        cam_file), "camera info:{} not found".format(cam_file)

    cam_dict = dict()
    if data_source == 'TagBA':
        with open(cam_file, "r") as f:
            cam_info = json.load(f)
        cam_dict['K'] = np.array([
            [cam_info['fx'], 0, cam_info['cx']],
            [0, cam_info['fy'], cam_info['cy']],
            [0, 0, 1]
        ], dtype=float)
        w = int(cam_info['horizontal_resolution'])
        h = int(cam_info['vertical_resolution'])
        cam_dict['shape'] = (w, h)
        cam_dict['dist_coeff'] = np.array(
            cam_info['distortion_coefficients'], dtype=float)
    elif data_source == 'Open3D':
        with open(cam_file, "r") as f:
            cam_info = json.load(f)
        cam_dict['K'] = np.array(
            cam_info['intrinsic_matrix'], dtype=float).reshape(3, 3).transpose()
        w = int(cam_info['width'])
        h = int(cam_info['height'])
        cam_dict['shape'] = (w, h)
        cam_dict['dist_coeff'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    elif data_source == 'SenseAR':
        with open(cam_file, 'r') as f:
            cam_lines = f.readlines()
            cam_dict['K'] = np.array([
                [float(cam_lines[2].split(': ')[1]), 0, 
                                            float(cam_lines[4].split(': ')[1])],
                [0, float(cam_lines[3].split(': ')[1]), 
                                            float(cam_lines[5].split(': ')[1])],
                [0, 0, 1]
            ], dtype=float)
    elif data_source == 'ARKit':
        with open(cam_file, "r") as f:
            cam_intrinsic_lines = f.readlines()

        cam_intrinsic_dict = dict()
        for line in cam_intrinsic_lines:
            line_data_list = [float(i) for i in line.split(',')]
            if len(line_data_list) == 0:
                continue
            cam_dict = dict()
            cam_dict['K'] = np.array([
                [line_data_list[2], 0, line_data_list[4]],
                [0, line_data_list[3], line_data_list[5]],
                [0, 0, 1]
            ], dtype=float)
            cam_intrinsic_dict[str(int(line_data_list[1])).zfill(5)] = cam_dict
        return cam_intrinsic_dict
    else:
        raise NotImplementedError(
            "Data parsing for source: {} not implemented".format(data_source))

    return cam_dict

def extract_frames(video_path, out_folder, size):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.resize(frame, size)
        cv2.imwrite(os.path.join(out_folder, str(i).zfill(5) + '.png'), frame)

def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    """Load camera intrinsics"""
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f:
        cam_intrinsic_lines = f.readlines()
    
    cam_intrinsics = []
    for line in cam_intrinsic_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_intrinsics.append([float(i) for i in line_data_list])

    """load camera poses"""
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()

    cam_poses = []
    for line in cam_pose_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])

    
    lines = []
    ip = 0
    length = len(cam_poses)
    for i in range(len(cam_intrinsics)):
        while (ip + 1< length and 
            abs(cam_poses[ip + 1][0] - cam_intrinsics[i][0]) < 
                                abs(cam_poses[ip][0] - cam_intrinsics[i][0])):
            ip += 1
        cam_pose = cam_poses[ip][:4] + cam_poses[ip][5:] + [cam_poses[ip][4]]
        line = [str(a) for a in cam_pose]
        # line = [str(a) for a in cam_poses[ip]]
        line[0] = str(i).zfill(5)
        lines.append(' '.join(line) + '\n')
    
    dirname = os.path.dirname(out_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_file, 'w') as f:
        f.writelines(lines)