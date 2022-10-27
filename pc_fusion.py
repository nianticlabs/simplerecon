""" 
    Fuses depth maps into point clouds using the PC fuser from 
    https://github.com/alexrich021/3dvnet/blob/main/mv3d/eval/pointcloudfusion_custom.py

    This script follows the format in test.py. It expects a model to use for 
    depth prediction.

    Example command:
    python pc_fusion.py --name HERO_MODEL \
                --output_base_path OUTPUT_PATH \
                --config_file models/hero_model.yaml \
                --load_weights_from_checkpoint models/hero_model.ckpt \
                --data_config configs/data/scannet_default_test.yaml \
                --num_workers 8 \
                --batch_size 8;
"""

import os
from pathlib import Path

import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiment_modules.depth_model import DepthModel
import options
import tools.torch_point_cloud_fusion as torch_point_cloud_fusion
from utils.dataset_utils import get_dataset
from utils.generic_utils import (to_gpu, reverse_imagenet_normalize)

import modules.cost_volume as cost_volume

def main(opts):
    
    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    # fusion params
    N_CONSISTENT_THRESH = opts.n_consistent_thresh
    Z_THRESH = opts.pc_fusion_z_thresh
    VOXEL_DOWNSAMPLE = opts.voxel_downsample

    # output location
    pc_output_folder_name = f"{N_CONSISTENT_THRESH}_{Z_THRESH}_{VOXEL_DOWNSAMPLE}_{opts.fusion_max_depth}"

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, opts.name, 
                                        opts.dataset, opts.frame_tuple_type)

    # ouput path
    pcs_output_dir = os.path.join(results_path, "pcs", pc_output_folder_name)
    
    Path(os.path.join(pcs_output_dir)).mkdir(parents=True, exist_ok=True)
    print(f"".center(80, "#"))
    print(f" Running PC Fusion!".center(80, "#"))
    print(f"Output directory:\n{pcs_output_dir} ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")

    # load model
    model = DepthModel.load_from_checkpoint(
                                opts.load_weights_from_checkpoint,
                                args=None)
    if (opts.fast_cost_volume and  
            isinstance(model.cost_volume, cost_volume.FeatureVolumeManager)):
        model.cost_volume = model.cost_volume.to_fast()
    model = model.cuda().eval()

    with torch.inference_mode():
        for scan in tqdm(scans):
            
            # set up dataset with current scan
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
                        include_full_depth_K=True,
                        skip_frames=opts.skip_frames,
                        skip_to_frame=opts.skip_to_frame,
                        image_width=opts.image_width,
                        image_height=opts.image_height,
                        pass_frame_id=True,
                    )

            dataloader = torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=opts.batch_size, 
                                                shuffle=False, 
                                                num_workers=opts.num_workers, 
                                                drop_last=False,
                                            )

            # loop and collate data
            images_list = []
            depths_list = []
            poses_list = []
            K_list = []

            for _, batch in enumerate(tqdm(dataloader)):
                
                # get data, move to GPU
                cur_data, src_data = batch

                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                outputs = model(
                                "test", 
                                cur_data, src_data, 
                                unbatched_matching_encoder_forward=True, 
                                return_mask=True,
                            )

                depth_pred_s0_b1hw = outputs["depth_pred_s0_b1hw"].cuda()

                depth_pred_s0_b1hw[depth_pred_s0_b1hw > 
                                                    opts.fusion_max_depth] = 0

                upsampled_depth_pred = F.interpolate(
                                                depth_pred_s0_b1hw, 
                                                size=(480, 640), 
                                                mode="nearest",
                                            )

                depths_list.append(upsampled_depth_pred)

                poses_list.append(cur_data["cam_T_world_b44"].clone())
                
                K_33 = cur_data["K_s0_b44"].clone()
                K_33[:,0] *= (640/depth_pred_s0_b1hw.shape[-1]) 
                K_33[:,1] *= (480/depth_pred_s0_b1hw.shape[-2])

                K_list.append(K_33.clone())

                cur_data["high_res_color_b3hw"] = F.interpolate(
                                            cur_data["high_res_color_b3hw"], 
                                            size=(480, 640), 
                                            mode="bilinear",
                                        )
                image = cur_data["high_res_color_b3hw"].cuda()
                image = reverse_imagenet_normalize(image)
                images_list.append(image)

            # pass data to pc fuser
            depths_preds_bhw = torch.cat(depths_list, dim=0).squeeze(1)
            poses_b44 = torch.cat(poses_list, dim=0)
            image_bhw3 = torch.cat(images_list, dim=0).permute(0,2,3,1)*255
            K_b33 = torch.cat(K_list, dim=0)[:,:3,:3]

            fused_pts, fused_rgb, _ = torch_point_cloud_fusion.process_scene(
                                        depths_preds_bhw, 
                                        image_bhw3.to(torch.uint8), 
                                        poses_b44,
                                        K_b33,
                                        Z_THRESH, 
                                        N_CONSISTENT_THRESH,
                                    )
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(fused_pts)
            pcd_pred.colors = o3d.utility.Vector3dVector(fused_rgb / 255.)
            pcd_pred = pcd_pred.voxel_down_sample(VOXEL_DOWNSAMPLE)

            pcd_filepath = os.path.join(pcs_output_dir, f"{scan}.ply")
            o3d.io.write_point_cloud(pcd_filepath, pcd_pred)

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
