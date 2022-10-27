import os
import pickle
from pathlib import Path

import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from tqdm import tqdm

from experiment_modules.depth_model import DepthModel
import options
from tools import fusers_helper
from tools.mesh_renderer import (DEFAULT_CAM_FRUSTUM_MATERIAL,
                                 DEFAULT_MESH_MATERIAL, Renderer,
                                 SmoothBirdsEyeCamera, camera_marker,
                                 create_light_array, get_image_box,
                                 transform_trimesh)
from utils.dataset_utils import get_dataset
from utils.generic_utils import to_gpu
from utils.visualization_utils import colormap_image, save_viz_video_frames

import modules.cost_volume as cost_volume

def main(opts):
    print("Setting batch size to 1.")
    opts.batch_size = 1

    # get dataset
    dataset_class, scans = get_dataset(opts.dataset, 
                        opts.dataset_scan_split_file, opts.single_debug_scan_id)

    model = DepthModel.load_from_checkpoint(
                                opts.load_weights_from_checkpoint, args=None)
    if (opts.fast_cost_volume and  
            isinstance(model.cost_volume, cost_volume.FeatureVolumeManager)):
        model.cost_volume = model.cost_volume.to_fast()

    model = model.cuda().eval()

    # path where results for this model, dataset, and tuple type are.
    results_path = os.path.join(opts.output_base_path, opts.name, 
                                        opts.dataset, opts.frame_tuple_type)

    mesh_output_folder_name = f"{opts.fusion_resolution}_{opts.fusion_max_depth}_{opts.depth_fuser}"

    if opts.mask_pred_depth:
        mesh_output_folder_name += "_masked"
    if opts.fuse_color:
        mesh_output_folder_name += "_color"

    incremental_mesh_output_dir = os.path.join(
                    results_path, "incremental_meshes", mesh_output_folder_name)

    Path(incremental_mesh_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"".center(80, "#"))
    print(f" Running Fusion! Using {opts.depth_fuser} ".center(80, "#"))
    print(f"Incremental Mesh Output directory:"
                            f"\n{incremental_mesh_output_dir} ".center(80, "#"))
    if opts.use_precomputed_partial_meshes:
        print(f" Loading precomputed incremental meshes. ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")
    
    # path where cached depth maps are
    depth_output_dir = os.path.join(results_path, "depths")
    Path(os.path.join(depth_output_dir)).mkdir(parents=True, exist_ok=True)
    print(f"".center(80, "#"))
    print(f" Reading cached depths if they exist. ".center(80, "#"))
    print(f"Directory:\n{depth_output_dir} ".center(80, "#"))
    if opts.cache_depths:
        print(f" Caching depths if we need to compute them. ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")

    video_output_dir = os.path.join(results_path, "viz", 
                            "reconstruction_videos", mesh_output_folder_name)
    Path(os.path.join(video_output_dir)).mkdir(parents=True, exist_ok=True)
    print(f"".center(80, "#"))
    print(f" Outputting videos. ".center(80, "#"))
    print(f"Video Output directory:\n{video_output_dir} ".center(80, "#"))
    print(f"".center(80, "#"))
    print("")

    fpv_renderer = Renderer(height=192, width=256)
    birdseye_renderer = Renderer(height=192, width=256)

    with torch.inference_mode():
        for scan in tqdm(scans):
            smooth_birdseye = SmoothBirdsEyeCamera()

            Path(os.path.join(incremental_mesh_output_dir, 
                                    scan)).mkdir(parents=True, exist_ok=True)
            
            # initialize fuser if we need to fuse
            if opts.run_fusion:
                fuser = fusers_helper.get_fuser(opts, scan)

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
                    include_high_res_color=opts.fuse_color and opts.run_fusion,
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

            mesh_render_fpv_frames = []
            mesh_render_birdeye_frames = []
            
            viz_depth_panel = True
            all_meshes_precomputed = True
            for batch_ind, batch in enumerate(tqdm(dataloader)):
                
                # get data, move to GPU
                cur_data, src_data = batch
                if "frame_id_string" in cur_data:
                    frame_id = cur_data["frame_id_string"][0]
                else:
                    frame_id = f"{str(batch_ind):6d}"

                cur_data = to_gpu(cur_data, key_ignores=["frame_id_string"])
                src_data = to_gpu(src_data, key_ignores=["frame_id_string"])

                # To save time and compute , we should load meshes if they've 
                # all been computed and stored. We don't currently have a 
                # mechanism for picking up fusion from a partial mesh. We should 
                # only load and continue vizzing if we have a continious stream 
                # of saved meshes. If this panics, run this script without 
                # loading partial meshes
                trimesh_path = os.path.join(incremental_mesh_output_dir, scan, 
                                                            f"{frame_id}.ply")
                if not Path(trimesh_path).is_file():
                    all_meshes_precomputed=False

                if all_meshes_precomputed and opts.use_precomputed_partial_meshes:
                    scene_trimesh_mesh = trimesh.load(trimesh_path, force='mesh')

                    if viz_depth_panel:
                        pickled_depths_path = os.path.join(depth_output_dir, 
                                                    scan, f"{frame_id}.pickle")

                        if Path(pickled_depths_path).is_file():
                            with open(pickled_depths_path, 'rb') as handle:
                                outputs = pickle.load(handle)
                        else:
                            outputs = model(
                                        "test", 
                                        cur_data, 
                                        src_data, 
                                        unbatched_matching_encoder_forward=True,
                                        return_mask=True,
                                    )
                        
                        depth_pred = outputs["depth_pred_s0_b1hw"]

                else:
                    if not opts.run_fusion:
                        raise Exception("No precomputed partial mesh found and "
                                                    "run_fusion is disabled.")

                    # check if depths are precomputed.
                    pickled_depths_path = os.path.join(depth_output_dir, scan, 
                                                        f"{frame_id}.pickle")

                    if Path(pickled_depths_path).is_file():
                        with open(pickled_depths_path, 'rb') as handle:
                            outputs = pickle.load(handle)
                    else:
                        outputs = model(
                                    "test", 
                                    cur_data, 
                                    src_data, 
                                    unbatched_matching_encoder_forward=True,
                                    return_mask=True,
                                )

                        if opts.cache_depths:
                            Path(os.path.join(depth_output_dir, 
                                    scan)).mkdir(parents=True, exist_ok=True)

                            output_path = os.path.join(depth_output_dir, scan, 
                                                        f"{frame_id}.pickle")

                            outputs["K_full_depth_b44"] = cur_data["K_full_depth_b44"]
                            outputs["K_s0_b44"] = cur_data["K_s0_b44"]
                            outputs["frame_id"] = frame_id
                            if "frame_id" in src_data:
                                outputs["src_ids"] = src_data["frame_id_string"]

                            with open(output_path, 'wb') as handle:
                                pickle.dump(outputs, handle)


                    depth_pred = outputs["depth_pred_s0_b1hw"]
                    
                    if opts.mask_pred_depth:
                        overall_mask_b1hw = outputs[    
                                        "overall_mask_bhw"
                                    ].cuda().unsqueeze(1).float()
                        overall_mask_b1hw = F.interpolate(
                                                overall_mask_b1hw, 
                                                size=(192, 256), 
                                                mode="nearest"
                                            ).bool()
                        depth_pred[~overall_mask_b1hw] = 0
                    
                    color_frame = (cur_data["high_res_color_b3hw"] 
                                        if  "high_res_color_b3hw" in cur_data 
                                        else cur_data["image_b3hw"])
                    fuser.fuse_frames(depth_pred, cur_data["K_s0_b44"], 
                                                cur_data["cam_T_world_b44"], 
                                                color_frame)

                    Path(os.path.join(incremental_mesh_output_dir, 
                                    scan)).mkdir(parents=True, exist_ok=True)
                    mesh_path=os.path.join(incremental_mesh_output_dir, scan, 
                                                            f"{frame_id}.ply")
                    fuser.export_mesh(path=mesh_path)

                    if opts.fuse_color:
                        scene_trimesh_mesh = trimesh.load(trimesh_path, 
                                                                force='mesh')
                    else:
                        scene_trimesh_mesh = fuser.get_mesh(
                                                        convert_to_trimesh=True)

                world_T_cam_44 = cur_data["world_T_cam_b44"].squeeze().cpu().numpy()
                K_33 = cur_data["K_s0_b44"].squeeze().cpu().numpy()

                render_height = opts.viz_render_height
                render_width = opts.viz_render_width
                K_33[0] *= (render_width/depth_pred.shape[-1]) 
                K_33[1] *= (render_height/depth_pred.shape[-2])

                light_pos = world_T_cam_44.copy()
                light_pos[2, 3] += 5.0
                lights = create_light_array(
                                    pyrender.PointLight(intensity=30.0), 
                                    light_pos, 
                                    x_length=12, 
                                    y_length=12, 
                                    num_x=6, 
                                    num_y=6,
                                )
                meshes = ([] if scene_trimesh_mesh is None 
                                                    else [scene_trimesh_mesh])

                render_fpv = fpv_renderer.render_mesh(
                                            meshes,   
                                            render_height, render_width, 
                                            world_T_cam_44, K_33, 
                                            True, 
                                            lights=lights,
                                        )

                meshes = ([] if scene_trimesh_mesh is None 
                                                    else [scene_trimesh_mesh])
                mesh_materials = ([None] if opts.fuse_color 
                                                else [DEFAULT_MESH_MATERIAL])

                fpv_camera = trimesh.scene.Camera(
                                    resolution=(render_height, render_width), 
                                    focal=(K_33[0][0], K_33[1][1])
                                )

                cam_marker_size = 0.7
                cam_marker_mesh = camera_marker(fpv_camera, 
                                            cam_marker_size=cam_marker_size)[1]

                np_vertices = np.array(cam_marker_mesh.vertices)

                np_vertices = (world_T_cam_44 @ np.concatenate([np_vertices, 
                                    np.ones((np_vertices.shape[0], 1))], 1).T).T

                np_vertices = np_vertices/np_vertices[:,3][:,None]
                cam_marker_mesh = trimesh.Trimesh(vertices=np_vertices[:,:3], 
                                                    faces=cam_marker_mesh.faces)

                meshes.append(cam_marker_mesh)
                mesh_materials.append(DEFAULT_CAM_FRUSTUM_MATERIAL)

                our_depth_3hw = colormap_image(depth_pred.squeeze(0), 
                                                    vmin=0, vmax=3.0)
                our_depth_hw3 = our_depth_3hw.permute(1,2,0)
                pil_depth = Image.fromarray(
                                np.uint8(
                                    our_depth_hw3.cpu().detach().numpy() * 255))

                image_mesh = get_image_box(
                                    pil_depth, 
                                    cam_marker_size=cam_marker_size,
                                    fovs=(fpv_camera.fov[0], fpv_camera.fov[1])
                                )

                image_mesh = transform_trimesh(image_mesh, world_T_cam_44)
                meshes.append(image_mesh)
                mesh_materials.append(None)

                image_mesh = get_image_box(
                                    pil_depth, 
                                    cam_marker_size=cam_marker_size, 
                                    flip=True,
                                    fovs=(fpv_camera.fov[0], fpv_camera.fov[1])
                                )

                image_mesh.vertices[:,2] += 0.01
                image_mesh = transform_trimesh(image_mesh, world_T_cam_44)
                meshes.append(image_mesh)
                mesh_materials.append(None)

                birdeye_world_T_cam_44 = smooth_birdseye.get_bird_eye_trans(
                                            scene_trimesh_mesh, 
                                            fpv_pose=world_T_cam_44
                                        )
                
                if opts.back_face_alpha:
                    render_birdseye = birdseye_renderer.render_mesh_cull_composite(
                                            meshes=meshes, 
                                            height=render_height,
                                            width=render_width, 
                                            world_T_cam=birdeye_world_T_cam_44, 
                                            K=K_33, 
                                            get_colour=True, 
                                            mesh_materials=mesh_materials, 
                                            lights=lights, 
                                            alpha=opts.back_face_alpha,
                                        )
                else:
                    render_birdseye = birdseye_renderer.render_mesh(
                                    meshes, 
                                    render_height, render_width, 
                                    birdeye_world_T_cam_44, 
                                    K_33, True, mesh_materials=mesh_materials, 
                                    lights=lights,
                                )

                mesh_render_fpv_frames.append(render_fpv)
                mesh_render_birdeye_frames.append(render_birdseye)
            
            fps = (opts.standard_fps if opts.skip_frames is None 
                                else round(opts.standard_fps/opts.skip_frames))

            save_viz_video_frames(mesh_render_fpv_frames, 
                            os.path.join(video_output_dir, 
                            scan.replace("/", "_") + "_fpv.mp4"), fps=fps)
            save_viz_video_frames(mesh_render_birdeye_frames, 
                            os.path.join(video_output_dir, 
                            scan.replace("/", "_") + "_birdseye.mp4"), fps=fps)
                            
            del(dataloader)
            del(dataset)
            
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
