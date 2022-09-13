import numpy as np
import open3d as o3d
import torch
import trimesh
from datasets.scannet_dataset import ScannetDataset
from utils.generic_utils import reverse_imagenet_normalize

from tools.tsdf import TSDF, TSDFFuser


class DepthFuser():
    def __init__(
            self,
            gt_path="", 
            fusion_resolution=0.04, 
            max_fusion_depth=3.0, 
            fuse_color=False
        ):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth

class OurFuser(DepthFuser):
    """ 
    This is the fuser used for scores in the SimpleRecon paper. Note that 
    unlike open3d's fuser this implementation does not do voxel hashing. If a 
    path to a known mehs reconstruction is provided, this function will limit 
    bounds to that mesh's extent, otherwise it'll use a wide volume to prevent
    clipping.

    It's best not to use this fuser unless you need to recreate numbers from the
    paper.
    
    """
    def __init__(
            self, 
            gt_path="", 
            fusion_resolution=0.04, 
            max_fusion_depth=3, 
            fuse_color=False,
        ):
        super().__init__(
                    gt_path, 
                    fusion_resolution, 
                    max_fusion_depth, 
                    fuse_color,
                )

        if gt_path is not None:
            gt_mesh = trimesh.load(gt_path, force='mesh')
            tsdf_pred = TSDF.from_mesh(gt_mesh, voxel_size=fusion_resolution)
        else:
            bounds = {}
            bounds["xmin"] = -10.0
            bounds["xmax"] = 10.0
            bounds["ymin"] = -10.0
            bounds["ymax"] = 10.0
            bounds["zmin"] = -10.0
            bounds["zmax"] = 10.0

            tsdf_pred = TSDF.from_bounds(bounds, voxel_size=fusion_resolution)

        self.tsdf_fuser_pred = TSDFFuser(tsdf_pred, max_depth=max_fusion_depth)

    def fuse_frames(self, depths_b1hw, K_b44, 
                    cam_T_world_b44, 
                    color_b3hw):
            self.tsdf_fuser_pred.integrate_depth(
                                    depth_b1hw=depths_b1hw.half(),
                                    cam_T_world_T_b44=cam_T_world_b44.half(),
                                    K_b44=K_b44.half(),
                                )

    def export_mesh(self, path, export_single_mesh=True):
        _ = trimesh.exchange.export.export_mesh(
                        self.tsdf_fuser_pred.tsdf.to_mesh(
                                export_single_mesh=export_single_mesh),
                        path,
                    )

    def get_mesh(self, export_single_mesh=True, convert_to_trimesh=True):
        return self.tsdf_fuser_pred.tsdf.to_mesh(
                                        export_single_mesh=export_single_mesh)

class Open3DFuser(DepthFuser):
    """ 
    Wrapper class for the open3d fuser. 
    
    This wrapper does not support fusion of tensors with higher than batch 1.
    """
    def __init__(
            self, 
            gt_path="", 
            fusion_resolution=0.04, 
            max_fusion_depth=3, 
            fuse_color=False, 
            use_upsample_depth=False,
        ):
        super().__init__(
                    gt_path, 
                    fusion_resolution, 
                    max_fusion_depth,
                    fuse_color,
                )

        self.fuse_color = fuse_color
        self.use_upsample_depth = use_upsample_depth
        self.fusion_max_depth = max_fusion_depth

        voxel_size = fusion_resolution * 100
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size) / 100,
            sdf_trunc=3 * float(voxel_size) / 100,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def fuse_frames(
            self, 
            depths_b1hw, 
            K_b44, 
            cam_T_world_b44, 
            color_b3hw,
        ):

        width = depths_b1hw.shape[-1]
        height = depths_b1hw.shape[-2]

        if self.fuse_color:
            color_b3hw = torch.nn.functional.interpolate(
                                                    color_b3hw,
                                                    size=(height, width),
                                                )
            color_b3hw = reverse_imagenet_normalize(color_b3hw)
            
        for batch_index in range(depths_b1hw.shape[0]):
            if self.fuse_color:
                image_i = color_b3hw[batch_index].permute(1,2,0)

                color_im = (image_i * 255).cpu().numpy().astype(
                                                            np.uint8
                                                        ).copy(order='C')
            else:
                # mesh will now be grey
                color_im = 0.7*torch.ones_like(
                                    depths_b1hw[batch_index]
                                ).squeeze().cpu().clone().numpy()
                color_im = np.repeat(
                                color_im[:, :, np.newaxis] * 255, 
                                3,
                                axis=2
                            ).astype(np.uint8)

            depth_pred = depths_b1hw[batch_index].squeeze().cpu().clone().numpy()
            depth_pred = o3d.geometry.Image(depth_pred)
            color_im = o3d.geometry.Image(color_im)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                            color_im, 
                                            depth_pred, 
                                            depth_scale=1.0,
                                            depth_trunc=self.fusion_max_depth,
                                            convert_rgb_to_intensity=False,
                                        )
            cam_intr = K_b44[batch_index].cpu().clone().numpy()
            cam_T_world_44 = cam_T_world_b44[batch_index].cpu().clone().numpy()
            
            self.volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    width=width, 
                    height=height, fx=cam_intr[0, 0], 
                    fy=cam_intr[1, 1],
                    cx=cam_intr[0, 2],
                    cy=cam_intr[1, 2]
                ),
                cam_T_world_44,
            )

    def export_mesh(self, path, use_marching_cubes_mask=None):
        o3d.io.write_triangle_mesh(path, self.volume.extract_triangle_mesh())
    
    def get_mesh(self, export_single_mesh=None, convert_to_trimesh=False):
        mesh = self.volume.extract_triangle_mesh()

        if convert_to_trimesh:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
        
        return mesh

def get_fuser(opts, scan):
    """Returns the depth fuser required. Our fuser doesn't allow for """

    if opts.dataset == "scannet":
        gt_path = ScannetDataset.get_gt_mesh_path(opts.dataset_path, 
                                                opts.split, scan)
    else:
        gt_path = None

    if opts.depth_fuser == "ours":
        if opts.fuse_color:
            print("WARNING: fusing color using 'ours' fuser is not supported, "
                    "Color will not be fused.")

        return OurFuser(
                    gt_path=gt_path,
                    fusion_resolution=opts.fusion_resolution,
                    max_fusion_depth=opts.fusion_max_depth,
                    fuse_color=False,
                )
    if opts.depth_fuser == "open3d":
        return Open3DFuser(
                    gt_path=gt_path,
                    fusion_resolution=opts.fusion_resolution,
                    max_fusion_depth=opts.fusion_max_depth,
                    fuse_color=opts.fuse_color,
                )
    else:
        raise ValueError("Unrecognized fuser!")