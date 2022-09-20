import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules.networks import MLP
from utils.generic_utils import combine_dims, tensor_B_to_bM, tensor_bM_to_B
from utils.geometry_utils import (BackprojectDepth, Project3D, get_camera_rays,
                                  pose_distance)


class CostVolumeManager(nn.Module):

    """
    Class to build a cost volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    collapsing over views by taking a dot product between each source and 
    reference feature, before summing over source views at each pixel location. 
    The final tensor is size batch_size x num_depths x H x  W tensor.
    """


    def __init__(
            self, 
            matching_height, 
            matching_width, 
            num_depth_bins=64,
            matching_dim_size=None,
            num_source_views=None,
        ):

        """
        matching_dim_size and num_source_views are not used for the standard 
        cost volume.

        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            matching_dim_size: number of channels per visual feature; the basic 
                dot product cost volume does not need this information at init.
            num_source_views: number of source views; the basic dot product cost 
                volume does not need this information at init.
        """
        super().__init__()

        self.num_depth_bins = num_depth_bins
        self.matching_height = matching_height
        self.matching_width = matching_width

        self.initialise_for_projection()


    def initialise_for_projection(self):

        """
        Set up for backwarping and projection of feature maps

        Args:
            batch_height: height of the current batch of features
            batch_width: width of the current batch of features
        """

        linear_ramp = torch.linspace(0, 1, 
                        self.num_depth_bins).view(1, self.num_depth_bins, 1, 1)
        self.register_buffer("linear_ramp_1d11", linear_ramp)

        self.backprojector = BackprojectDepth(height=self.matching_height,
                                                    width=self.matching_width)
        self.projector = Project3D()


    def get_mask(self, pix_coords_bk2hw):

        """
        Create a mask to ignore features from the edges or outside of source 
        images.
        
        Args:
            pix_coords_bk2hw: sampling locations of source features
            
        Returns:
            mask: a binary mask indicating whether to ignore a pixels
        """

        mask = torch.logical_and(
                    torch.logical_and(pix_coords_bk2hw[:, :, 0] > 2, 
                        pix_coords_bk2hw[:, :, 0] < self.matching_width - 2),
                    torch.logical_and(pix_coords_bk2hw[:, :, 1] > 2, 
                        pix_coords_bk2hw[:, :, 1] < self.matching_height - 2)
                )

        return mask


    def generate_depth_planes(self, batch_size: int, 
                                min_depth: Tensor, max_depth: Tensor) -> Tensor:
        """
        Creates a depth planes tensor of size batch_size x number of depth planes
        x matching height x matching width. Every plane contains the same depths
        and depths will vary with a log scale from min_depth to max_depth.

        Args:
            batch_size: number of these view replications to make for each 
                element in the batch.
            min_depth: minimum depth tensor defining the starting point for 
                depth planes.
            max_depth: maximum depth tensor defining the end point for 
                depth planes.

        Returns:
            depth_planes_bdhw: depth planes tensor.
        """
        linear_ramp_bd11 = self.linear_ramp_1d11.expand(
                                                batch_size, 
                                                self.num_depth_bins, 
                                                1, 
                                                1,
                                            )
        log_depth_planes_bd11 = (torch.log(min_depth) + 
                                    torch.log(max_depth / min_depth) 
                                        * linear_ramp_bd11)
        depth_planes_bd11 = torch.exp(log_depth_planes_bd11)

        depth_planes_bdhw = depth_planes_bd11.expand(
                                    batch_size, 
                                    self.num_depth_bins, 
                                    self.matching_height, 
                                    self.matching_width
                                )

        return depth_planes_bdhw


    def warp_features(
                    self, 
                    src_feats, 
                    src_extrinsics, 
                    src_Ks, 
                    cur_invK, 
                    depth_plane_b1hw, 
                    batch_size, 
                    num_src_frames, 
                    num_feat_channels,
                    uv_scale,
                ):
        """
        Warps every soruce view feature to the current view at the depth 
        plane defined by depth_plane_b1hw.

        Args:
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            depth_plane_b1hw: depth plane to use for every spatial location. For 
                SimpleRecon, this will be the same value at each location.
            batch_size: the batch size.
            num_src_frames: number of source views.
            num_feat_channels: number of feature channels for feature maps.
            uv_scale: normalization for image space coords before grid_sample.

        Returns:
            world_points_B4N: the world points at every backprojected depth 
                point in depth_plane_b1hw.
            depths: depths for each projected point in every source views.
            src_feat_warped: warped source view for every spatial location at 
                the depth plane.
            mask: depth mask where 1.0 indicated that the point projected to the
                source view is infront of the view.
        """
        
        # backproject points at that depth plane to the world, where the 
        # world is really the current view.
        world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
        world_points_B4N = world_points_b4N.repeat_interleave(num_src_frames, 
                                                                        dim=0)
        
        # project these points down to each source frame
        cam_points_B3N = self.projector(
                                    world_points_B4N, 
                                    src_Ks.view(-1, 4, 4), 
                                    src_extrinsics.view(-1, 4, 4)
                                )

        cam_points_B3hw = cam_points_B3N.view(-1, 3, self.matching_height, 
                                                            self.matching_width)
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        depths = cam_points_B3hw[:, 2:]

        uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1

        src_feat_warped = F.grid_sample(
                                    input=src_feats.view(
                                        -1, 
                                        num_feat_channels, 
                                        self.matching_height, 
                                        self.matching_width
                                    ),
                                    grid=uv_coords.type_as(src_feats),
                                    padding_mode='zeros',
                                    mode='bilinear',
                                    align_corners=False,
                                )

        # Reshape tensors to "unbatch"
        src_feat_warped = src_feat_warped.view(
                                            batch_size,
                                            num_src_frames,
                                            num_feat_channels,
                                            self.matching_height,
                                            self.matching_width,
                                        )

        depths = depths.view(
                        batch_size,
                        num_src_frames,
                        self.matching_height,
                        self.matching_width,
                    )

 
        mask_b = depths > 0
        mask = mask_b.type_as(src_feat_warped)
                             
        return world_points_B4N, depths, src_feat_warped, mask


    def build_cost_volume(
                        self, 
                        cur_feats: Tensor,
                        src_feats: Tensor,
                        src_extrinsics: Tensor,
                        src_poses: Tensor,
                        src_Ks: Tensor,
                        cur_invK: Tensor,
                        min_depth: Tensor,
                        max_depth: Tensor,
                        depth_planes_bdhw: Tensor = None,
                        return_mask: bool = False
                    ):
        """
        Build the cost volume. Using hypothesised depths, we backwarp src_feats 
        onto cur_feats using known intrinsics and take the dot product. 
        We sum the dot over all src_feats.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """

        del src_poses, return_mask

        batch_size, num_src_frames, num_feat_channels, _, _ = src_feats.shape

        uv_scale = torch.tensor(
                                [1 / self.matching_width, 
                                1 / self.matching_height], 
                                dtype=src_extrinsics.dtype, 
                                device=src_extrinsics.device
                            ).view(1, 1, 1, 2)

        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, 
                                                        min_depth, max_depth)

        # Intialize the cost volume and the counts
        all_dps = []

        # loop through depth planes
        for depth_id in range(self.num_depth_bins):

            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            _, _, src_feat_warped, mask = self.warp_features(
                                                        src_feats, 
                                                        src_extrinsics, 
                                                        src_Ks, 
                                                        cur_invK, 
                                                        depth_plane_b1hw, 
                                                        batch_size, 
                                                        num_src_frames, 
                                                        num_feat_channels,
                                                        uv_scale,
                                                    )


            # Compute the dot product between cur and src features
            dot_product_bkhw = torch.sum(
                                        src_feat_warped * 
                                            cur_feats.unsqueeze(1), 
                                        dim=2,
                                ) * mask

            # Sum over the frames
            dot_product_b1hw = dot_product_bkhw.sum(dim=1, keepdim=True)

            all_dps.append(dot_product_b1hw)

        cost_volume = torch.cat(all_dps, dim=1)

        return cost_volume, depth_planes_bdhw, None


    def indices_to_disparity(self, indices, depth_planes_bdhw):
        """ Convert cost volume indices to 1/depth for visualisation """
        depth = torch.gather(depth_planes_bdhw, dim=1, 
                                        index=indices.unsqueeze(1)).squeeze(1)
        return depth


    def forward(
            self, 
            cur_feats, 
            src_feats, 
            src_extrinsics, 
            src_poses, 
            src_Ks, 
            cur_invK, 
            min_depth, 
            max_depth, 
            depth_planes_bdhw=None, 
            return_mask=False
        ):
        """ Runs the cost volume and gets the lowest cost result """
        cost_volume, depth_planes_bdhw, overall_mask_bhw = \
                        self.build_cost_volume(
                                        cur_feats=cur_feats,
                                        src_feats=src_feats,
                                        src_extrinsics=src_extrinsics,
                                        src_Ks=src_Ks,
                                        cur_invK=cur_invK,
                                        src_poses=src_poses,
                                        min_depth=min_depth,
                                        max_depth=max_depth,
                                        depth_planes_bdhw=depth_planes_bdhw,
                                        return_mask = return_mask,
                                    )

        # for visualisation - ignore 0s in cost volume for minimum
        with torch.no_grad():
            lowest_cost = self.indices_to_disparity(
                                    torch.argmax(cost_volume.detach(), 1), 
                                    depth_planes_bdhw,
                                )

        return cost_volume, lowest_cost, depth_planes_bdhw, overall_mask_bhw


class FeatureVolumeManager(CostVolumeManager):

    """
    Class to build a feature volume from extracted features of an input 
    reference image and N source images.

    Achieved by backwarping source features onto current features using 
    hypothesised depths between min_depth_bin and max_depth_bin, and then 
    running an MLP on both visual features and each spatial and depth 
    index's metadata. The final tensor is size 
    batch_size x num_depths x H x  W tensor.

    """


    def __init__(self, 
                matching_height, 
                matching_width, 
                num_depth_bins=64, 
                mlp_channels=[202,128,128,1], 
                matching_dim_size = 16,
                num_source_views = 7):
        """
        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            mlp_channels: number of channels at every input/output of the MLP.
                mlp_channels[-1] defines the output size. mlp_channels[0] will 
                be ignored and computed in this initialization function to 
                account for all metadata.
            matching_dim_size: number of channels per visual feature.
            num_source_views: number of source views.
        """
        super().__init__(matching_height, matching_width, num_depth_bins)

        # compute dims for visual features and each metadata element
        num_visual_channels = matching_dim_size * (1 + num_source_views)
        num_depth_channels = 1 + num_source_views
        num_ray_channels = 3 * (1 + num_source_views)
        num_ray_angle_channels = num_source_views
        num_mask_channels = num_source_views
        num_num_dot_channels = num_source_views        
        num_pose_penalty_channels = 3 * (num_source_views)

        # update mlp channels
        mlp_channels[0] = (num_visual_channels
                        + num_depth_channels
                        + num_ray_channels
                        + num_ray_angle_channels
                        + num_mask_channels
                        + num_num_dot_channels
                        + num_pose_penalty_channels)

        # initialize the MLP
        self.mlp = MLP(channel_list=mlp_channels, disable_final_activation=True)

        # tell the world what's happening here.
        print(f"".center(80, "#"))
        print(f" Using FeatureVolumeManager ".center(80, "#"))
        print(f" Number of source views: ".ljust(30, " ") +
                f"{num_source_views}  ")
        print(f" Using all metadata.  ")
        print(f" Number of channels: ".ljust(30, " ") + f"{mlp_channels}  ")
        print(f"".center(80, "#"))
        print("")


    def build_cost_volume(self, 
                        cur_feats: Tensor,
                        src_feats: Tensor,
                        src_extrinsics: Tensor,
                        src_poses: Tensor,
                        src_Ks: Tensor,
                        cur_invK: Tensor,
                        min_depth: Tensor,
                        max_depth: Tensor,
                        depth_planes_bdhw: Tensor = None,
                        return_mask: bool = False,
                    ):

        """
        Build the feature volume. Using hypothesised depths, we backwarp 
        src_feats onto cur_feats using known intrinsics and run an MLP on both 
        visual features and each pixel and depth plane's metadata.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """

        (batch_size, num_src_frames, num_feat_channels, 
                            src_feat_height, src_feat_width) = src_feats.shape

        uv_scale = torch.tensor(
                        [1 / self.matching_width, 1 / self.matching_height], 
                        dtype=src_extrinsics.dtype, 
                        device=src_extrinsics.device,
                    ).view(1, 1, 1, 2)

        # construct depth planes if need be.
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, 
                                                        min_depth, max_depth)


        # get poses distances
        frame_pose_dist_B, r_measure_B, t_measure_B = pose_distance(
                                                tensor_bM_to_B(src_poses)
                                            )

        # shape all pose distance tensors.
        frame_pose_dist_bkhw = tensor_B_to_bM(
                                    frame_pose_dist_B, 
                                    batch_size=batch_size, 
                                    num_views=num_src_frames,
                                )[:,:,None,None].expand(
                                                        batch_size, 
                                                        num_src_frames, 
                                                        src_feat_height, 
                                                        src_feat_width,
                                                    )

        r_measure_bkhw = tensor_B_to_bM(
                                r_measure_B, 
                                batch_size=batch_size, 
                                num_views=num_src_frames
                            )[:,:,None,None].expand(frame_pose_dist_bkhw.shape)

        t_measure_bkhw = tensor_B_to_bM(
                                t_measure_B, 
                                batch_size=batch_size, 
                                num_views=num_src_frames,
                            )[:,:,None,None].expand(frame_pose_dist_bkhw.shape)

        # init an overall mask if need be
        overall_mask_bhw = None
        if return_mask:
            overall_mask_bhw = torch.zeros(
                        (batch_size, self.matching_height, self.matching_width),
                        device=src_feats.device,
                        dtype=torch.bool,
                    )

        # Intialize the cost volume and the counts
        all_dps = []

        # loop through depth planes
        for depth_id in range(self.num_depth_bins):
            
            # current depth plane
            depth_plane_b1hw = depth_planes_bdhw[:, depth_id].unsqueeze(1)
            
            # backproject points at that depth plane to the world, where the 
            # world is really the current view.
            world_points_b4N = self.backprojector(depth_plane_b1hw, cur_invK)
            world_points_B4N = world_points_b4N.repeat_interleave(
                                                        num_src_frames, dim=0)

            # project those points down to each source view.
            cam_points_B3N = self.projector(
                                        world_points_B4N, 
                                        src_Ks.view(-1, 4, 4), 
                                        src_extrinsics.view(-1, 4, 4)
                                    )

            cam_points_B3hw = cam_points_B3N.view(
                                            -1, 
                                            3, 
                                            self.matching_height, 
                                            self.matching_width,
                                        )

            # now sample source views at those projected points using 
            # grid_sample
            pix_coords_B2hw = cam_points_B3hw[:, :2]
            depths = cam_points_B3hw[:, 2:]

            uv_coords = 2 * pix_coords_B2hw.permute(0, 2, 3, 1) * uv_scale - 1

            # pad with zeros to bake in bounds protection when matching.
            src_feat_warped = F.grid_sample(
                                            input=src_feats.view(
                                                        -1, 
                                                        num_feat_channels, 
                                                        self.matching_height, 
                                                        self.matching_width
                                                    ),
                                            grid=uv_coords.type_as(src_feats),
                                            padding_mode='zeros',
                                            mode='bilinear',
                                            align_corners=False,
                                        )

            src_feat_warped = src_feat_warped.view(
                                                batch_size,
                                                num_src_frames,
                                                num_feat_channels,
                                                self.matching_height,
                                                self.matching_width,
                                            )

            depths = depths.view(
                            batch_size,
                            num_src_frames,
                            self.matching_height,
                            self.matching_width,
                        )

            # mask for depth validity for each image. This will be False when
            # a point in world_points_b4N is behind a source view.
            # We don't need to worry about including a pixel bounds mask as part
            # of the mlp since we're padding with zeros in grid_sample.
            mask_b = depths > 0
            mask = mask_b.type_as(src_feat_warped)
            
            if return_mask:
                # build a mask using depth validity and pixel coordinate 
                # validity by checking bounds of source views.
                depth_mask = torch.any(mask_b, dim=1)
                pix_coords_bk2hw = pix_coords_B2hw.view(
                                                    batch_size,
                                                    num_src_frames,
                                                    2,
                                                    self.matching_height,
                                                    self.matching_width,
                                                )
                bounds_mask = torch.any(self.get_mask(pix_coords_bk2hw), dim=1)
                overall_mask_bhw = torch.logical_and(depth_mask, bounds_mask)
                
                                
            # compute rays to world points for current frame 
            cur_points_rays_B3hw = F.normalize(
                                            world_points_B4N[:,:3,:], 
                                            dim=1
                                        ).view(-1,
                                                3,
                                                self.matching_height, 
                                                self.matching_width,
                                            )

            cur_points_rays_bk3hw = tensor_B_to_bM(cur_points_rays_B3hw, 
                                batch_size=batch_size, num_views=num_src_frames)
            
            # compute rays for world points source frame 
            src_poses_B44 = tensor_bM_to_B(src_poses)
            src_points_rays_B3hw = get_camera_rays(
                                                src_poses_B44,
                                                world_points_B4N[:,:3,:],
                                                in_camera_frame=False
                                            ).view(-1, 
                                                    3, 
                                                    self.matching_height, 
                                                    self.matching_width,
                                                )

            src_points_rays_bk3hw = tensor_B_to_bM(
                                                src_points_rays_B3hw, 
                                                batch_size=batch_size, 
                                                num_views=num_src_frames,
                                            )

            # combine current and source rays
            all_rays_bchw = combine_dims(
                        torch.cat(
                                [cur_points_rays_bk3hw[:,0,:,:,:][:,None,:,:,:], 
                                    src_points_rays_bk3hw],
                                dim=1,
                        ), 
                        1, 
                        3,
                    )

            # compute angle difference between rays (dot product)
            ray_angle_bkhw = F.cosine_similarity(
                                            cur_points_rays_bk3hw, 
                                            src_points_rays_bk3hw, 
                                            dim=2, 
                                            eps=1e-5
                                        )

            # Compute the dot product between cur and src features
            dot_product_bkhw = torch.sum(
                                        src_feat_warped * 
                                            cur_feats.unsqueeze(1), 
                                        dim=2,
                                    ) * mask

            # combine all visual features from across all images
            combined_visual_features_bchw = combine_dims(
                                        torch.cat(
                                            [src_feat_warped, 
                                                cur_feats.unsqueeze(1)], 
                                            dim=1,
                                        ),
                                        1,
                                        3,
                                    )

            # concat all input visual and metadata features.
            mlp_input_features_bchw = torch.cat(
                                        [
                                            combined_visual_features_bchw,
                                            mask, 
                                            depths, 
                                            depth_plane_b1hw, 
                                            dot_product_bkhw, 
                                            ray_angle_bkhw, 
                                            all_rays_bchw, 
                                            frame_pose_dist_bkhw, 
                                            r_measure_bkhw, 
                                            t_measure_bkhw
                                        ], 
                                        dim=1,
                                    )

            # run through the MLP!
            mlp_input_features_bhwc = mlp_input_features_bchw.permute(0,2,3,1)
            feature_b1hw = self.mlp(
                                mlp_input_features_bhwc
                            ).squeeze(-1).unsqueeze(1)

            # append MLP output to the final cost volume output.
            all_dps.append(feature_b1hw)

        feature_volume = torch.cat(all_dps, dim=1)

        return feature_volume, depth_planes_bdhw, overall_mask_bhw


    def to_fast(self) -> 'FastFeatureVolumeManager':
        manager = FastFeatureVolumeManager(
            self.matching_height,
            self.matching_width,
            num_depth_bins=self.num_depth_bins,
        )
        manager.mlp = self.mlp
        return manager


class FastFeatureVolumeManager(FeatureVolumeManager):
    """
    Class to build a feature volume from extracted features of an input 
    reference image and N source images.

    See FeatureVolumeManager for a full description. This class is much more 
    efficient in time, but worse for traning memory.
    """


    def __init__(self,
                matching_height, 
                matching_width, 
                num_depth_bins=64, 
                mlp_channels=[202,128,128,1], 
                matching_dim_size = 16,
                num_source_views = 7):
        """
        Args:
            matching_height: height of input feature maps
            matching_width: width of input feature maps
            num_depth_bins: number of depth planes used for warping
            mlp_channels: number of channels at every input/output of the MLP.
                mlp_channels[-1] defines the output size. mlp_channels[0] will 
                be ignored and computed in this initialization function to 
                account for all metadata.
            matching_dim_size: number of channels per visual feature.
            num_source_views: number of source views.
        """
        super().__init__(matching_height, matching_width, num_depth_bins)

        # compute dims for visual features and each metadata element
        num_visual_channels = matching_dim_size * (1 + num_source_views)
        num_depth_channels = 1 + num_source_views
        num_ray_channels = 3 * (1 + num_source_views)
        num_ray_angle_channels = num_source_views
        num_mask_channels = num_source_views
        num_num_dot_channels = num_source_views        
        num_pose_penalty_channels = 3 * (num_source_views)

        # update mlp channels
        mlp_channels[0] = (num_visual_channels
                        + num_depth_channels
                        + num_ray_channels
                        + num_ray_angle_channels
                        + num_mask_channels
                        + num_num_dot_channels
                        + num_pose_penalty_channels)

        # initialize the MLP
        self.mlp = MLP(channel_list=mlp_channels, disable_final_activation=True)

        # tell the world what's happening here.
        print(f"".center(80, "#"))
        print(f" Using FastFeatureVolumeManager ".center(80, "#"))
        print(f" Number of source views: ".ljust(30, " ") +
                f"{num_source_views}  ")
        print(f" Using all metadata.  ")
        print(f" Number of channels: ".ljust(30, " ") + f"{mlp_channels}  ")
        print(f"".center(80, "#"))
        print("")


    def warp_features(
                    self,
                    src_feats,
                    src_extrinsics,
                    src_Ks,
                    cur_invK,
                    depth_plane_bdhw,
                    batch_size,
                    num_src_frames,
                    num_feat_channels,
                    uv_scale
                ):
        """
        Warps every soruce view feature to the current view at the depth 
        plane defined by depth_plane_b1hw.

        Args:
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            depth_plane_bdhw: depth planes to use for every spatial location. 
                For SimpleRecon, this will be the same value at each location at
                each plane.
            batch_size: the batch size.
            num_src_frames: number of source views.
            num_feat_channels: number of feature channels for feature maps.
            uv_scale: normalization for image space coords before grid_sample.

        Returns:

            world_points_bkd4hw: the world points at every backprojected depth 
                point in depth_plane_b1hw.
            depths_bkdhw: depths for each projected point in every source views.
            src_feat_warped_bkdfhw: warped source view for every spatial 
                location at the depth plane.
            mask_bkdhw: depth mask where 1.0 indicated that the point projected 
                to the source view is infront of the view.
            pix_coords_bkd2hw: pixel coords where features were sampled for 
                each source view and depth plane.
        """
        num_depth_planes = depth_plane_bdhw.shape[1]
        depth_plane_B1hw = einops.rearrange(depth_plane_bdhw, 'b d h w -> (b d) 1 h w')
        
        # backproject points at that depth plane to the world, where the 
        # world is really the current view.
        world_points_B4N = self.backprojector(
                                depth_plane_B1hw,
                                einops.repeat(cur_invK, 'b i j -> (b d) i j', 
                                d=num_depth_planes),
                            )

        world_points_B4N = einops.repeat(
                                    world_points_B4N, 
                                    '(b d) i N -> (b k d) i N',
                                    k=num_src_frames,
                                    d=num_depth_planes,
                                )

        # project these points down to each source image
        cam_points_B3N = self.projector(
                                    world_points_B4N,
                                    einops.repeat(
                                                src_Ks, 
                                                'b k i j -> (b k d) i j', 
                                                k=num_src_frames,
                                                d=num_depth_planes,
                                            ),
                                    einops.repeat(
                                                src_extrinsics, 
                                                'b k i j -> (b k d) i j', 
                                                k=num_src_frames,
                                                d=num_depth_planes,
                                            ),
                                )

        cam_points_B3hw = einops.rearrange(
                                        cam_points_B3N, 
                                        'B i (h w) -> B i h w',
                                        h=self.matching_height,
                                        w=self.matching_width,
                                    )
        pix_coords_B2hw = cam_points_B3hw[:, :2]
        pix_coords_bkd2hw = einops.rearrange(
                                        pix_coords_B2hw,
                                        '(b k d) i h w -> b k d i h w',
                                        k=num_src_frames,
                                        d=num_depth_planes,
                                        b=batch_size,
                                        i=2,
                                    )
        depths_B1hw = cam_points_B3hw[:, 2:]

        uv_coords = 2 * einops.rearrange(
                                pix_coords_B2hw,
                                'B i h w -> B h w i'
                            ) * uv_scale - 1

        src_feats_Bchw = einops.repeat(
                                src_feats,
                                'b k f h w -> (b k d) f h w',
                                d=num_depth_planes,
                            )

        src_feat_warped_Bfhw = F.grid_sample(
                                        input=src_feats_Bchw,
                                        grid=uv_coords.type_as(src_feats_Bchw),
                                        padding_mode='zeros',
                                        mode='bilinear',
                                        align_corners=False,
                                    )

        # Reshape tensors to "unbatch"
        src_feat_warped_bkdfhw = einops.rearrange(
                                            src_feat_warped_Bfhw, 
                                            '(b k d) f h w -> b k d f h w',
                                            b=batch_size,
                                            k=num_src_frames,
                                            d=num_depth_planes,
                                            f=num_feat_channels,
                                            h=self.matching_height,
                                            w=self.matching_width,
                                        )

        depths_bkdhw = einops.rearrange(
                                    depths_B1hw, 
                                    '(b k d) i h w -> b k (d i) h w',
                                    b=batch_size,
                                    k=num_src_frames,
                                    d=num_depth_planes,
                                    i=1,
                                )

        # mask values landing outside the image and optionally near the border
        # mask_b = torch.logical_and(self.get_mask(pix_coords_bk2hw), depths > 0)
        mask_b = depths_bkdhw > 0
        mask_bkdhw = mask_b.type_as(src_feat_warped_bkdfhw)

        world_points_bkd4hw = einops.rearrange(
                                        world_points_B4N,
                                        '(b k d) i (h w) -> b k d i h w',
                                        b=batch_size,
                                        k=num_src_frames,
                                        d=num_depth_planes,
                                        h=self.matching_height,
                                        w=self.matching_width,
                                    )

        return world_points_bkd4hw, depths_bkdhw, src_feat_warped_bkdfhw, mask_bkdhw, pix_coords_bkd2hw


    def build_cost_volume(
                    self, 
                    cur_feats: Tensor,
                    src_feats: Tensor,
                    src_extrinsics: Tensor,
                    src_poses: Tensor,
                    src_Ks: Tensor,
                    cur_invK: Tensor,
                    min_depth: Tensor,
                    max_depth: Tensor,
                    depth_planes_bdhw: Tensor = None,
                    return_mask: bool = False,
                ):

        """
        Build the feature volume. Using hypothesised depths, we backwarp 
        src_feats onto cur_feats using known intrinsics and run an MLP on both 
        visual features and each pixel and depth plane's metadata.

        Args:
            cur_feats: current image matching features - B x C x H x W where H 
                and W should be self.matching_height and self.matching_width
            src_feats: source image matching features - B x num_src_frames x C x 
                H x W where H and W should be self.matching_height and 
                self.matching_width
            src_extrinsics: source image camera extrinsics w.r.t the current cam 
                - B x num_src_frames x 4 x 4. Will tranform from current camera
                coordinate frame to a source frame's coordinate frame.
            src_poses: source image camera poses w.r.t the current camera - B x 
                num_src_frames x 4 x 4. Will tranform from a source camera's
                coordinate frame to the current frame'ss coordinate frame.
            src_Ks: source image inverse intrinsics - B x num_src_frames x 4 x 4
            cur_invK: current image inverse intrinsics - B x 4 x 4
            min_depth: minimum depth to use at the nearest depth plane.
            max_depth: maximum depth to use at the furthest depth plane.
            depth_planes_bdhw: optionally, provide a depth plane to use instead 
                of constructing one here.
            return_mask: should we return a mask for source view information 
                w.r.t to the current image's view. When true overall_mask_bhw is 
                not None.

        Returns:
            feature_volume: the feature volume of size bdhw.
            depth_planes_bdhw: the depth planes used.
            overall_mask_bhw: None when return_mask is False, otherwise a tensor 
                of size BxHxW where True indicates a there is some valid source 
                view feature information that was used to match the current 
                view's feature against. 
        """

        batch_size, num_src_frames, num_feat_channels, src_feat_height, src_feat_width = src_feats.shape

        uv_scale = torch.tensor(
                            [1 / self.matching_width, 1 / self.matching_height], 
                            dtype=src_extrinsics.dtype,
                            device=src_extrinsics.device,
                        ).view(1, 1, 1, 2)

        # construct depth planes if need be.
        if depth_planes_bdhw is None:
            depth_planes_bdhw = self.generate_depth_planes(batch_size, 
                                                        min_depth, max_depth)

        num_depth_planes = depth_planes_bdhw.shape[1]

        # get poses distances
        frame_penalty_B, r_measure_B, t_measure_B = pose_distance(
                                                tensor_bM_to_B(src_poses)
                                            )

        # shape all pose distance tensors.
        frame_penalty_bkdhw = einops.repeat(
                                    frame_penalty_B, 
                                    '(b k) -> b k d h w',
                                    b=batch_size, 
                                    k=num_src_frames, 
                                    d=num_depth_planes,
                                    h=src_feat_height,
                                    w=src_feat_width,
                                )
        r_measure_bkdhw = einops.repeat(
                                    r_measure_B,
                                    '(b k) -> b k d h w',
                                    b=batch_size, 
                                    k=num_src_frames, 
                                    d=num_depth_planes,
                                    h=src_feat_height, 
                                    w=src_feat_width,
                                )
        t_measure_bkdhw = einops.repeat(
                                    t_measure_B, 
                                    '(b k) -> b k d h w',
                                    b=batch_size, 
                                    k=num_src_frames, 
                                    d=num_depth_planes,
                                    h=src_feat_height, 
                                    w=src_feat_width,
                                )

        # get warped and sampled features.
        world_points_bkd4hw, depths_bkdhw, src_feat_warped_bkdchw, mask_bkdhw, pix_coords_bkd2hw = self.warp_features(
            src_feats,
            src_extrinsics,
            src_Ks,
            cur_invK,
            depth_planes_bdhw,
            batch_size,
            num_src_frames,
            num_feat_channels,
            uv_scale
        )

        # init an overall mask if need be
        overall_mask_bhw = None
        if return_mask:
            depth_mask = torch.any(mask_bkdhw[:, :, -1], dim=1)
            bounds_mask = torch.any(self.get_mask(pix_coords_bkd2hw[:, :, -1]), dim=1)
            overall_mask_bhw = torch.logical_and(depth_mask, bounds_mask)

        # combine all visual features from across all images
        combined_visual_features_bdchw = torch.cat(
            [
                einops.rearrange(
                    src_feat_warped_bkdchw, 
                    'b k d c h w -> b d (k c) h w',
                ),
                einops.repeat(
                    cur_feats,
                    'b c h w -> b d c h w',
                    d=num_depth_planes,
                ),
            ], 
            dim=2,
        )

        # compute rays to world points for current frame 
        cur_points_rays_bkd3hw = F.normalize(world_points_bkd4hw[:, :, :, :3], dim=3)

        # compute rays for world points source frame 
        src_points_rays_bkd3hw = F.normalize(
            world_points_bkd4hw[:, :, :, :3] - 
                einops.repeat(
                            src_poses[:, :, :3, 3], 
                            'b k i -> b k d i h w',
                            d=num_depth_planes, 
                            h=self.matching_height,
                            w=self.matching_width, i=3
                        ),
            dim=3,
        )

        # combine current and source rays
        all_rays_bdchw = einops.rearrange(
                torch.cat(
                    [cur_points_rays_bkd3hw[:, 0:1], src_points_rays_bkd3hw],
                    dim=1,
                ),
                'b k d i h w -> b d (k i) h w',
            )

        # compute angle difference between rays (normalized dot product)
        ray_angle_bkdhw = F.cosine_similarity(cur_points_rays_bkd3hw, 
                                        src_points_rays_bkd3hw, dim=3, eps=1e-5)

        # Compute the dot product between cur and src features
        dot_product_bkdhw = torch.einsum('bkdchw,bchw->bkdhw', 
                                src_feat_warped_bkdchw, cur_feats) * mask_bkdhw

        # concat all input visual and metadata features.
        mlp_input_features_Bhwc = einops.rearrange(
                        torch.cat(
                            [
                                combined_visual_features_bdchw.transpose(1, 2),
                                mask_bkdhw,
                                depths_bkdhw,
                                depth_planes_bdhw.unsqueeze(1),
                                dot_product_bkdhw,
                                ray_angle_bkdhw,
                                all_rays_bdchw.transpose(1, 2),
                                frame_penalty_bkdhw,
                                r_measure_bkdhw,
                                t_measure_bkdhw
                            ], 
                            dim=1
                        ),
                        'b c d h w -> (b d) h w c',
                    )
                                    
        # run through the MLP!
        feature_volume_bdhw = einops.rearrange(
                                        self.mlp(mlp_input_features_Bhwc), 
                                        '(b d) h w i -> b (d i) h w',
                                        b=batch_size,
                                        d=num_depth_planes,
                                        i=1,
                                    )

        return feature_volume_bdhw, depth_planes_bdhw, overall_mask_bhw