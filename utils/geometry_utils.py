import kornia
import numpy as np
import torch
import torch.jit as jit
import torch.nn.functional as F
from torch import Tensor

from utils.generic_utils import batched_trace


@torch.jit.script
def to_homogeneous(input_tensor: Tensor, dim: int = 0) -> Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified 
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN


class BackprojectDepth(jit.ScriptModule):
    """
    Layer that projects points from 2D camera to 3D space. The 3D points are 
    represented in homogeneous coordinates.
    """

    def __init__(self, height: int, width: int):
        super().__init__()

        self.height = height
        self.width = width

        xx, yy = torch.meshgrid(
                            torch.arange(self.width), 
                            torch.arange(self.height), 
                            indexing='xy',
                        )
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5

        pix_coords_13N = to_homogeneous(
                                pix_coords_2hw,
                                dim=0,
                            ).flatten(1).unsqueeze(0)

        # make these tensors into buffers so they are put on the correct GPU 
        # automatically
        self.register_buffer("pix_coords_13N", pix_coords_13N)

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) -> Tensor:
        """ 
        Backprojects spatial points in 2D image space to world space using 
        invK_b44 at the depths defined in depth_b1hw. 
        """
        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N)
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N


class Project3D(jit.ScriptModule):
    """
    Layer that projects 3D points into the 2D camera
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps).view(1, 1, 1))

    @jit.script_method
    def forward(self, points_b4N: Tensor,
                K_b44: Tensor, cam_T_world_b44: Tensor) -> Tensor:
        """
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44

        cam_points_b3N = P_b44[:, :3] @ points_b4N
        
        # from Kornia and OpenCV, https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#convert_points_from_homogeneous
        mask = torch.abs(cam_points_b3N[:, 2:]) > self.eps
        depth_b1N = (cam_points_b3N[:, 2:] + self.eps)
        scale = torch.where(mask, 1.0 / depth_b1N, torch.tensor(1.0, device=depth_b1N.device))
        
        pix_coords_b2N = cam_points_b3N[:, :2] * scale
        
        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


class NormalGenerator(jit.ScriptModule):
    def __init__(self, height: int, width: int, 
                smoothing_kernel_size: int=5, smoothing_kernel_std: float=2.0):
        """ 
        Estimates normals from depth maps.
        """
        super().__init__()
        self.height = height
        self.width = width

        self.backproject = BackprojectDepth(self.height, self.width)

        self.kernel_size = smoothing_kernel_size
        self.std = smoothing_kernel_std

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) -> Tensor:
        """ 
        First smoothes incoming depth maps with a gaussian blur, backprojects 
        those depth points into world space (see BackprojectDepth), estimates
        the spatial gradient at those points, and finally uses normalized cross 
        correlation to estimate a normal vector at each location.

        """
        depth_smooth_b1hw = kornia.filters.gaussian_blur2d(
                                depth_b1hw, 
                                (self.kernel_size, self.kernel_size),
                                (self.std, self.std),
                            )
        cam_points_b4N = self.backproject(depth_smooth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)

        gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)

        return F.normalize( 
                        torch.cross(
                            gradients_b32hw[:, :, 0], 
                            gradients_b32hw[:, :, 1],
                            dim=1,
                        ),
                        dim=1,
                    )

def get_angle_dif(matA_b33, matB_b33):
    """Computes the angle difference between two rotation matrices."""
    trace = batched_trace(torch.matmul(matA_b33, 
                                            matB_b33.transpose(dim0=1, dim1=2)))
    angle_diff_b = torch.arccos((trace - 1) / 2)

    return angle_diff_b

def get_camera_rays(
            world_T_cam_b44, 
            world_points_b3N, 
            in_camera_frame, 
            cam_T_world_b44=None,
            eps=1e-4,
        ):
    """
    Computes camera rays for given camera data and points, optionally shifts 
    rays to camera frame.
    """

    if in_camera_frame:
        batch_size = world_points_b3N.shape[0]
        num_points = world_points_b3N.shape[2]
        world_points_b4N = torch.cat(
            [
                world_points_b3N,
                torch.ones(batch_size, 1, num_points).to(world_points_b3N.device),
            ],
            1,
        )
        camera_points_b3N = torch.matmul(cam_T_world_b44[:, :3, :4], 
                                                            world_points_b4N)
        rays_b3N = camera_points_b3N
    else:
        rays_b3N = world_points_b3N - world_T_cam_b44[:, 0:3, 3][:, :, None].expand(
                        world_points_b3N.shape
                    )

    rays_b3N = torch.nn.functional.normalize(rays_b3N, dim=1)

    return rays_b3N


def pose_distance(pose_b44):
    """
    DVMVS frame pose distance.
    """

    R = pose_b44[:, :3, :3]
    t = pose_b44[:, :3, 3]
    R_trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    R_measure = torch.sqrt(2 * 
                (1 - torch.minimum(torch.ones_like(R_trace)*3.0, R_trace) / 3))
    t_measure = torch.norm(t, dim=1)
    combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)

    return combined_measure, R_measure, t_measure

def qvec2rotmat(qvec):
    """
    Quaternion to 3x3 rotation matrix.
    """
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotx(t):
    """ 
    3D Rotation about the x-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])

def roty(t):
    """ 
    3D Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])

def rotz(t):
    """ 
    3D Rotation about the z-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
