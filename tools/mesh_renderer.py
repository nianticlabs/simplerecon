import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import pyrender
import trimesh
import trimesh.visual
from PIL import ImageOps

DEFAULT_MESH_MATERIAL = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.5,
                            roughnessFactor=0.8,
                            alphaMode='OPAQUE',
                            baseColorFactor=(.6, .6, .6, 1.)
                        )

DEFAULT_CAM_FRUSTUM_MATERIAL = pyrender.MetallicRoughnessMaterial(
                            metallicFactor=0.5,
                            roughnessFactor=0.8,
                            alphaMode='OPAQUE',
                            baseColorFactor=(1.0, 110/255, 0.0, 1.),
                        )

class Renderer():
    """OpenGL mesh renderer

        Used to render depthmaps from a mesh for visualization.
    """

    def __init__(self, height=480, width=640, flat_render=False):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene(ambient_light=0.4)
        self.flat_render = flat_render

    def render(self, height, width, 
                intrinsics, pose, meshes, lights=None, render_flags=None):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        for mesh in meshes:
            self.scene.add(mesh)

        cam = pyrender.IntrinsicsCamera(
                                cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                            )
        pose = self.fix_pose(pose)
        self.scene.add(cam, pose=pose)

        # lighting
        
        if lights is not None:
            for light in lights:
                self.scene.add(light[0], pose=light[1])

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            if render_flags is not None:
                return self.renderer.render(self.scene, render_flags)
            else:
                return self.renderer.render(self.scene)

    def __call__(self, height, width, intrinsics, pose, meshes):
        return self.render(height, width, intrinsics, pose, meshes)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh, mesh_material=None):
        if mesh_material is None:
            return pyrender.Mesh.from_trimesh(mesh)
        else:
            return pyrender.Mesh.from_trimesh(mesh, mesh_material)

    def delete(self):
        self.renderer.delete()

    def render_mesh(
            self,
            meshes,
            height,
            width,
            world_T_cam,
            K, 
            get_colour=False,
            mesh_materials=None,
            lights=None, 
            render_flags=None,
        ):

        if isinstance(meshes, list):
            meshes = meshes.copy()

        try:
            for mesh_ind, mesh in enumerate(meshes):
                if mesh_materials is None:
                    meshes[mesh_ind] = self.mesh_opengl(mesh)
                else:
                    meshes[mesh_ind] = self.mesh_opengl(mesh, 
                                                    mesh_materials[mesh_ind])


            rendered_colour, rendered_depth = self.render(
                                                height, 
                                                width,
                                                K,
                                                world_T_cam,
                                                meshes,
                                                lights=lights,
                                                render_flags=render_flags,
                                            )

            if get_colour:
                return rendered_colour
            else:
                return rendered_depth

        except pyrender.renderer.GLError:
            print('opengl error')
            return None
    
    def render_mesh_cull_composite(self, alpha, **kwargs):
        """ Renders a composite where backfaces are rendered in with an alpha """
        culled_render = self.render_mesh(**kwargs)
        non_culled_render = self.render_mesh(**kwargs, 
                            render_flags=pyrender.RenderFlags.SKIP_CULL_FACES)

        return culled_render * (1-alpha) + non_culled_render * alpha
        

def render_colour(renderer, meshes, world_T_cam, K, height=256, width=320):
    colour = renderer.render_mesh(meshes, height, width, 
                                            world_T_cam, K, get_colour=True)
    if colour is None:
        del renderer
        renderer = Renderer()
        colour = renderer.render_mesh(meshes, height, width, 
                                            world_T_cam, K, get_colour=True)
        
    return colour


class SmoothBirdsEyeCamera():
    """ Creates a smoothed birdseye view of a scene using information on what 
        the scene looks (existing mesh being rendered) and the current fpv 
        camera location. NOTE: Follows Scannet convention, so Z is up."""

    def __init__(
            self,
            look_at_moving_alpha=0.9,
            mean_mesh_moving_alpha=np.array([0.8,0.8,0.8]),
        ):

        self.current_cam_loc = None
        self.current_look_at = None
        self.current_mean_loc = None
        self.fpv_cam_look_at = None

        self.time_steps = 0

        self.look_at_moving_alpha = look_at_moving_alpha
        self.mean_mesh_moving_alpha = mean_mesh_moving_alpha

        return

    def get_bird_eye_trans(self, 
                        trimesh_mesh, 
                        fpv_pose=None,
                        z_offset=6,
                        backwards_offset=7,
                    ):
        """ Returns a transform for a birdseye camera smoothed for this 
            timestep. The location will be some distance (backwards_offset) 
            behind where the mean of the scene and the location of the fpv 
            camera are, but the height will be a fixed distance above the scene 
            (z_offset). 
            
            The look-at vector will be some smoothed vector looking towards what 
            the fpv is lookig at.
         """
        # if a mesh is provided, then use the mean location of those vertices
        # to get an idea of where the scene lies. If not provided, just use the 
        # location of the fpv camera. 

        if trimesh_mesh is not None:
            mean_mesh_location = np.array(trimesh_mesh.vertices).mean(0)
            mean_mesh_location += fpv_pose[:3,3] * 5
            mean_mesh_location = mean_mesh_location/6.0
        else:
            mean_mesh_location = fpv_pose[:3,3].copy()

        # keep a rolling averager of where the mesh is on average
        if self.current_mean_loc is None:
            self.current_mean_loc = mean_mesh_location
        else:
            self.current_mean_loc = self.mean_mesh_moving_alpha * self.current_mean_loc \
                                            + (1-self.mean_mesh_moving_alpha) * mean_mesh_location

        # get the lookat of the fpv camera        
        fpv_cam_transform = np.linalg.inv(fpv_pose[:3,:3])
        z_vec = np.zeros((3,))
        # z is y axis in ScanNet
        z_vec[1] = -1
        current_fpv_look_at = fpv_cam_transform @ z_vec

        # smooth the look vector over time.
        if self.fpv_cam_look_at is None:
            self.fpv_cam_look_at = current_fpv_look_at
        else:
            self.fpv_cam_look_at = 0.05*current_fpv_look_at + 0.95*self.fpv_cam_look_at
            self.fpv_cam_look_at = self.fpv_cam_look_at/np.linalg.norm(self.fpv_cam_look_at)
        
        # we only want sideways components from the look at vector, so ignore 
        # the z component (up).
        offset_vec = self.fpv_cam_look_at/np.linalg.norm(self.fpv_cam_look_at[:2])

        # offset the location of to the mean location minus 7m along the vector
        # of the lookat.
        birdseye_location = self.current_mean_loc - offset_vec * backwards_offset

        # height is a fixed value above the scene.
        birdseye_location[2] = self.current_mean_loc[2] + z_offset
        self.current_cam_loc = birdseye_location
        
        # compute the new lookat for the birdseye view as the difference between
        # where the birdseye camera is and the current fpv location. 
        self.current_look_at = self.current_mean_loc - self.current_cam_loc
        self.current_look_at = self.current_look_at/np.linalg.norm(self.current_look_at)


        # now get a rotation matrix from the lookat vector, construct R and t.
        cam_R = np.zeros((3,3))
        cam_t = np.zeros((3,))
        cam_mat = np.identity(4)

        temp_vec = np.zeros((3,))
        temp_vec[2] = 1

        right_vec = np.cross(self.current_look_at, temp_vec)
        up_vec = np.cross(self.current_look_at, right_vec)

        cam_t = self.current_cam_loc
        cam_R[:,0] = right_vec
        cam_R[:,1] = up_vec
        cam_R[:,2] = self.current_look_at

        cam_mat[:3,:3] = cam_R
        cam_mat[:3,3] = cam_t


        return cam_mat 

def camera_marker(
                camera,
                cam_marker_size=0.4,
                rect_width=0.04,
                sphere_rad=0.08,
                origin_size=None,
            ):
    """
    Creates a visual marker mesh for a camera with specific geometry.

    Args:
        camera: trimesh.scene.Camera defining camera intriniscs.
        cam_marker_size: marker scale. Larger makes a larger marker.
        rect_width: width of each rectangular leg in the frustum.
        sphere_rad: radius of the sphere at the origin.
        origin_size: size of the trimesh axis object.

    Returns:
        meshes: contains a trimesh camera axis at the origin and a mesh for 
            colorful inflated camera frustum.
    """

    # create sane origin size from marker height
    if origin_size is None:
        origin_size = cam_marker_size / 10.0

    # append the visualizations to an array
    meshes = [trimesh.creation.axis(origin_size=origin_size)]

    # calculate vertices from camera FOV angles
    x = cam_marker_size * np.tan(np.deg2rad(camera.fov[0]) / 2.0)
    y = cam_marker_size * np.tan(np.deg2rad(camera.fov[1]) / 2.0)
    z = cam_marker_size
    # combine the points into the vertices of an FOV visualization
    points = np.array(
        [(0, 0, 0),
         (-x, -y, z),
         (x, -y, z),
         (x, y, z),
         (-x, y, z)],
        dtype=float)

    # create line segments for the FOV visualization
    # a segment from the origin to each bound of the FOV
    segments = np.column_stack(
        (np.zeros_like(points), points)).reshape(
        (-1, 3))

    mesh_vertices = []
    mesh_faces = []

    #### diagonals

    sphere = trimesh.creation.icosphere(radius=sphere_rad)
    mesh_vertices.append(sphere.vertices)
    mesh_faces.append(sphere.faces)

    max_face_val = sphere.faces.max()

    rect = trimesh.creation.box([rect_width,rect_width,cam_marker_size])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += 0.5 * cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] + np.tan(np.deg2rad(camera.fov[1]) / 2.0) * np_vertices[:,2]
    np_vertices[:,1] = np_vertices[:,1] + np.tan(np.deg2rad(camera.fov[0]) / 2.0) * np_vertices[:,2]
    
    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)
    
    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([rect_width,rect_width,cam_marker_size])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += 0.5 * cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] - np.tan(np.deg2rad(camera.fov[1]) / 2.0) * np_vertices[:,2]
    np_vertices[:,1] = np_vertices[:,1] + np.tan(np.deg2rad(camera.fov[0]) / 2.0) * np_vertices[:,2]
    
    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)
    
    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([rect_width,rect_width,cam_marker_size])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += 0.5 * cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] - np.tan(np.deg2rad(camera.fov[1]) / 2.0) * np_vertices[:,2]
    np_vertices[:,1] = np_vertices[:,1] - np.tan(np.deg2rad(camera.fov[0]) / 2.0) * np_vertices[:,2]
    
    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)

    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([rect_width,rect_width,cam_marker_size])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += 0.5 * cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] + np.tan(np.deg2rad(camera.fov[1]) / 2.0) * np_vertices[:,2]
    np_vertices[:,1] = np_vertices[:,1] - np.tan(np.deg2rad(camera.fov[0]) / 2.0) * np_vertices[:,2]
    
    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)
    
    max_face_val = rect.faces.max()

    ### rect at far plane
    rect = trimesh.creation.box([np.tan(np.deg2rad(camera.fov[1]) / 2.0)*2.0*cam_marker_size,rect_width,rect_width])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += cam_marker_size
    np_vertices[:,1] = np_vertices[:,1] - np.tan(np.deg2rad(camera.fov[0]) / 2.0)*cam_marker_size

    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)

    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([np.tan(np.deg2rad(camera.fov[1]) / 2.0)*2.0*cam_marker_size,rect_width,rect_width])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += cam_marker_size
    np_vertices[:,1] = np_vertices[:,1] + np.tan(np.deg2rad(camera.fov[0]) / 2.0)*cam_marker_size

    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)

    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([rect_width,np.tan(np.deg2rad(camera.fov[0]) / 2.0)*2.0*cam_marker_size,rect_width])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] - np.tan(np.deg2rad(camera.fov[1]) / 2.0)*cam_marker_size

    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)

    max_face_val = rect.faces.max()

    rect = trimesh.creation.box([rect_width,np.tan(np.deg2rad(camera.fov[0]) / 2.0)*2.0*cam_marker_size,rect_width])
    np_vertices = np.array(rect.vertices)
    np_vertices[:,2] += cam_marker_size
    np_vertices[:,0] = np_vertices[:,0] + np.tan(np.deg2rad(camera.fov[1]) / 2.0)*cam_marker_size

    mesh_vertices.append(np_vertices)
    rect.faces += max_face_val+1
    mesh_faces.append(rect.faces)

    max_face_val = rect.faces.max()

    ### assemble final mesh

    vertices = np.concatenate(mesh_vertices, 0)
    faces = np.concatenate(mesh_faces, 0)
    
    frustum = trimesh.Trimesh(vertices=vertices, faces=faces)

    meshes.append(frustum)

    return meshes

def get_image_box(
                pil_image, 
                aspect_ratio=4.0/3.0, 
                cam_marker_size=1.0, 
                flip=False, 
                fovs=None,
            ):
    """ Gets a textured mesh of an image. """
    if fovs is not None:
        width = cam_marker_size*np.tan(np.deg2rad(fovs[1]) / 2.0) * 2.0
        height = cam_marker_size*np.tan(np.deg2rad(fovs[0]) / 2.0) * 2.0
    else:
        height = 1.0
        width = height*aspect_ratio
        width*=cam_marker_size
        height*=cam_marker_size

    if flip:
        pil_image = ImageOps.mirror(pil_image)
        width = -width

    vertices = np.zeros((4,3))
    vertices[0,:] = [width/2, height/2, cam_marker_size]
    vertices[1,:] = [width/2, -height/2, cam_marker_size]
    vertices[2,:] = [-width/2, -height/2, cam_marker_size]
    vertices[3,:] = [-width/2, height/2, cam_marker_size]


    faces = np.zeros((2,3))
    faces[0,:] = [0,1,2]
    faces[1,:] = [2,3,0]
    # faces[2,:] = [2,3]
    # faces[3,:] = [3,0]

    uvs = np.zeros((4,2))

    uvs[0,:] = [1.0,0]
    uvs[1,:] = [1.0,1.0]
    uvs[2,:] = [0,1.0]
    uvs[3,:] = [0,0]

    face_normals = np.zeros((2,3))
    face_normals[0,:] = [0.0, 0.0, 1.0]
    face_normals[1,:] = [0.0, 0.0, 1.0]


    material =  trimesh.visual.texture.SimpleMaterial(
                                                image=pil_image, 
                                                ambient=(1.0,1.0,1.0,1.0), 
                                                diffuse=(1.0,1.0,1.0,1.0),
                                            )    
    texture = trimesh.visual.TextureVisuals(
                                        uv=uvs, 
                                        image=pil_image,
                                        material=material,
                                    )

    mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_normals=face_normals,
                visual=texture,
                validate=True,
                process=False
            )

    return mesh

def create_light_array(light_type, center_loc, 
                    x_length=10.0, y_length=10.0, 
                    num_x=5, num_y=5):
    """" Creates an array of lights. """
    lights = []

    # lights.append([light_type, center_loc])


    x_offsets = np.linspace(-x_length, x_length, num_x).squeeze()
    y_offsets = np.linspace(-y_length, y_length, num_y).squeeze()

    X, Y = np.meshgrid(x_offsets, y_offsets)

    coords = np.stack([X,Y], 2).reshape(num_x*num_y,2)

    for coord_ind in range(num_y*num_x):
        x = coords[coord_ind, 0]
        y = coords[coord_ind, 1]
        corner_pos = center_loc.copy()
        corner_pos[2, 0] += x
        corner_pos[2, 1] += y
        lights.append([light_type, corner_pos])

    return lights


def transform_trimesh(mesh, transform):
    """ Applies a transform to a trimesh. """
    np_vertices = np.array(mesh.vertices)
    np_vertices = (transform @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T).T
    np_vertices = np_vertices/np_vertices[:,3][:,None]
    mesh.vertices[:,0] = np_vertices[:,0]
    mesh.vertices[:,1] = np_vertices[:,1]
    mesh.vertices[:,2] = np_vertices[:,2]
    
    return mesh
