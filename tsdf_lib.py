import open3d as o3d
import numpy as np
import numpy.linalg as la
import cupy as cp
from skimage import measure

from .cuda_kernels import source_module


class TSDFVolume:

    def __init__(self, voxel_size, vol_bnds=None, vol_origin=None, vol_dim=None, trunc_margin=0.015):
        """
        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
    
        print('Init TSDF Volume')
        print(f'vol_dim: {vol_dim}')
        print(f'vol_origin: {vol_origin}')
        print(f'vol_bnds: {vol_bnds}')
        print(f'voxel_size: {voxel_size}')

        if vol_dim is not None and vol_origin is not None:
            self._vol_dim = vol_dim
            self._vol_origin = vol_origin
        elif vol_bnds is not None:
            self._vol_bnds = np.ascontiguousarray(vol_bnds, dtype=np.float32)
            self._vol_dim = np.round((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / voxel_size).astype(np.int32)
            self._vol_origin = self._vol_bnds[:, 0].copy()
        else:
            raise ValueError("must either provide 'vol_dim' or (both 'vol_bnds' and 'vol_origin')")

        self._voxel_size = voxel_size
        self._trunc_margin = trunc_margin
        self._color_const = np.float32(256 * 256)

        print(f"TSDF volume dim: {self._vol_dim}, # points: {np.prod(self._vol_dim)}")

        # Copy voxel volumes to GPU
        self.block_x, self.block_y, self.block_z = 8, 8, 16  # block_x * block_y * block_z must equal to 1024

        x_dim, y_dim, z_dim = int(self._vol_dim[0]), int(self._vol_dim[1]), int(self._vol_dim[2])
        self.grid_x = int(np.ceil(x_dim / self.block_x))
        self.grid_y = int(np.ceil(y_dim / self.block_y))
        self.grid_z = int(np.ceil(z_dim / self.block_z))

        # initialize tsdf values to be -1
        xyz = x_dim * y_dim * z_dim
        self._tsdf_vol_gpu = cp.zeros(shape=xyz, dtype=cp.float32) - 1
        self._weight_vol_gpu = cp.zeros(shape=xyz, dtype=cp.float32)
        self._color_vol_gpu = cp.zeros(shape=xyz, dtype=cp.float32)

        # integrate function using PyCuda
        self._cuda_integrate = source_module.get_function("integrate")
        self._cuda_ray_casting = source_module.get_function("rayCasting")

        # load and save
        self._depth_scale = 1000

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, weight=1.0):
        """ Integrate an RGB-D frame into the TSDF volume.

        Args:
            color_im (np.ndarray): input RGB image of shape (H, W, 3)
            depth_im (np.ndarray): input depth image of shape (H, W)
            cam_intr (np.ndarray): Camera intrinsics matrix of shape (3, 3)
            cam_pose (np.ndarray): Camera pose of shape (4, 4)
            weight (float, optional): weight to be assigned for the current observation. Defaults to 1.0.
        """
        im_h, im_w = depth_im.shape

        # color image is always from host
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])

        self._cuda_integrate(
            (self.grid_x, self.grid_y, self.grid_z),
            (self.block_x, self.block_y, self.block_z),
            (
                self._tsdf_vol_gpu,
                self._weight_vol_gpu,
                self._color_vol_gpu,
                cp.asarray(self._vol_dim.astype(np.int32)),
                cp.asarray(self._vol_origin.astype(np.float32)),
                cp.float32(self._voxel_size),
                cp.asarray(cam_intr.reshape(-1).astype(np.float32)),
                cp.asarray(cam_pose.reshape(-1).astype(np.float32)),
                cp.int32(im_h),
                cp.int32(im_w),
                cp.asarray(color_im.astype(np.float32)),
                cp.asarray(depth_im.astype(np.float32)),
                cp.float32(self._trunc_margin),
                cp.float32(weight),
            )
        )

    def ray_casting(self, im_w, im_h, cam_intr, cam_pose, start_row=0, start_col=0, to_host=True):
        """
        Render an image patch
        """
        depth_im_gpu = cp.zeros((im_h, im_w), dtype=np.float32)
        color_im_gpu = cp.zeros((3, im_h, im_w), dtype=np.uint8)

        self._cuda_ray_casting(
            (int(np.ceil(im_w / 32)), int(np.ceil(im_h / 32)), 1),
            (32, 32, 1),
            (
                self._tsdf_vol_gpu,
                self._color_vol_gpu,
                self._weight_vol_gpu,
                cp.asarray(self._vol_dim.astype(np.int32)),
                cp.asarray(self._vol_origin.astype(np.float32)),
                cp.float32(self._voxel_size),
                cp.asarray(cam_intr.reshape(-1).astype(np.float32)),
                cp.asarray(cam_pose.reshape(-1).astype(np.float32)),
                cp.asarray(la.inv(cam_pose).reshape(-1).astype(np.float32)),
                cp.int32(start_row),
                cp.int32(start_col),
                cp.int32(im_h),
                cp.int32(im_w),
                color_im_gpu,
                depth_im_gpu,
            )
        )
        if not to_host:
            return depth_im_gpu, color_im_gpu

        depth_im = cp.asnumpy(depth_im_gpu)
        color_im = cp.asnumpy(color_im_gpu).transpose(1, 2, 0)
        return depth_im, color_im

    def get_volume(self):
        x_dim, y_dim, z_dim = self._vol_dim
        tsdf_vol_cpu = cp.asnumpy(self._tsdf_vol_gpu).reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        color_vol_cpu = cp.asnumpy(self._color_vol_gpu).reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        weight_vol_cpu = cp.asnumpy(self._weight_vol_gpu).reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        return tsdf_vol_cpu, color_vol_cpu, weight_vol_cpu

    def get_surface_cloud_marching_cubes(self):
        tsdf_vol, color_vol, weight_vol = self.get_volume()

        # Marching cubes
        verts, faces, normals, values = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)

        # remove false surface
        verts_weight = weight_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        verts_val = tsdf_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        valid_idx = (verts_weight > 0) & (np.abs(verts_val) < 0.2)
        verts_ind = verts_ind[valid_idx]
        verts = verts[valid_idx]
        normals = normals[valid_idx]

        # make normals point outwards (negative -> positive) direction
        back_verts = verts - normals
        forward_verts = verts + normals
        back_verts = np.clip(back_verts, a_min=np.zeros(3), a_max=np.array(tsdf_vol.shape) - 1)
        forward_verts = np.clip(forward_verts, a_min=np.zeros(3), a_max=np.array(tsdf_vol.shape) - 1)
        back_ind = np.round(back_verts).astype(int)
        forward_ind = np.round(forward_verts).astype(int)

        back_val = tsdf_vol[back_ind[:, 0], back_ind[:, 1], back_ind[:, 2]]
        forward_val = tsdf_vol[forward_ind[:, 0], forward_ind[:, 1], forward_ind[:, 2]]
        normals[(forward_val - back_val) < 0] *= -1

        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T

        surface_cloud = o3d.geometry.PointCloud()
        surface_cloud.points = o3d.utility.Vector3dVector(verts)
        surface_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
        surface_cloud.normals = o3d.utility.Vector3dVector(normals)
        return surface_cloud
    
    def get_conservative_volume(self, voxel_size=0.01):
        tsdf_vol, _, _ = self.get_volume()
        verts = np.vstack(np.where(tsdf_vol < -0.2)).T
        verts = verts * self._voxel_size + self._vol_origin

        conservative_volume = o3d.geometry.PointCloud()
        conservative_volume.points = o3d.utility.Vector3dVector(verts)
        conservative_volume = conservative_volume.voxel_down_sample(voxel_size=voxel_size)
        return conservative_volume

    def save(self, output_path):
        np.savez_compressed(output_path,
                            vol_dim=self._vol_dim,
                            vol_origin=self._vol_origin,
                            vol_bounds=self._vol_bnds,
                            voxel_size=self._voxel_size,
                            trunc_margin=self._trunc_margin,
                            tsdf_vol=(self._tsdf_vol_gpu.get() * self._depth_scale).astype(np.int16),
                            weight_vol=(self._weight_vol_gpu.get()).astype(np.uint16),
                            color_vol=(self._color_vol_gpu.get()).astype(np.uint32)
                            )
        print(f"tsdf volume has been saved to: {output_path}")

    @classmethod
    def load(cls, input_path):
        loaded = np.load(input_path)
        print('loaded voxel_size:', loaded['voxel_size'])
        print('loaded vol_bnds', loaded.get('vol_bounds'))
        print('loaded vol_origin', loaded.get('vol_origin'))
        print('loaded vol_dim:', loaded.get('vol_dim'))
        print('loaded trunc_margin:', loaded['trunc_margin'])
        obj = cls(voxel_size=loaded['voxel_size'], vol_bnds=loaded.get('vol_bounds'),
                  vol_dim=loaded.get('vol_dim'), vol_origin=loaded.get('vol_origin'),
                  trunc_margin=loaded['trunc_margin'])
        obj._tsdf_vol_gpu = cp.asarray((loaded['tsdf_vol'].astype(np.float32)) / obj._depth_scale)
        obj._weight_vol_gpu = cp.asarray(loaded['weight_vol'].astype(np.float32))
        obj._color_vol_gpu = cp.asarray(loaded['color_vol'].astype(np.float32))
        print(f"tsdf volume has been loaded from: {input_path}")
        return obj
