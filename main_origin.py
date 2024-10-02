# cd ..
# python -m kf_pycuda.main

import os
import argparse
import json
from pprint import pprint
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm

from .kinect_fusion import KinectFusion
from .kf_config import get_config

def get_surface_cloud_marching_cubes_from_volume(tsdf_vol, color_vol, weight_vol, voxel_size, vol_origin, color_const=256*256):
    import numpy as np
    import open3d as o3d
    from skimage import measure

    # TSDF 볼륨 값 범위 확인
    print(f"TSDF Volume Min: {tsdf_vol.min()}, Max: {tsdf_vol.max()}")

    # marching_cubes의 surface level이 TSDF 범위 내에 있는지 확인 (일반적으로 0 사용)
    level = 0
    if level < tsdf_vol.min() or level > tsdf_vol.max():
        #raise ValueError(f"Surface level {level} is out of TSDF volume range: {tsdf_vol.min()} to {tsdf_vol.max()}")
        surface_cloud = o3d.geometry.PointCloud()
        return surface_cloud

    # Marching cubes 알고리즘 적용
    verts, faces, normals, values = measure.marching_cubes(tsdf_vol, level=level)
    verts_ind = np.round(verts).astype(int)

    # 유효한 인덱스 필터링
    valid_mask = (
        (verts_ind[:, 0] >= 0) & (verts_ind[:, 0] < tsdf_vol.shape[0]) &
        (verts_ind[:, 1] >= 0) & (verts_ind[:, 1] < tsdf_vol.shape[1]) &
        (verts_ind[:, 2] >= 0) & (verts_ind[:, 2] < tsdf_vol.shape[2])
    )
    verts_ind = verts_ind[valid_mask]
    verts = verts[valid_mask]
    normals = normals[valid_mask]

    # 잘못된 표면 제거
    verts_weight = weight_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    verts_val = tsdf_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    valid_idx = (verts_weight > 0) & (np.abs(verts_val) < 0.2)
    verts_ind = verts_ind[valid_idx]
    verts = verts[valid_idx]
    normals = normals[valid_idx]

    # 노말 방향 조정
    back_verts = verts - normals
    forward_verts = verts + normals
    back_verts = np.clip(back_verts, a_min=0, a_max=np.array(tsdf_vol.shape) - 1)
    forward_verts = np.clip(forward_verts, a_min=0, a_max=np.array(tsdf_vol.shape) - 1)
    back_ind = np.round(back_verts).astype(int)
    forward_ind = np.round(forward_verts).astype(int)

    back_val = tsdf_vol[back_ind[:, 0], back_ind[:, 1], back_ind[:, 2]]
    forward_val = tsdf_vol[forward_ind[:, 0], forward_ind[:, 1], forward_ind[:, 2]]
    normals[(forward_val - back_val) < 0] *= -1

    verts = verts * voxel_size + vol_origin

    # 색상 정보 추출
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / color_const)
    colors_g = np.floor((rgb_vals - colors_b * color_const) / 256)
    colors_r = rgb_vals - colors_b * color_const - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T

    surface_cloud = o3d.geometry.PointCloud()
    surface_cloud.points = o3d.utility.Vector3dVector(verts)
    surface_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
    surface_cloud.normals = o3d.utility.Vector3dVector(normals)
    return surface_cloud


def process_marching_cubes_in_chunks(tsdf_vol, color_vol, weight_vol, voxel_size, vol_origin, color_const, chunk_size):
    import numpy as np
    import open3d as o3d

    all_points = []
    all_colors = []
    all_normals = []

    z_dim, y_dim, x_dim = tsdf_vol.shape

    # 작은 청크로 볼륨을 나누기
    for z in range(0, z_dim, chunk_size):
        for y in range(0, y_dim, chunk_size):
            for x in range(0, x_dim, chunk_size):
                # 청크 선택
                tsdf_chunk = tsdf_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                color_chunk = color_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                weight_chunk = weight_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]

                # 청크가 모두 -1.0인지 확인
                if np.all(np.isclose(tsdf_chunk, -1.0)):
                    print(f"Skipping chunk at z:{z}, y:{y}, x:{x} (all -1.0)")
                    continue  # 해당 청크 건너뛰기

                # marching_cubes 수행
                surface_chunk = get_surface_cloud_marching_cubes_from_volume(
                    tsdf_chunk, color_chunk, weight_chunk, voxel_size, vol_origin, color_const=color_const)

                # 각 청크의 좌표에 오프셋 적용
                offset = np.array([x, y, z]) * voxel_size  # 청크의 시작 좌표
                verts_with_offset = np.asarray(surface_chunk.points) + offset  # 오프셋 적용

                # 각 청크의 points, colors, normals 데이터를 수집
                all_points.append(verts_with_offset)
                all_colors.append(np.asarray(surface_chunk.colors))
                all_normals.append(np.asarray(surface_chunk.normals))

    # 최종적으로 수집된 데이터를 하나의 PointCloud로 병합
    full_surface = o3d.geometry.PointCloud()
    if all_points:
        full_surface.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        full_surface.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        full_surface.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))

    return full_surface


def get_surface_cloud_marching_cubes_from_volume2(tsdf_vol, color_vol, weight_vol, voxel_size, vol_origin, color_const=256*256):
    from skimage import measure

    # Marching cubes 알고리즘 적용
    verts, faces, normals, values = measure.marching_cubes(tsdf_vol, level=0.0)
    verts_ind = np.round(verts).astype(int)

    # 유효한 인덱스 필터링
    valid_mask = (
        (verts_ind[:, 0] >= 0) & (verts_ind[:, 0] < tsdf_vol.shape[0]) &
        (verts_ind[:, 1] >= 0) & (verts_ind[:, 1] < tsdf_vol.shape[1]) &
        (verts_ind[:, 2] >= 0) & (verts_ind[:, 2] < tsdf_vol.shape[2])
    )
    verts_ind = verts_ind[valid_mask]
    verts = verts[valid_mask]
    normals = normals[valid_mask]

    # 잘못된 표면 제거
    verts_weight = weight_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    verts_val = tsdf_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    valid_idx = (verts_weight > 0) & (np.abs(verts_val) < 0.2)
    verts_ind = verts_ind[valid_idx]
    verts = verts[valid_idx]
    normals = normals[valid_idx]

    # 노말 방향 조정
    back_verts = verts - normals
    forward_verts = verts + normals
    back_verts = np.clip(back_verts, a_min=0, a_max=np.array(tsdf_vol.shape) - 1)
    forward_verts = np.clip(forward_verts, a_min=0, a_max=np.array(tsdf_vol.shape) - 1)
    back_ind = np.round(back_verts).astype(int)
    forward_ind = np.round(forward_verts).astype(int)

    back_val = tsdf_vol[back_ind[:, 0], back_ind[:, 1], back_ind[:, 2]]
    forward_val = tsdf_vol[forward_ind[:, 0], forward_ind[:, 1], forward_ind[:, 2]]
    normals[(forward_val - back_val) < 0] *= -1

    verts = verts * voxel_size + vol_origin

    # 색상 정보 추출
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / color_const)
    colors_g = np.floor((rgb_vals - colors_b * color_const) / 256)
    colors_r = rgb_vals - colors_b * color_const - colors_g * 256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T

    surface_cloud = o3d.geometry.PointCloud()
    surface_cloud.points = o3d.utility.Vector3dVector(verts)
    surface_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
    surface_cloud.normals = o3d.utility.Vector3dVector(normals)
    return surface_cloud

def process_marching_cubes_in_chunks(tsdf_vol, color_vol, weight_vol, voxel_size, vol_origin, color_const, chunk_size):
    import numpy as np
    import open3d as o3d

    all_points = []
    all_colors = []
    all_normals = []

    z_dim, y_dim, x_dim = tsdf_vol.shape

    # 작은 청크로 볼륨을 나누기
    for z in range(0, z_dim, chunk_size):
        for y in range(0, y_dim, chunk_size):
            for x in range(0, x_dim, chunk_size):
                # 청크 선택
                tsdf_chunk = tsdf_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                color_chunk = color_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]
                weight_chunk = weight_vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size]

                # marching_cubes 수행
                surface_chunk = get_surface_cloud_marching_cubes_from_volume(
                    tsdf_chunk, color_chunk, weight_chunk, voxel_size, vol_origin, color_const=color_const)

                # 각 청크의 points, colors, normals 데이터를 수집
                all_points.append(np.asarray(surface_chunk.points))
                all_colors.append(np.asarray(surface_chunk.colors))
                all_normals.append(np.asarray(surface_chunk.normals))

    # 최종적으로 수집된 데이터를 하나의 PointCloud로 병합
    full_surface = o3d.geometry.PointCloud()
    if all_points:
        full_surface.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        full_surface.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        full_surface.normals = o3d.utility.Vector3dVector(np.vstack(all_normals))

    return full_surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default='~/dataset')
    parser.add_argument("-v", "--video", type=str, default="0001")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--depth_trunc", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output_folder", type=str, default=".")
    parser.add_argument("--save_tsdf", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser()
    video_folder = dataset / args.video
    color_folder = video_folder / 'color'
    if not os.path.isdir(color_folder):
        print(f"{color_folder} doesn't exist.")
        exit(1)

    color_files = sorted(os.listdir(color_folder))
    if args.end_frame == -1:
        args.end_frame = len(color_files)
    color_files = color_files[args.start_frame:args.end_frame]

    data_cfg_path = video_folder / 'config.json'
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['cam_intr'] = np.asarray(cfg['cam_intr'])

    kf_cfg = get_config()
    cfg.update(kf_cfg)
    pprint(cfg)

    kf = KinectFusion(cfg=cfg)

    # initialize TSDF with the first frame
    color_im_path = str(video_folder / 'color' / f'{color_files[0]}')
    prefix = color_files[0].split('-')[0]
    depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
    color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
    depth_im[depth_im > 2] = 0
    kf.initialize_tsdf_volume(color_im, depth_im, visualize=True)
    
    counter = 0
    n = 10

    # Update TSDF volume
    for start_idx, end_idx in [(1,11), (11, 21), (21, 31), (31, 41), (41, 51)]:
        for color_file in tqdm(color_files[start_idx:end_idx]):
            color_im_path = str(video_folder / 'color' / f'{color_file}')
            prefix = color_file.split('-')[0]
            depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
            color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
            depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
            depth_im[depth_im > args.depth_trunc] = 0
            if not kf.update(color_im, depth_im):
                print(f"{color_file = } is skipped.")
                continue
            
            counter+=1

            if not counter % n == 0:
                continue
        
            volume = kf.tsdf_volume._tsdf_vol_gpu
            print(f'volume info - min: {volume.min()}, max: {volume.max()}, mean: {volume.mean()}')
            #print(f'tsdf volume({kf.tsdf_volume._tsdf_vol_gpu.shape}): {kf.tsdf_volume._tsdf_vol_gpu} ')
            #print(f'color volume({kf.tsdf_volume._color_vol_gpu.shape}): {kf.tsdf_volume._color_vol_gpu} ')
            #print(f'weight volume({kf.tsdf_volume._weight_vol_gpu.shape}): {kf.tsdf_volume._weight_vol_gpu} ')

        cam_frames = []
        for cam_pose in kf.cam_poses:
            cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
            cam_frame.transform(cam_pose)
            cam_frames.append(cam_frame)

        factor = 4

        tsdf_vol, color_vol, weight_vol = kf.tsdf_volume.get_volume()

        #factor = 1  # 다운샘플링 비율, 2로 설정하면 해상도가 절반으로 줄어듭니다.
        tsdf_vol = tsdf_vol[::factor, ::factor, ::factor]/factor
        color_vol = color_vol[::factor, ::factor, ::factor]
        weight_vol = weight_vol[::factor, ::factor, ::factor]

        # TSDF에서 surface cloud 업데이트
        surface = get_surface_cloud_marching_cubes_from_volume2(
            tsdf_vol, color_vol, weight_vol,
            cfg['tsdf_voxel_size']*factor, kf.tsdf_volume._vol_origin, color_const=256*256)

        #surface = kf.tsdf_volume.get_surface_cloud_marching_cubes()
        o3d.visualization.draw_geometries([kf.vol_box, surface] + cam_frames)

    if args.output_folder:
        output_folder = video_folder / args.output_folder
        kf.save(output_folder, save_tsdf=args.save_tsdf)

        # save camera poses in txt files.
        assert len(kf.cam_poses) == len(color_files)
        output_pose_dir = output_folder / 'poses'
        output_pose_dir.mkdir(parents=True, exist_ok=True)
        for color_file, cam_pose in zip(color_files, kf.cam_poses):
            prefix = color_file.split('-')[0]
            pose_file = output_pose_dir / f'{prefix}-pose.txt'
            np.savetxt(pose_file, cam_pose, fmt='%.6f')

        # save a copy of the down-sampled point cloud for convenience
        voxel_size = 0.005
        output_path = output_folder / f'scan-{voxel_size:.3f}.pcd'
        surface_down = surface.voxel_down_sample(voxel_size=voxel_size)
        o3d.io.write_point_cloud(str(output_path), surface_down)
        print(f"Reconstruction results have been saved to {output_path}.")


if __name__ == '__main__':
    main()
