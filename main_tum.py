# cd ..
# python -m kf_pycuda.main

import os
import argparse
import json
import time

from pprint import pprint
from pathlib import Path

import numpy as np
import cupy as cp
import cv2
import open3d as o3d
from tqdm import tqdm

from .kinect_fusion import KinectFusion
from .kf_config import get_config

from multiprocessing import Process, shared_memory, Event
from skimage import measure


def get_surface_cloud_marching_cubes_from_volume(tsdf_vol, color_vol, weight_vol, voxel_size, vol_origin, color_const=256*256):
    # Marching cubes 알고리즘 적용
    verts, faces, normals, values = measure.marching_cubes(tsdf_vol, level=0)
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
    
def create_coordinate_frame(size=1.0):
    # 좌표축의 점 정의
    points = np.array([
        [0, 0, 0],
        [size, 0, 0],   # X축 끝점
        [0, size, 0],   # Y축 끝점
        [0, 0, size],   # Z축 끝점
    ])
    # 좌표축의 선 정의
    lines = [
        [0, 1],  # X축
        [0, 2],  # Y축
        [0, 3],  # Z축
    ]
    # 좌표축의 색상 정의
    colors = [
        [1, 0, 0],  # X축: 빨간색
        [0, 1, 0],  # Y축: 초록색
        [0, 0, 1],  # Z축: 파란색
    ]
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(points)
    frame.lines = o3d.utility.Vector2iVector(lines)
    frame.colors = o3d.utility.Vector3dVector(colors)
    return frame

def visualization_process(tsdf_shm_name, color_shm_name, weight_shm_name, cam_pose_shm_name, cam_poses_count_shm_name, event, cfg):
    # 공유 메모리 연결
    tsdf_shm = shared_memory.SharedMemory(name=tsdf_shm_name)
    color_shm = shared_memory.SharedMemory(name=color_shm_name)
    weight_shm = shared_memory.SharedMemory(name=weight_shm_name)
    cam_pose_shm = shared_memory.SharedMemory(name=cam_pose_shm_name)
    cam_poses_count_shm = shared_memory.SharedMemory(name=cam_poses_count_shm_name)

    # Shared Memory에 연결된 numpy 배열 (1차원 배열로 연결)
    x_dim, y_dim, z_dim = cfg['vol_dim']
    #xyz = x_dim * y_dim * z_dim
    tsdf_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=tsdf_shm.buf)
    color_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=color_shm.buf)
    weight_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=weight_shm.buf)

     # cam_poses 및 개수 배열에 연결
    max_cam_poses = 2000  # main.py에서 설정한 최대 카메라 프레임 수
    cam_poses_shared = np.ndarray((max_cam_poses, 4, 4), dtype=np.float32, buffer=cam_pose_shm.buf)
    cam_poses_count = np.ndarray(1, dtype=np.int32, buffer=cam_poses_count_shm.buf)

    # vol_box 생성
    vol_box = o3d.geometry.OrientedBoundingBox()
    vol_bnds = cfg['vol_bnds']
    vol_box.center = vol_bnds.mean(axis=1)
    vol_box.extent = vol_bnds[:, 1] - vol_bnds[:, 0]
    vol_box.color = [1, 0, 0]

    # Open3D 시각화 창 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    surface = o3d.geometry.PointCloud()
    vis.add_geometry(vol_box)
    vis.add_geometry(surface)

    # 카메라 프레임을 나타낼 LineSet 생성
    cam_frames = o3d.geometry.LineSet()
    vis.add_geometry(cam_frames)

    # 초기 뷰 설정 (박스의 옆면에서 중심을 바라보는 방향)
    ctr = vis.get_view_control()
    ctr.set_up([0, 0, 1])        # Z축을 위쪽으로 설정
    ctr.set_front([-1, 0, 0])    # X축 음의 방향에서 바라봄
    ctr.set_lookat([0, 0, 0])    # 박스의 중심을 바라봄
    ctr.set_zoom(0.5)

    while True:
        # 사용자 이벤트 처리 (Open3D 창 관련)
        vis.poll_events()
        vis.update_renderer()

        # 잠시 대기
        time.sleep(0.01)

        #print('viewer!')

        # 이벤트 대기 및 데이터 업데이트 처리
        if not event.is_set():
            continue

        event.clear()

        #event.wait()  # 이벤트가 설정될 때까지 대기
        #event.clear()

        #print('get event!')

        # reshape 먼저 수행한 후 transpose 적용
        #tsdf_vol = tsdf_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        #color_vol = color_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        #weight_vol = weight_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        
        #factor = 4  # 다운샘플링 비율, 2로 설정하면 해상도가 절반으로 줄어듭니다.
        #tsdf_vol = tsdf_vol[::factor, ::factor, ::factor]/factor
        #color_vol = color_vol[::factor, ::factor, ::factor]
        #weight_vol = weight_vol[::factor, ::factor, ::factor]

        #print(f'volume shape: {tsdf_vol.shape}')

        # TSDF에서 surface cloud 업데이트
        surface_new = get_surface_cloud_marching_cubes_from_volume(
            tsdf_vol_shared, color_vol_shared, weight_vol_shared,
            cfg['tsdf_voxel_size'], cfg['vol_origin'], color_const=256*256)

        # Surface 업데이트
        surface.points = surface_new.points
        surface.colors = surface_new.colors
        surface.normals = surface_new.normals
        vis.update_geometry(surface)

        # cam_poses 개수를 확인하고 업데이트
        current_cam_poses = cam_poses_count[0]
        #print(f'shared cam poses: {current_cam_poses}')
        if current_cam_poses > 0:
            cam_frame_size = 0.1
            points = []
            lines = []
            colors = []
            for idx in range(current_cam_poses):
                cam_pose = cam_poses_shared[idx]
                #print(f'draw cam pose({cam_pose})')
                frame = create_coordinate_frame(cam_frame_size)
                frame.transform(cam_pose)
                num_points = len(points)
                points.extend(np.asarray(frame.points))
                lines.extend(np.asarray(frame.lines) + num_points)
                colors.extend(np.asarray(frame.colors))
            cam_frames.points = o3d.utility.Vector3dVector(points)
            cam_frames.lines = o3d.utility.Vector2iVector(lines)
            cam_frames.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(cam_frames)

    vis.destroy_window()

def show_image(color_im, depth_im):
    color_im_bgr = color_im[:, :, [2,1,0]]
    #print(f'shape: {color_im_bgr.shape}')

    depth_min = depth_im.min()
    depth_max = depth_im.max()
    #print(f'depth min: {depth_min}m, depth_max: {depth_max}m')

    depth_im_normalized = (depth_im - depth_min) / (depth_max - depth_min)  # 0 ~ 1로 정규화
    depth_im_normalized = (depth_im_normalized * 255).astype(np.uint8)

    depth_im_jet = cv2.applyColorMap(depth_im_normalized, cv2.COLORMAP_JET)
    combined_im = np.hstack((color_im_bgr, depth_im_jet))
    cv2.imshow('color & depth image', combined_im)

    cv2.waitKey(1)


def extract_timestamp(file_path):
    """
    파일 경로에서 타임스탬프를 추출하여 실수형으로 반환합니다.
    예: '/path/to/rgb/1311875744.828744.png' -> 1311875744.828744
    """
    filename = os.path.basename(file_path)
    timestamp_str = os.path.splitext(filename)[0]
    try:
        return float(timestamp_str)
    except ValueError:
        raise ValueError(f"파일명에서 타임스탬프를 추출할 수 없습니다: {file_path}")


def match_rgb_depth(rgb_list, depth_list, max_time_diff=0.05):
    """
    RGB 리스트와 Depth 리스트를 타임스탬프 기준으로 매칭합니다.
    - rgb_list: RGB 이미지의 전체 경로 리스트 (정렬된 상태)
    - depth_list: Depth 이미지의 전체 경로 리스트 (정렬된 상태)
    - max_time_diff: 매칭 허용 최대 시간 차이 (초)

    반환값:
    - matched_pairs: [(rgb_path, depth_path), ...] 형태의 리스트
    """
    # 작은 리스트와 큰 리스트 결정
    if len(rgb_list) <= len(depth_list):
        small, large = rgb_list, depth_list
    else:
        small, large = depth_list, rgb_list

    # 타임스탬프 추출
    small_times = [extract_timestamp(f) for f in small]
    large_times = [extract_timestamp(f) for f in large]

    matched_pairs = []
    used = set()
    j = 0  # 큰 리스트의 인덱스

    for st, sf in zip(small_times, small):
        # 큰 리스트에서 st에 가장 가까운 위치 찾기
        while j < len(large_times) and large_times[j] < st:
            j += 1

        candidates = []
        if j < len(large_times):
            candidates.append(j)
        if j > 0:
            candidates.append(j - 1)

        closest = -1
        min_diff = float('inf')

        for c in candidates:
            if c not in used:
                diff = abs(st - large_times[c])
                if diff < min_diff:
                    min_diff = diff
                    closest = c

        if closest != -1 and min_diff <= max_time_diff:
            matched_pairs.append((sf, large[closest]))
            used.add(closest)

    return matched_pairs


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
    video_folder = dataset
    #color_folder = video_folder / 'color'
    #if not os.path.isdir(color_folder):
    #    print(f"{color_folder} doesn't exist.")
    #    exit(1)

    #color_files = sorted(os.listdir(color_folder))
    #if args.end_frame == -1:
    #    args.end_frame = len(color_files)
    #color_files = color_files[args.start_frame:args.end_frame]

    stride = 1

    rgbdir = os.path.join(video_folder, 'rgb')
    depthdir = os.path.join(video_folder, 'depth')

    rgb_list = sorted(os.listdir(rgbdir))[::stride]
    depth_list = sorted(os.listdir(depthdir))[::stride]

    matched_pairs = match_rgb_depth(rgb_list, depth_list, max_time_diff=1.00)

    data_cfg_path = video_folder / 'config.json'
    with open(data_cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg['cam_intr'] = np.asarray(cfg['cam_intr'])

    kf_cfg = get_config()
    cfg.update(kf_cfg)
    
    resolution_down_scale = 2
    cfg['im_h'] = cfg['im_h'] // resolution_down_scale
    cfg['im_w'] = cfg['im_w'] // resolution_down_scale
    cfg['cam_intr'] = cfg['cam_intr'] / resolution_down_scale
    cfg['cam_intr'][2,2] = 1.0

    cfg['bound_dx'] = [-1.0, 1.0]
    cfg['bound_dy'] = [-1.0, 1.0]
    cfg['bound_z'] = [-0.5, 0.5]

    pprint(cfg)
    
    kf = KinectFusion(cfg=cfg)

    #print(f'kf = {kf}')

    #color_im_path = str(video_folder / 'color' / f'{color_files[0]}')
    #prefix = color_files[0].split('-')[0]
    #depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
    #color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
    #depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
    #depth_im[depth_im > args.depth_trunc] = 0
    
    dfile, imfile = matched_pairs[0]
    color_im = cv2.imread(os.path.join(rgbdir, imfile))
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(os.path.join(depthdir, dfile), cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale'] / 4
    depth_im[depth_im > args.depth_trunc] = 0

    color_im = cv2.resize(color_im, (cfg['im_w'], cfg['im_h']))
    depth_im = cv2.resize(depth_im, (cfg['im_w'], cfg['im_h']))
        
    print(f'color image shape: {color_im.shape}')
    print(f'depth image shape: {depth_im.shape}')

    kf.initialize_tsdf_volume(color_im, depth_im, visualize=False)
    print(f'kf init done')
  
    # TSDF 초기화 후
    cfg['vol_origin'] = kf.tsdf_volume._vol_origin
    cfg['vol_bnds'] = kf.tsdf_volume._vol_bnds
    # TSDF 볼륨 초기화 이후 vol_dim을 cfg에 추가
    x_dim, y_dim, z_dim = kf.tsdf_volume._vol_dim  # TSDF 볼륨 크기를 가져옴
   
    factor = 2
    cfg['tsdf_voxel_size'] *= factor

    x_dim = (x_dim + factor - 1) // factor
    y_dim = (y_dim + factor - 1) // factor
    z_dim = (z_dim + factor - 1) // factor

    cfg['vol_dim'] = (x_dim, y_dim, z_dim)
    xyz = x_dim * y_dim * z_dim
    
    #ueue = Queue()
    #is_process = Process(target=visualization_process, args=(queue, cfg))
    #is_process.start()
    # TSDF 볼륨, 컬러 볼륨, 가중치 볼륨을 위한 공유 메모리 생성
    tsdf_shm = shared_memory.SharedMemory(create=True, size=xyz*4)
    color_shm = shared_memory.SharedMemory(create=True, size=xyz*4)
    weight_shm = shared_memory.SharedMemory(create=True, size=xyz*4)

    #print(f'tsdf_shm size: {kf.tsdf_volume._tsdf_vol_gpu.nbytes}')

    # Shared Memory에 연결된 numpy 배열
    tsdf_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=tsdf_shm.buf)
    color_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=color_shm.buf)
    weight_vol_shared = np.ndarray((x_dim, y_dim, z_dim), dtype=np.float32, buffer=weight_shm.buf)
    #print(f'tsdf_vol_shared shape: {kf.tsdf_volume._tsdf_vol_gpu.shape}')

    # cam_poses를 위한 고정된 크기의 shared memory 생성
    max_cam_poses = 2000  # 예상되는 최대 카메라 프레임 수
    cam_pose_shm = shared_memory.SharedMemory(create=True, size=max_cam_poses * 16 * 4)  # 4x4 행렬 16개 * float32 크기

    # cam_poses 공유 메모리에 연결된 numpy 배열
    cam_poses_shared = np.ndarray((max_cam_poses, 4, 4), dtype=np.float32, buffer=cam_pose_shm.buf)

    # 현재 사용 중인 cam_poses 개수를 추적하기 위한 shared memory 생성
    cam_poses_count_shm = shared_memory.SharedMemory(create=True, size=4)  # int32 크기
    cam_poses_count = np.ndarray(1, dtype=np.int32, buffer=cam_poses_count_shm.buf)

    # 동기화 이벤트 생성
    event = Event()

    # 시각화 프로세스를 shared memory와 event로 실행
    #vis_process = Process(target=visualization_process, args=(tsdf_shm.name, color_shm.name, weight_shm.name, cam_pose_shm.name, event, cfg))
    vis_process = Process(target=visualization_process, args=(tsdf_shm.name, color_shm.name, weight_shm.name, cam_pose_shm.name, cam_poses_count_shm.name, event, cfg))
    vis_process.start()

    print('vis process start')

    #import time
    #time.sleep(20)
    
    # 업데이트 카운터 초기화
    update_counter = 0
    n = 10  # 원하는 업데이트 간격으로 설정하세요

    # Update TSDF volume
    #for color_file in tqdm(color_files[1:]):
    for (dfile, imfile) in tqdm(matched_pairs[1:]):

        #color_im_path = str(video_folder / 'color' / f'{color_file}')
        #print(f'read color im path: {color_im_path}')
        #prefix = color_file.split('-')[0]
        #depth_im_path = str(video_folder / 'depth' / f'{prefix}-depth.png')
        #print(f'read depth im path: {depth_im_path}')
        #color_im = cv2.cvtColor(cv2.imread(color_im_path), cv2.COLOR_BGR2RGB)
        #depth_im = cv2.imread(depth_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale']
        color_im = cv2.imread(os.path.join(rgbdir, imfile))
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(os.path.join(depthdir, dfile), cv2.IMREAD_UNCHANGED).astype(np.float32) / cfg['depth_scale'] / 4
        depth_im[depth_im > args.depth_trunc] = 0
        
        color_im = cv2.resize(color_im, (cfg['im_w'], cfg['im_h']))
        depth_im = cv2.resize(depth_im, (cfg['im_w'], cfg['im_h']))

        show_image(color_im, depth_im)

        if not kf.update(color_im, depth_im):
            continue

        update_counter += 1

        if not update_counter % n == 0:
            continue
        
        # cupy 배열을 numpy로 변환한 후 1차원 배열로 공유 메모리에 복사
        #print(f'volume dtype: {cp.asnumpy(kf.tsdf_volume._tsdf_vol_gpu).dtype}')
        
        #print(f'copy volume to shared memory(shm size: {tsdf_shm.size}, data size: {cp.asnumpy(kf.tsdf_volume._tsdf_vol_gpu).nbytes})')
        
        #volume = kf.tsdf_volume._tsdf_vol_gpu
        #print(f'volume info - min: {volume.min()}, max: {volume.max()}, mean: {volume.mean()}')
        
        #tsdf_vol, color_vol, weight_vol = kf.tsdf_volume.get_volume()
        x_dim_orig, y_dim_orig, z_dim_orig = kf.tsdf_volume._vol_dim
        tsdf_vol = cp.asnumpy(kf.tsdf_volume._tsdf_vol_gpu.reshape((z_dim_orig, y_dim_orig, x_dim_orig))[::factor, ::factor, ::factor]).transpose(2, 1, 0)/4
        color_vol = cp.asnumpy(kf.tsdf_volume._color_vol_gpu.reshape((z_dim_orig, y_dim_orig, x_dim_orig))[::factor, ::factor, ::factor]).transpose(2, 1, 0)
        weight_vol = cp.asnumpy(kf.tsdf_volume._weight_vol_gpu.reshape((z_dim_orig, y_dim_orig, x_dim_orig))[::factor, ::factor, ::factor]).transpose(2, 1, 0)


        #tsdf_vol = tsdf_vol[::factor, ::factor, ::factor]/4
        #color_vol = color_vol[::factor, ::factor, ::factor]
        #weight_vol = weight_vol[::factor, ::factor, ::factor]

        #tsdf_vol = tsdf_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        #color_vol = color_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
        #weight_vol = weight_vol_shared.reshape((z_dim, y_dim, x_dim)).transpose(2, 1, 0)
    
        np.copyto(tsdf_vol_shared, tsdf_vol)
        np.copyto(color_vol_shared, color_vol)
        np.copyto(weight_vol_shared, weight_vol)

        #np.copyto(tsdf_vol_shared, cp.asnumpy(kf.tsdf_volume._tsdf_vol_gpu[::factor, ::factor, ::factor]/factor))
        #np.copyto(color_vol_shared, cp.asnumpy(kf.tsdf_volume._color_vol_gpu[::factor, ::factor, ::factor]))
        #np.copyto(weight_vol_shared, cp.asnumpy(kf.tsdf_volume._weight_vol_gpu[::factor, ::factor, ::factor]))

        # cam_poses를 공유 메모리에 복사 (최대 max_cam_poses 크기 내에서)
        current_cam_poses = len(kf.cam_poses)
        #print(f'current cam poses: {current_cam_poses}')
        if current_cam_poses <= max_cam_poses:
            np.copyto(cam_poses_shared[:current_cam_poses], kf.cam_poses)

        # 현재 사용 중인 cam_poses 개수를 저장
        cam_poses_count[0] = current_cam_poses

        #print('set event!')
        # 이벤트 신호를 통해 시각화 프로세스에 데이터 업데이트 알림
        event.set()
        #if not kf.update(color_im, depth_im):
        #    print(f"{color_file = } is skipped.")
        #if not kf.update(color_im, depth_im):
        #    print(f"{color_file = } is skipped.")
        #else:
        #    update_counter += 1  # 카운터 증가
        #    if update_counter % n == 0:
        #        #n번 업데이트마다 시각화 프로세스에 데이터 전송
        #        print('get volume')
        #        tsdf_vol, color_vol, weight_vol = kf.tsdf_volume.get_volume()
        #        #tsdf_vol, color_vol, weight_vol = None, None, None
        #        #print(f'tsdf_vol shape: {tsdf_vol.shape}')
        #        #print(f'color_vol shape: {color_vol.shape}')
        #        #print(f'weight_vol shape: {weight_vol.shape}')

        #        print('put data')
        #        queue.put((tsdf_vol.copy(), color_vol.copy(), weight_vol.copy(), kf.cam_poses.copy()))
        #        print('done')

            # TSDF 볼륨 데이터 가져오기
            #print('get volume')
            #tsdf_vol, color_vol, weight_vol = kf.tsdf_volume.get_volume()
            #rint('get volume done')
        #    # 큐에 데이터 전송
        #    queue.put((tsdf_vol, color_vol, weight_vol, kf.cam_poses.copy()))

    #print(f'cam poses shape: {np.asarray(kf.cam_poses).shape}')
    #cam_frames = []
    #for cam_pose in kf.cam_poses:
    #    cam_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1)
    #    cam_frame.transform(cam_pose)
    #    cam_frames.append(cam_frame)
    #surface = kf.tsdf_volume.get_surface_cloud_marching_cubes()
    #print(f'points in surface shape: {np.asarray(surface.points).shape}')
    #print(f'vol_box shape : {np.asarray(kf.vol_box).shape}')
    #o3d.visualization.draw_geometries([kf.vol_box, surface] + cam_frames)

    # 종료 신호 전송
    #queue.put(None)
    event.set()
    vis_process.join()

    tsdf_shm.close()
    tsdf_shm.unlink()
    color_shm.close()
    color_shm.unlink()
    weight_shm.close()
    weight_shm.unlink()
    cam_pose_shm.close()
    cam_pose_shm.unlink()

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
