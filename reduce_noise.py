import argparse
import numpy as np
import time

import os, os.path as osp
from open3d_utils import *

inv = np.linalg.inv
dist = np.linalg.norm
norm = dist

connections = [
                [0,1],
                [0,15],
                [15,17],
                [0,16],
                [16,18],
                [1,2],
                [1,5],
                [2,3],
                [3,4],
                [5,6],
                [6,7],
                [1,8],
                [8,9],
                [9,10],
                [10,11],
                [11,24],
                [24,22],
                [24,23],
                [8,12],
                [12,13],
                [13,14],
                [14,21],
                [21,19],
                [21,20]
              ]


def visualize(poses_3d, FPS=24):
    default_colors = [
        [0.,0.,0.],  # Black
        [1.,0.,0.],  # Red
        [0.,1.,0.],  # Green
        [0.,0.,1.],  # Blue
    ]

    P3D = [poses_3d] if isinstance(poses_3d, np.ndarray) else poses_3d

    vis = open3d.visualization.Visualizer()

    vis.create_window()
    line_sets = []
    for ix, i in enumerate(P3D):
        line_set = open3d.geometry.LineSet()
        line_set.lines = open3d.Vector2iVector(lines)
        line_set.points = open3d.Vector3dVector(i[0])
        line_set.colors = open3d.Vector3dVector(np.array([default_colors[ix%len(P3D)]] * len(lines), dtype=np.float32))
        line_sets.append(line_set)
        vis.add_geometry(line_set)
        
    coord = open3d.create_mesh_coordinate_frame(0.1)
    vis.add_geometry(coord)

    for idx in range(len(P3D[0])):
        for jx in range(len(line_sets)):
            line_sets[jx].points = open3d.Vector3dVector(P3D[jx][idx])

        vis.update_geometry()

        vis.poll_events()
        vis.update_renderer()

        time.sleep(1. / FPS)
        # time.sleep(0.2)

    vis.destroy_window()


if __name__ == '__main__':
    import joblib
    np.set_printoptions(suppress=True, precision=4)

    joint_labels = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 
    7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 
    17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}


    data_dir = "output/data"
    # animation_file = osp.join(data_dir, "outdoors_freestyle_00.mp4_poses.npy")
    # animation_file = osp.join(data_dir, "thatswhatilike_2_poses.npy")

    # poses_3d = np.load(animation_file)[200:]

    data = joblib.load('vibe_output_global.pkl')
    poses_3d = data[0]['joints3d_new']
    # import pdb; pdb.set_trace()
    total_frames = poses_3d.shape[0]
    N = total_frames

    total_joints = len(connections) + 1

    print(total_joints, poses_3d.shape[1])
    assert total_joints == poses_3d.shape[1]

    mean_bone_length = np.zeros(total_joints, dtype=np.float32)

    root_j = 8 # midhip
    root_pos = poses_3d[:,np.newaxis,root_j].copy()
    poses_3d -= root_pos

    # set first frame of rootpos to origin (0,0,0)
    root_pos -= root_pos[0]
    root_pos *= 0.8 # scale the root motion

    lines = connections
    for c in connections:
        j_parent = c[0]
        j = c[1]

        # get the average bone/link length
        joint_child = poses_3d[:, j]
        joint_parent = poses_3d[:, j_parent]
        joint_dist = np.linalg.norm(joint_child - joint_parent, axis=1)
        mean_dist = np.mean(joint_dist)
        mean_bone_length[j] = mean_dist

    mean_bone_length[5:8] = mean_bone_length[2:5] # symmetric arms
    mean_bone_length[12:15] = mean_bone_length[9:12] # symmetric legs
    mean_bone_length[22:25] = mean_bone_length[19:22] # symmetric toes
    mean_bone_length[16] = mean_bone_length[15] # symmetric eyes
    mean_bone_length[18] = mean_bone_length[17] # symmetric ears

    print(mean_bone_length)

    ori_poses_3d = poses_3d.copy()
    # normalize so that all timeframe bones are same length
    for c in connections:
        j_parent = c[0]
        j = c[1]

        local_t = ori_poses_3d[:,j] - ori_poses_3d[:,j_parent]
        scale = dist(local_t, axis=-1)
        poses_3d[:,j] = poses_3d[:,j_parent] + local_t / scale[:,np.newaxis] * mean_bone_length[j]

    # apply moving average on position difference between frames to smoothen noise
    WINDOW_SIZE = 5
    pad = WINDOW_SIZE // 2
    window_is_odd = int(WINDOW_SIZE % 2 == 1)

    pos_window_avg = np.zeros_like(poses_3d[1:] - poses_3d[:-1])
    root_pos_window_avg = np.zeros_like(root_pos[1:] - root_pos[:-1])

    for idx in range(len(pos_window_avg)):
        start = max(0, idx - pad)
        end = min(N, idx + pad + window_is_odd)
        pos_window_avg[idx] = np.mean(poses_3d[start:end], axis=0)
        root_pos_window_avg[idx] = np.mean(root_pos[start:end], axis=0)

    poses_3d_copy = poses_3d.copy()
    poses_3d_copy[1:] = pos_window_avg
    poses_3d[:,:,1] += 1.

    # clamp the z factor
    root_z = root_pos_window_avg[:,0,2]
    N = len(root_z) // 4
    from scipy.signal import savgol_filter
    root_z[:] = savgol_filter(root_z, N if N % 2 != 0 else N-1, 2)

    # add back root trajectory
    poses_3d[1:] += root_pos_window_avg
    poses_3d_copy[1:] += root_pos_window_avg

    np.save("output/data/noise_reduce.npy", poses_3d_copy)

    visualize([poses_3d, poses_3d_copy], 20)
