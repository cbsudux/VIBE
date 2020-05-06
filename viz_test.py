import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import joblib
from lib.models.smpl import get_smpl_faces
import trimesh

import numpy as np
from lib.models.spin import Regressor, hmr


joint_labels ={0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 
7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 
17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}


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


# def display_model(
#         model_info,
#         model_faces=None,
#         with_joints=False,
#         kintree_table=None,
#         ax=None,
#         batch_idx=0,
#         show=True,
#         savepath=None):
#     """
#     Displays mesh batch_idx in batch of model_info, model_info as returned by
#     generate_random_model
#     """
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#     # joints = model_info['verts'][batch_idx], model_info['joints'][
#         # batch_idx]
#     # verts, joints = model_info['verts'][batch_idx], model_info['joints'][
#         # batch_idx]
#     # if model_faces is None:
#     #     ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
#     # else:
#     #     mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
#     #     face_color = (141 / 255, 184 / 255, 226 / 255)
#     #     edge_color = (50 / 255, 50 / 255, 50 / 255)
#     #     mesh.set_edgecolor(edge_color)
#     #     mesh.set_facecolor(face_color)
#     #     ax.add_collection3d(mesh) 
#     if with_joints:
#         draw_skeleton(joints, kintree_table=kintree_table, ax=ax)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_xlim(-0.7, 0.7)
#     ax.set_ylim(-0.7, 0.7)
#     ax.set_zlim(-0.7, 0.7)
#     ax.view_init(azim=-90, elev=100)
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     if savepath:
#         print('Saving figure at {}.'.format(savepath))
#         plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
#     if show:
#         plt.show()
#     return ax


# def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=True):
#     if ax is None:
#         fig = plt.figure(frameon=False)
#         ax = fig.add_subplot(111, projection='3d')
#     else:
#         ax = ax

#     colors = []
#     left_right_mid = ['r', 'g', 'b']
#     kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
#     for c in kintree_colors:
#         colors += left_right_mid[c]
#     # For each 24 joint
#     # import pdb;pdb.set_trace()

#     for i in range(1, kintree_table.shape[1]):
#         j1 = kintree_table[0][i]
#         j2 = kintree_table[1][i]
#         ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
#                 [joints3D[j1, 1], joints3D[j2, 1]],
#                 [joints3D[j1, 2], joints3D[j2, 2]],
#                 color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
#         if with_numbers:
#             ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
#     return ax

#based on https://github.com/vchoutas/smplx/blob/03813b7ffab9e9a9a0dfbf441329dedf5ae6176e/examples/demo.py#L69
def plot_skeleton(poses):
  fig = plt.figure(figsize=(5,5))
  ax = fig.add_subplot(121, projection='3d')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1.5, 1)

  # import pdb;pdb.set_trace()
  # for i,joints3D in enumerate(poses):
  joints3D = poses[0]
  # print('Plotting pose ', i)
  for pair in connections:
    j1 = pair[0]
    j2 = pair[1]
    # ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
    #         [joints3D[j1, 1], joints3D[j2, 1]],
    #         [joints3D[j1, 2], joints3D[j2, 2]],
    #         color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)
    # ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
            [joints3D[j1, 2], joints3D[j2, 2]],
            [joints3D[j1, 1], joints3D[j2, 1]],
            color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)
    # ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)

  ax.view_init(-170, 80)  
  plt.show()

  for angle in range(0,360,10):
    ax.view_init(-angle, 60)
    print(-angle)
    plt.draw()
    plt.pause(.0001)
    # plt.cla()

def plot_mesh_joints(joints,vertices=None,faces=None):
  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(121, projection='3d')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_zlim(-1.5, 1)

  ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color='r')
  for i in range(24):
    ax.text(joints[i, 0], joints[i, 2], joints[i, 1], str(i))
  ax.view_init(-140, 30)

  plt.draw()
  plt.show()

vibe_data = joblib.load('output/sample_video/vibe_output.pkl')

vibe_3d_pose = vibe_data[1]['joints3d']
print('VIBE output keys \n {}'.format(list(vibe_data[1].keys())))

# joints_3d = vibe_3d_pose[6][:25]
joints_3d = vibe_3d_pose[:25]
# plot_mesh_joints(joints=joints_3d)
plot_skeleton(poses = joints_3d)