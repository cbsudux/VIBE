import numpy as np
from numpy.random import normal as normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib

from numpy import load

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

poses = load('thatswhatilike_1_poses.npy')

number_of_frames = poses.shape[0] - 1 # Number of frames
# number_of_frames = 300
fps = 24 # Frame per sec

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1.5, 1)
ax.view_init(-170, 80) 
ax.invert_xaxis()

lines_3d = [plt.plot([],[],[], color='blue', linestyle='-', linewidth=2, marker='o', markersize=5)[0] for _ in range(len(connections))]


def update(frame_id):
    joints3D = poses[frame_id][:25]

    for i,pair in enumerate(connections):
        j1 = pair[0]
        j2 = pair[1]
        lines_3d[i].set_xdata([joints3D[j1, 0], joints3D[j2, 0]])
        lines_3d[i].set_ydata([joints3D[j1, 2], joints3D[j2, 2]])
        lines_3d[i].set_3d_properties([joints3D[j1, 1], joints3D[j2, 1]], zdir = 'z')

    return lines_3d,

from time import time
t0 = time()
update(0)
t1 = time()
dt = 1./fps
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, update, number_of_frames, interval=1)

# plt.show()
ani.save(f'thatswhatilike_1_animation_{fps}fps.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])