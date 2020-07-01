import numpy as np
import time
from open3d_utils import *

dist = np.linalg.norm
norm = dist

def normalize(x):
    return x / dist(x)


joint_names = [
    'MidHip', # 0
    'RHip', 'RKnee', 'RAnkle', 'RToeBase', 'RToeEnd', # 1 2 3 4 5
    'LHip', 'LKnee', 'LAnkle', 'LToeBase', 'LToeEnd', # 6 7 8 9 10
    'Neck', 'Nose', 'HeadTop', # 11 12 13
    'LShoulder', 'LElbow', 'LWrist', # 14 15 16
    'RShoulder', 'RElbow', 'RWrist', # 17 18 19
]

parents = [
    -1,
    0, 1, 2, 3, 4,
    0, 6, 7, 8, 9,
    0, 11, 12,
    11, 14, 15,
    11, 17, 18
]

original_joint_labels = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 
    7: 'LWrist', 8: 'MidHip', 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 
    17: 'REar', 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel'}
original_joint_names = original_joint_labels.values()


class VIBESkeleton(object):
    def __init__(self):

        self.joint_names = joint_names
        self.parents = parents

        self.total_joints = len(self.parents)

        self.root_joint = self.parents.index(-1)

        """
        inds: list of joint indices belonging to that group. Let's call the first joint index in the group as "first joint" 
        root_orientation: the orientation of the "first joint" with respect to the root joint
        is_fixed_pivot: If True, means no rotation (dof) between "first joint" and root joint. 
                        If False, means 3 dof rotation between "first joint" and root joint
        """
        self.joint_groupings = {
            "leg_right": {"inds": [1,2,3,4,5], "root": 0, "root_orientation": [0,-np.deg2rad(90),0], "is_fixed_pivot": True },
            "leg_left": {"inds": [6,7,8,9,10], "root": 0, "root_orientation": [0,np.deg2rad(90),0], "is_fixed_pivot": True},
            "upper_body": {"inds": [11,12,13], "root": 0, "root_orientation": [0,0,np.deg2rad(90)], "is_fixed_pivot": False },
            "arm_left": {"inds": [14,15,16], "root": 11, "root_orientation": [0,np.deg2rad(90),0], "is_fixed_pivot": False },
            "arm_right": {"inds": [17,18,19], "root": 11, "root_orientation": [0,-np.deg2rad(90),0], "is_fixed_pivot": False }
        }

        self.symmetric_joints = {5: 1} # "pelvis_left symmetric with pelvis right"

        self.order_of_groupings = ["leg_right", "leg_left", "upper_body", "arm_left", "arm_right"]

    @staticmethod
    def preprocess(data): 
        """
        Converts the original joint data into our custom order (joint_names)
        data: (frames x N joints x 3), N should be 25
        """
        N, J = data.shape[:2]
        processed_data = np.zeros((N, len(joint_names), 3))

        # Add custom HeadTop
        pos_midear = np.mean(data[:,17:19], axis=1)
        pos_mideye = np.mean(data[:,15:17], axis=1)
        # nose_to_midear = (pos_midear + pos_mideye) * 0.5 - data[:,0]
        nose_to_midear = (pos_midear * 0.3 + pos_mideye * 0.7) - data[:,0]
        j = joint_names.index("HeadTop")
        processed_data[:, j] = data[:,0] + nose_to_midear * 2.5

        # Add custom RToeBase
        RAnkle = data[:,11] + (data[:,24] - data[:,11]) * 1.3
        pos_midtoeR = np.mean(data[:,22:24], axis=1)
        j = joint_names.index("RToeEnd")
        processed_data[:, j] = pos_midtoeR
        j = joint_names.index("RToeBase")
        t = 0.65
        processed_data[:, j] = pos_midtoeR * t + RAnkle * (1-t)
        
        # Add custom LToeBase
        LAnkle = data[:,14] + (data[:,21] - data[:,14]) * 1.3
        pos_midtoeL = np.mean(data[:,19:21], axis=1)
        j = joint_names.index("LToeEnd")
        processed_data[:, j] = pos_midtoeL
        j = joint_names.index("LToeBase")
        processed_data[:, j] = pos_midtoeL * t + LAnkle * (1-t)

        for jx, j in enumerate(joint_names):
            if j in original_joint_names:
                processed_data[:,jx] = data[:, original_joint_names.index(j)]

        # make left and right UpLegs symmetric
        root_pos = processed_data[:, 0:1, :].copy()
        processed_data -= root_pos

        R_hip = processed_data[:, 1]
        L_hip = processed_data[:, 6]
        L_to_R_hip_vector = normalize(R_hip - L_hip)
        hip_dist = norm(R_hip - L_hip)
        new_R_hip = L_to_R_hip_vector * hip_dist * 0.5
        new_L_hip = -L_to_R_hip_vector * hip_dist * 0.5
        R_diff = (new_R_hip - R_hip)[:,np.newaxis,:]
        L_diff = (new_L_hip - L_hip)[:,np.newaxis,:]

        processed_data[:, 1] = new_R_hip
        processed_data[:, 6] = new_L_hip
        processed_data[:, 2:6] += R_diff
        processed_data[:, 7:11] += L_diff

        processed_data += root_pos

        processed_data[:,:,:] -= processed_data[0,0]

        return processed_data


def visualize(poses_3d, lines=None, FPS=24):
    default_colors = [
        [0.,0.,0.],  # Black
        [1.,0.,0.],  # Red
        [0.,1.,0.],  # Green
        [0.,0.,1.],  # Blue
    ]

    if lines is None:
        lines = []
        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue
            lines.append([j, j_parent])

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
    np.set_printoptions(suppress=True, precision=4)
    from transforms3d.euler import euler2mat

    def euler2mat_deg(ee):
        return euler2mat(*np.deg2rad(ee))

    e2m = euler2mat
    e2md = euler2mat_deg

    # load original data
    poses_3d = np.load("../output/data/bruno_mars_filtered.npy")

    poses_3d = VIBESkeleton.preprocess(poses_3d)

    # N, J = poses_3d.shape[:2]
    # R_YZ_180 = euler2mat_deg([0, 180, 180])
    # poses_3d = np.dot(R_YZ_180, np.mat(poses_3d.reshape(N*J, 3)).T).T
    # poses_3d = np.array(poses_3d).reshape(N, J, 3)

    VIS = 1
    if VIS:
        visualize(poses_3d)

    out_file = "../output/data/processed_bruno_mars.npy"
    np.save(out_file, poses_3d)
    print("Saved to %s"%(out_file))
