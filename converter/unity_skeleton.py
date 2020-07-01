import numpy as np
import sys
from copy import deepcopy

from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
import time

from converter.lookat import camera_lookAt
from converter.node_class import Node, NodeSkeleton


def mat2euler_deg(M):
    ee = mat2euler(M)
    return np.rad2deg(ee)

def euler2mat_deg(ee):
    return euler2mat(*np.deg2rad(ee))

# import quaternion
# m2q = quaternion.from_rotation_matrix
# q2m = quaternion.as_rotation_matrix
# def quaternion_slerp(q1, q2, w=0.5):
#     return quaternion.slerp_evaluate(q1, q2, w)


def quaternion_slerp(q1, q2, w=0.5):
    dot = np.dot(q1, q2)

    # q3 = q2.copy() if dot >= 0 else -q2
    sign = np.sign(dot)
    dot = np.abs(dot)

    if dot > 0.999999:
        num3 = 1 - w
        num2 = w * sign
    else:
        num5 = np.arccos(dot)
        num6 = 1.0 / np.sin(num5)
        num3 = np.sin((1.0 - w) * num5) * num6
        num2 = np.sin(w * num5) * num6 * sign

    out = num3*q1 + num2*q2
    return out

m2q = mat2quat # quaternion.from_rotation_matrix
q2m = quat2mat # quaternion.as_rotation_matrix
e2m = euler2mat
e2md = euler2mat_deg
m2ed = mat2euler_deg
inv = np.linalg.inv
dist = np.linalg.norm

def assert_same(x, x2, eps=1e-4):
    assert np.all(np.abs(x - x2) < eps)


DTYPE = np.float32
vec3_zeros = np.zeros(3, dtype=DTYPE)


def rigid_transform_3D(A, B, translation=True):
    """
    Taken from http://nghiaho.com/uploads/code/rigid_transform_3D.py_
    """
    # Input: expects Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector

    assert A.shape == B.shape

    AA = np.mat(A) if not isinstance(A, np.matrix) else A.copy()
    BB = B.copy()
    if translation:
        centroid_A = np.mean(AA, axis=0)
        centroid_B = np.mean(BB, axis=0)

        # centre the points
        AA = AA - centroid_A
        BB = BB - centroid_B

    # dot is matrix multiplication for array
    H = AA.T * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T * U.T

    if translation:
        t = -R*centroid_A.T + centroid_B.T
        return R, t

    return R, None


class UnityAvatarSkeleton:

    def __init__(self, use_constraints=True):
        # ======== START OF INPUT ========
        self.joint_names = [
                     "Hips",  # 0
                     "RightUpLeg", "RightLeg",  "RightFoot", "RightToe_End",  # 1,2,3,4
                     "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe_End",       # 5,6,7,8
                     "Spine", "Spine1", "Spine2", "Neck", "Head",              # 9,10,11,12,13
                     "LeftShoulder","LeftArm", "LeftForeArm", "LeftHand",      # 14,15,16,17
                     "RightShoulder", "RightArm", "RightForeArm", "RightHand", # 18,19,20,21
        ]

        self.parents = [
                    -1,
                    0, 1, 2, 3, 
                    0, 5, 6, 7,
                    0, 9, 10, 11, 12,
                    11, 14, 15, 16,
                    11, 18, 19, 20 
        ]

        self.root_n_group = { # USED FOR IK
            0:  [[1,2,3,4], [5,6,7,8], [9,10,11]],  # solve root (hips) and pelvis first using SVD
            # then, solve for legs and first half of spine (before spine2 at index 11)
            11: [[12,13], [14,15,16,17], [18,19,20,21]] # then at spine2 (index 11), solve SVD
            # then solve for the rest (remaining spine/neck and arms) using lookat function again
        }

        self.bone_lengths = [
                    0,
                    8.6, 48.5, 43, 13,  # right leg
                    -1, -1, -1, -1, # left leg: symmetrical
                    2.86, 15.7, 12.86, 15.7, 5.7, # upper body
                    15.1, 11., 27., 27.,  # left arm
                    -1, -1, -1, -1  # right arm: symmetrical
        ]
        self.bone_lengths[5:9] = self.bone_lengths[1:5]
        self.bone_lengths[18:] = self.bone_lengths[14:18]

        self.forward_axis = 0  # bones forward pointing axis 
        self.axis_of_rotation = (self.forward_axis + 1) % 3 # up axis

        ROOT_ROTATION = e2md([-90,0,90])

        bl = self.bone_lengths
        # make sure that special_case is consistent with the forward_axis
        special_case = {
            1: {"LR": [0,0,180], "LT": [-bl[1],0,bl[1]]},  # RightUpLeg
            3: {"LR": [0,0,70], "LT": [bl[3],0,0]},  # RightFoot, LR needs to be adjusted for RightFoot_end
            4: {"LR": [0,0,20], "LT": [bl[4],0,0]},  # RightFoot_end, LR needs to be adjusted
            5: {"LR": [0,0,180], "LT": [-bl[5],0,-bl[5]]},  # LeftUpLeg
            7: {"LR": [0,0,70], "LT": [bl[7],0,0]},  # LeftFoot, LR needs to be adjusted for LeftFoot_end
            8: {"LR": [0,0,20], "LT": [bl[8],0,0]},  # LeftFoot_end, LR needs to be adjusted
            14: {"LR": [90,90,0], "LT": [11.4,0,-4.3]},  # LeftShoulder
            18: {"LR": [-90,-90,0], "LT": [11.4,0,4.3]},  # RightShoulder

            # # ADD BEND TO SPINE
            # 9: {"LR": [0, 0, -5], "LT": [bl[9], 0, 0]},  # Spine
            # 10: {"LR": [0, 0, 10], "LT": [bl[10], 0, 0]},  # Spine1
            # 11: {"LR": [0, 0, 5], "LT": [bl[11], 0, 0]},  # Spine2
            # 12: {"LR": [0, 0, -5], "LT": [bl[12], 0, 0]},  # Neck
            # 14: {"LR": [95, 90, 0], "LT": [11.4, 0, -4.3]},  # LeftShoulder
            # 18: {"LR": [-95, -90, 0], "LT": [11.4, 0, 4.3]},  # RightShoulder
        }

        # JOINT CONSTRAINTS
        self.USE_CONSTRAINTS = use_constraints
        constraints = {
            1: {"min": [-45,-70,-50], "max": [70,40,110]},
            2: {"min": [-1,-45,-165], "max": [1,45,0]},
            3: {"min": [0,0,-30], "max": [0,0,30]},
            9: {"min": [-10,-25,-80], "max": [10,25,20]}, # Spine
            10: {"min": [-25,-10,-15], "max": [25,10,15]}, # Spine1
            11: {"min": [-20,-10,-15], "max": [20,10,15]}, # Spine2
            12: {"min": [-70,-50,-85], "max": [70,50,60]}, # Neck
            14: {"min": [0,-30,-30], "max": [0,30,50]}, # shoulders
            15: {"min": [0,-145,-140], "max": [0,60,140]}, # shoulders to arm
            16: {"min": [0,-155,0], "max": [0,0,0]}, # elbow
        }
        for ix in [1,2,3,14,15,16]: # legs and hands symmetric
            cc = deepcopy(constraints[ix])
            tt = cc["min"][1] # Y-axis needs to be flipped
            cc["min"][1] = cc["max"][1] * -1
            cc["max"][1] = tt * -1
            constraints[ix+4] = cc

        # ======== END OF INPUT ========

        self.constraints = constraints
        self.constraints_radians = deepcopy(self.constraints)
        for j in self.constraints_radians:
            self.constraints_radians[j]["min"] = np.radians(self.constraints[j]["min"])
            self.constraints_radians[j]["max"] = np.radians(self.constraints[j]["max"])

        self.joint_subchilds = {}
        for j in range(self.get_N_joints()):
            self.joint_subchilds[j] = self.get_all_joint_childs(j, subchilds=True)

        self.root_inds_to_child = {k: [vv[0] for vv in v] for k, v in self.root_n_group.items()}

        assert len(self.parents) == len(self.joint_names)

        total_joints = self.get_N_joints()
        self.root_joint = self.parents.index(-1)
        self.local_translations = np.zeros((total_joints, 3), dtype=DTYPE)
        ax = self.forward_axis
        for j in range(total_joints):
            self.local_translations[j, ax] = bl[j]

        self.local_rotations = np.zeros((total_joints, 3, 3), dtype=DTYPE) 
        self.local_rotations[:] = np.eye(3)
        self.local_rotations[self.root_joint] = ROOT_ROTATION
        for j,v in special_case.items():
            self.local_rotations[j] = e2md(v["LR"])
            self.local_translations[j] = v["LT"]

        nodes = self.create_nodes()
        self.add_node_hierarchy(nodes)
        self.node_skeleton = self.create_node_skeleton(nodes)

        self.nodes = nodes
        self.set_node_local_poses(self.local_rotations, self.local_translations)

    def create_nodes(self):
        joint_names = self.get_joint_names()
        N_joints = len(joint_names)

        print("Creating nodes...")
        nodes = []
        for i in range(N_joints):
            node_name = joint_names[i]
            node = Node(node_name)
            nodes.append(node)
        return nodes

    def add_node_hierarchy(self, nodes):
        N_joints = self.get_N_joints()
        parents = self.get_joint_parents()

        print("Adding hierarchy...")
        for i in range(N_joints):
            if parents[i] >= 0:
                nodes[parents[i]].AddChild(nodes[i])

    def create_node_skeleton(self, nodes):
        sk = NodeSkeleton()
        sk.SetNodes(nodes)
        return sk

    def reset_to_tpose(self):
        self.set_node_local_poses(self.local_rotations, self.local_translations)

    def get_N_joints(self):
        return len(self.joint_names)

    def get_joint_names(self):
        return self.joint_names

    def get_joint_parents(self):
        return self.parents

    def get_all_joint_childs(self, j, subchilds=False):
        c = []
        for ix, p in enumerate(self.parents):
            if p == j:
                c.append(ix)
                if subchilds:
                    c += self.get_all_joint_childs(ix, subchilds=subchilds)
        return c

    def set_node_local_poses(self, local_rotations=None, local_translations=None):
        N_joints = self.get_N_joints()
        if local_rotations is None and local_translations is None:
            return

        if local_rotations is not None:
            assert len(local_rotations) == N_joints
        else:
            local_rotations = [None] * N_joints

        if local_translations is not None:
            assert local_translations.shape == (N_joints, 3)
        else:
            local_translations = [None] * N_joints

        for joint in range(N_joints):
            self.set_node_local_pose_by_index(joint, \
                local_rotations[joint], local_translations[joint])

    def set_node_local_pose_by_index(self, j, LR=None, LT=None):
        node = self.nodes[j]

        M = node.EvaluateLocalTransform()
        if LR is not None:
            M[:3,:3] = LR 

        if LT is not None:
            M[:3,3] = LT

        node.SetLocalTransform(M)

    def get_node_transform_by_index(self, j, global_=True):
        node = self.nodes[j]
        t_func = node.EvaluateGlobalTransform if global_ \
                        else node.EvaluateLocalTransform
        return t_func()

    def get_node_transforms(self, LR=None, LT=None, global_=True):
        self.set_node_local_poses(LR, LT)
        MM = self.node_skeleton.EvaluateNodeTransforms(local=not global_)
        return MM

    def visualize_transforms(self, GT=None, extra_scene_objs=[], frame_size=None):
        from open3d_utils import open3d, V2iV, V3dV

        if GT is None:
            GT = self.get_node_transforms()
        if frame_size is None:
            frame_size = self.bone_lengths[1]

        coords = [open3d.create_mesh_coordinate_frame(frame_size*1.33)]
        lines = []
        for j, j_parent in enumerate(self.parents):
            coord = open3d.create_mesh_coordinate_frame(frame_size)
            coord.transform(GT[j])
            coords.append(coord)
            
            if j_parent != -1:
                lines.append([j_parent, j])

        line_set = open3d.geometry.LineSet()
        line_set.lines = V2iV(lines)
        line_set.points = V3dV(GT[:,:3,3])

        open3d.visualization.draw_geometries(coords + [line_set] + extra_scene_objs)

    def compute_root_joint_rotation(self, root_ind, source_positions, target_positions, extra_inds=[]):
        """
        Compute the rigid transform rotation of a joint's ("root_ind") children from source to target
        Extra inds (optional) means additional indices to add to source+target for calculating rotation
        """
        if root_ind not in self.root_inds_to_child and len(extra_inds) == 0:
            return False, None

        assert source_positions.shape == target_positions.shape
        assert len(source_positions) == self.get_N_joints()

        child_js = self.root_inds_to_child[root_ind]
        targets = np.zeros((len(child_js), 3), dtype=np.float32)
        sources = targets.copy()

        for ix, j in enumerate(child_js):
            targets[ix] = target_positions[j] - target_positions[root_ind]
            sources[ix] = source_positions[j] - source_positions[root_ind]
        for ind in extra_inds:
            targets = np.vstack((targets, target_positions[ind] - target_positions[self.parents[ind]]))
            sources = np.vstack((sources, source_positions[ind] - source_positions[self.parents[ind]]))

        # perform SVD
        apply_R, _ = rigid_transform_3D(sources, targets, translation=False)

        # sense check
        out = np.dot(apply_R, sources.T).T
        assert_same(out, targets)

        return True, apply_R

    def solve_node_lookat_R(self, j, target):
        tmp_GT = self.get_node_transform_by_index(j, global_=True)

        up = tmp_GT[:3,self.axis_of_rotation]
        M = camera_lookAt(target, vec3_zeros, up, forward_axis=self.forward_axis)
        R = M[:3,:3]

        parent_GT = self.get_node_transform_by_index(self.parents[j], global_=True)
        loc_R = np.dot(inv(parent_GT[:3,:3]), R)
        return loc_R, R

    def inverse_kinematics(self, positions, ignore_joints=[]):
        N_joints = self.get_N_joints()

        assert positions.shape == (N_joints, 3)

        # do IK
        for root_idx, grp_inds in self.root_n_group.items():
            if root_idx in ignore_joints:
                continue       
            GT2 = self.get_node_transforms()
            pj = self.parents[root_idx]
            
            # solve root using SVD
            ret, apply_R = self.compute_root_joint_rotation(root_idx, GT2[:,:3,3], positions)
            assert ret

            root_R = np.dot(apply_R, GT2[root_idx,:3,:3])
            root_LR = root_R.copy()
            if root_idx == 11: # TODO: CONFIG
                # for Spine1, slerp rotation between Spine2 and Spine
                ppj = self.parents[pj] # Spine
                ppj_R = GT2[ppj,:3,:3]
                pj_Q = quaternion_slerp(m2q(ppj_R), m2q(root_R), w=0.5) # slerp
                pj_R = q2m(pj_Q)
                pj_LR = np.dot(inv(ppj_R), pj_R)
                if self.USE_CONSTRAINTS:
                    is_constrained, pj_LR = self.constrain_joint_LR(pj, pj_LR)
                    if is_constrained: 
                        print("CONSTRAINED %d"%(pj))
                        pj_R = np.dot(ppj_R, pj_LR)
                    root_LR = np.dot(inv(pj_R), root_R)
                    is_constrained, root_LR = self.constrain_joint_LR(root_idx, root_LR)
                    if is_constrained: # NEED TO ROTATE ALL POSITIONS BASED ON THIS PIVOT
                        print("CONSTRAINED %d"%(root_idx))
                        new_root_R = np.dot(pj_R, root_LR)
                        apply_R = np.dot(new_root_R, inv(root_R))
                        ppp = positions[self.joint_subchilds[root_idx]] - positions[root_idx]
                        ppp_rotated = np.dot(apply_R, np.mat(ppp).T).T
                        positions[self.joint_subchilds[root_idx]] = ppp_rotated + positions[root_idx]
                        root_R = np.dot(pj_R, root_LR)
                        # angles_rotated = list(mat2euler(inv(self.local_rotations[root_idx]).dot(root_LR)))
                        # print(np.degrees(angles_rotated))
                        # self.set_node_local_pose_by_index(root_idx, LR=root_LR)
                        # self.visualize_transforms()
                        # self.set_node_local_pose_by_index(root_idx, LR=np.dot(inv(pj_R), root_R))
                        # self.visualize_transforms()
                    # TODO: re-slerp pj_LR?
                self.set_node_local_pose_by_index(pj, LR=pj_LR)
                root_LR = np.dot(inv(pj_R), root_R)

            self.set_node_local_pose_by_index(root_idx, LR=root_LR)

            root_pos = self.get_node_transform_by_index(root_idx)[:3,3]
            diff = root_pos - positions[root_idx]
            if np.sum(diff) != 0:
                positions[self.joint_subchilds[root_idx]] += diff

            # using lookat function
            for inds in grp_inds:
                for ix, j in enumerate(inds[:-1]): # don't need to set any rotation for last end node, ignore
                    next_ind = inds[ix+1]

                    # CHECK
                    if next_ind in ignore_joints:
                        continue

                    target = positions[next_ind] - positions[j]
                    rot_matrix, R = self.solve_node_lookat_R(j, target)

                    is_constrained = False

                    if self.USE_CONSTRAINTS:
                        if j in [2, 3, 6, 7]:  # TODO: CONFIG
                            angles = list(mat2euler(rot_matrix))
                            angles[self.forward_axis] = 0
                            rot_matrix = euler2mat(*angles)

                        is_constrained, rot_matrix2 = self.constrain_joint_LR(j, rot_matrix)
                        if is_constrained:
                            rot_matrix = rot_matrix2

                            """
                            EXTRA CONSTRAINTS FOR LEGS:
                            if distance offset is large, try comparing it with the distance offset if there is
                            a different rotation config of the parent(s)
                            Here, we simply compare to the rotation where the base parent has forward-axis (x) rotation set to 0
                            """
                            if j in [2, 6]: # TODO: CONFIG
                                pj = self.parents[j]
                                pj_R = self.get_node_transform_by_index(pj, global_=True)[:3,:3]
                                R = np.dot(pj_R, rot_matrix)
                                tt = np.dot(R, self.local_translations[next_ind])
                                dist_offset = dist(tt - target)
                                if dist_offset > 0.2 * self.bone_lengths[next_ind]:
                                    pj_LR = self.get_node_transform_by_index(pj, global_=False)[:3, :3]
                                    angles = list(mat2euler(pj_LR))
                                    angles[self.forward_axis] = 0
                                    pj_LR_x0 = euler2mat(*angles)
                                    ppj_R = self.get_node_transform_by_index(self.parents[pj], global_=True)[:3, :3]
                                    pj_R_x0 = np.dot(ppj_R, pj_LR_x0)
                                    self.set_node_local_pose_by_index(pj, pj_LR_x0) # TEMPORARILY, JUST FOR LOOKAT
                                    rot_matrix_x0, _ = self.solve_node_lookat_R(j, target)
                                    is_constrained1, rot_matrix_x0 = self.constrain_joint_LR(j, rot_matrix_x0)
                                    R = np.dot(pj_R_x0, rot_matrix_x0)
                                    tt = np.dot(R, self.local_translations[next_ind])
                                    dist_offset2 = dist(tt - target)
                                    if dist_offset2 < dist_offset:
                                        print("YES", j)
                                        rot_matrix = rot_matrix_x0
                                        # out_j_t = self.get_node_transform_by_index(j, global_=True)[:3,3]
                                        # positions[self.joint_subchilds[pj]] += out_j_t - positions[j]
                                    else:
                                        self.set_node_local_pose_by_index(pj, pj_LR)  # SET BACK TO PREVIOUS
                            """END EXTRA CONSTRAINTS FOR LEGS"""

                    self.set_node_local_pose_by_index(j, rot_matrix)

                    out = self.get_node_transform_by_index(next_ind, global_=True)
                    if is_constrained:
                        # update all child positions by the constrained offset
                        pos_diff = out[:3,3] - positions[next_ind]
                        positions[self.joint_subchilds[j]] += pos_diff
                    else:
                        assert_same(out[:3,3], positions[next_ind])

    def constrain_joint_LR(self, j, LR):
        if j not in self.constraints_radians:
            return False, LR

        is_constrained = False
        LR2 = LR.copy()
        min_con = self.constraints_radians[j]["min"]
        max_con = self.constraints_radians[j]["max"]
        angles_rotated = list(mat2euler(inv(self.local_rotations[j]).dot(LR)))
        for ax in [0, 1, 2]:
            if max_con[ax] > min_con[ax]:
                if angles_rotated[ax] > max_con[ax] or angles_rotated[ax] < min_con[ax]:
                    is_constrained = True
                    angles_rotated[ax] = np.clip(angles_rotated[ax], min_con[ax], max_con[ax])
        if is_constrained:
            LR2 = np.dot(self.local_rotations[j], euler2mat(*angles_rotated))
        return is_constrained, LR2

    def get_random_rotations(self, max_rotation_theta=np.radians(15.0)):
        N = self.get_N_joints()

        # random rotation
        euler_ax_angles = np.random.uniform(-max_rotation_theta, max_rotation_theta, size=(N, 3))
        rotations = np.array([e2m(*R) for R in euler_ax_angles])

        LR = self.get_node_transforms(global_=False)[:,:3,:3]
        for j, R in enumerate(rotations):
            rotations[j] = np.dot(LR[j], R)
        return rotations

    # TESTSSS
    def rotation_unit_test(self):
        GT = self.get_node_transforms()
        LR = self.get_node_transforms(global_=False)[:,:3,:3]
        rand_rotations = self.get_random_rotations()
        self.set_node_local_poses(rand_rotations)
        GT2 = self.get_node_transforms()

        self.set_node_local_poses(LR)
        GT3 = self.get_node_transforms()
        self.set_node_local_poses(rand_rotations)
        GT4 = self.get_node_transforms()

        assert_same(GT, GT3)
        assert_same(GT2, GT4)

        self.set_node_local_poses(LR)
        print("PASSED Rotation Test")

    def inverse_kinematics_unit_test(self):
        self.USE_CONSTRAINTS = False
        np.random.seed(123)

        GT = self.get_node_transforms()
        positions = GT[:,:3,3]

        # randomly rotate
        local_rotations = self.get_random_rotations()
        self.set_node_local_poses(local_rotations)
        GTX = self.get_node_transforms()
        positionsX = GTX[:,:3,3]

        self.inverse_kinematics(positions)
        GT2 = self.get_node_transforms()
        self.inverse_kinematics(positionsX)
        GTX2 = self.get_node_transforms()

        # all global positions/translations must be same
        assert_same(positions, GT2[:, :3, 3])
        assert_same(positionsX, GTX2[:, :3, 3])

        # global rotations for key root joints must be same
        all_root_inds = list(self.root_inds_to_child.keys())
        assert_same(GT[all_root_inds, :3, :3], GT2[all_root_inds, :3, :3])
        assert_same(GTX[all_root_inds, :3, :3], GTX2[all_root_inds, :3, :3])

        print("PASSED IK Test")


def run_unit_test(usc):
    usc.rotation_unit_test()
    usc.inverse_kinematics_unit_test()

def test_visualizer(usc):
    from open3d_utils import open3d, V2iV, V3dV

    GT = usc.get_node_transforms()

    lines = []
    coords = []
    for j, j_parent in enumerate(usc.parents):
        coord = open3d.create_mesh_coordinate_frame(usc.bone_lengths[1])
        coord.transform(GT[j])
        coords.append(coord)
        
        if j_parent != -1:
            lines.append([j_parent, j])

    line_set = open3d.geometry.LineSet()
    line_set.lines = V2iV(lines)
    line_set.points = V3dV(GT[:,:3,3])

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    for coord in coords:
        vis.add_geometry(coord)

    prev_GT = GT.copy()

    LR = usc.get_node_transforms(global_=False)[:,:3,:3]

    print("Currently visualizing joint constraint limits...")
    constraints = usc.constraints
    for j in constraints:
        print(usc.joint_names[j])
        for ax in range(3):
            min_ax = constraints[j]["min"][ax]
            max_ax = constraints[j]["max"][ax]
            # if np.abs(max_ax - min_ax) < 1e-3:
            if max_ax - min_ax < 1e-3:
                continue
            print(ax)
            usc.set_node_local_poses(LR)
            GT = usc.get_node_transforms() 
            for jx, coord in enumerate(coords):
                T = np.dot(GT[jx], inv(prev_GT[jx]))
                coord.transform(T)
            prev_GT = GT.copy()
            ax_euler = np.zeros(3)
            for a in range(min_ax, max_ax):
                ax_euler[ax] = a
                apply_R = e2md(ax_euler)
                R = np.dot(LR[j], apply_R)

                usc.set_node_local_pose_by_index(j, R)
                GT = usc.get_node_transforms()
                for jx, coord in enumerate(coords):
                    T = np.dot(GT[jx], inv(prev_GT[jx]))
                    coord.transform(T)
                line_set.points = V3dV(GT[:,:3,3])

                prev_GT = GT.copy()

                vis.update_geometry()

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.01)
                vis.run()

if __name__ == '__main__':

    np.set_printoptions(suppress=True, precision=4)


    usc = UnityAvatarSkeleton()
    # usc.visualize_transforms()
    test_visualizer(usc)
    # run_unit_test(usc)

    # usc.visualize_transforms()
    # randomly rotate
    # local_rotations = usc.get_random_rotations()
    # usc.set_node_local_poses(local_rotations)
    # GTX = usc.get_node_transforms()
    # positionsX = GTX[:,:3,3]

    # usc.set_node_local_poses(usc.local_rotations)
    # usc.inverse_kinematics(positionsX)
    # usc.visualize_transforms()

    # usc.reset_to_tpose()
    # usc.visualize_transforms()
    