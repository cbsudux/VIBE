import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from converter.unity_skeleton import UnityAvatarSkeleton
from converter.vibe_skeleton import VIBESkeleton, visualize

from converter.rotate_unity import quat_to_unity_quat, position_to_unity_position, transform_to_unity_transform
from open3d_utils import open3d, V2iV, V3dV

inv = np.linalg.inv
dist = np.linalg.norm
d2r = np.deg2rad
r2d = np.rad2deg

vec3_zeros = np.zeros(3, dtype=np.float32)

def mat2euler_deg(M):
    ee = mat2euler(M)
    return np.rad2deg(ee)

def euler2mat_deg(ee):
    return euler2mat(*np.deg2rad(ee))

e2m = euler2mat
e2md = euler2mat_deg
m2ed = mat2euler_deg

def assert_same(x, x2, eps=1e-4):
    assert np.all(np.abs(x - x2) < eps)

def normalize(x):
    return x / dist(x)

def lookat_2D(M, direction, axis_of_rotation=0):
    assert direction.size == 3
    assert 0 <= axis_of_rotation <= 2

    ax = np.zeros(3, dtype=np.float32)
    ax[axis_of_rotation] = 1
    other_axes = np.where(ax==0)[0]
    a1,a2 = other_axes

    # project to y and z axis vectors to form yz plane
    dot_f_dt = np.dot(M[:3,a1], direction)
    dot_r_dt = np.dot(M[:3,a2], direction)
    # compute angle between the two points on yz plane
    # the angle is such that the z vector is pointing towards the direction
    at = np.arctan2(dot_f_dt, dot_r_dt)
    ax *= -at
    R_axis = euler2mat(*ax)
    return R_axis

def visualize_skeleton_rotations(skeleton_obj, rotations, root_positions=None):#, all_GT):
    """
    skeleton_obj: Instance of UnityAvatarSkeleton
    rotations: (batch, N, joints, 3, 3) or (N, joints, 3, 3), storing the rotation matrices per frame per joint, where N = frames
    root_positions: (batch, N, 3) or (N, 3), where N = frames
    """
    import time

    parents = skeleton_obj.parents
    root_j = parents.index(-1)

    if len(rotations) == 0:
        return
    assert rotations.shape[-3:] == (len(parents), 3, 3)

    if len(rotations.shape) == 4:
        rotations = rotations[np.newaxis,...]  # (1, N, joints, 3, 3)
    batch, frames = rotations.shape[:2]

    if root_positions is not None:
        shape = root_positions.shape
        assert shape[-2:] == (frames, 3)
        if len(shape) >= 3:
            assert shape[0] == batch and len(shape) == 3
        else:
            root_positions = root_positions[np.newaxis,...] # (1, N, 3)

    lines = []
    for j, j_parent in enumerate(parents):
        if j_parent != -1:
            lines.append([j_parent, j])

    GTS = []
    for b in range(batch):
        skeleton_obj.set_node_local_poses(rotations[b,0])
        if root_positions is not None:
            skeleton_obj.set_node_local_pose_by_index(root_j, LT=root_positions[b,0])
        GTS.append(skeleton_obj.get_node_transforms())

    colors = [[0,0,0], [1,0,0], [0,0,1], [0,1,0]]

    line_sets = []
    for b in range(batch):
        color_ = colors[b%batch]
        line_set = open3d.geometry.LineSet()
        line_set.lines = V2iV(lines)
        line_set.points = V3dV(GTS[b][:, :3, 3])
        line_set.colors = V3dV([color_] * len(lines))
        line_sets.append(line_set)

    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    for line_set in line_sets:
        vis.add_geometry(line_set)

    # add coords to vis
    frame_size = skeleton_obj.bone_lengths[1]
    coords = []
    for b in range(batch):
        coords.append([])
        for M in GTS[b]:
            coord = open3d.create_mesh_coordinate_frame(frame_size)
            coord.transform(M)
            coords[b].append(coord)
            vis.add_geometry(coord)

    vis.add_geometry(open3d.create_mesh_coordinate_frame(frame_size*1.33))

    global FRAME, NEXT, QUIT
    QUIT = 0
    FRAME = 0
    NEXT = 0
    def next_callback(vis):
        global NEXT, FRAME
        NEXT = 1
        FRAME += 1
        if FRAME >= frames:
            FRAME = 0
    def prev_callback(vis):
        global NEXT, FRAME
        NEXT = 1
        FRAME -= 1
        if FRAME < 0:
            FRAME = frames - 1

    def quit_callback(vis):
        global QUIT
        QUIT = 1

    next_key = "N"
    prev_key = "P"
    quit_key = "Q"
    vis.register_key_callback(ord(next_key.upper()), next_callback)
    vis.register_key_callback(ord(prev_key.upper()), prev_callback)
    vis.register_key_callback(ord(quit_key.upper()), quit_callback)
    print("Press %s to go to next frame, %s to go prev frame"%(next_key, prev_key))
    print("Press %s to quit."%(quit_key))

    vis.update_geometry()
    vis.update_renderer()

    while 1:
        vis.poll_events()
        time.sleep(0.01)
        if QUIT:
            break
        if NEXT:
            NEXT = 0
            print("Frame %d"%FRAME)
            for b in range(batch):
                skeleton_obj.set_node_local_poses(rotations[b][FRAME])
                if root_positions is not None:
                    skeleton_obj.set_node_local_pose_by_index(root_j, LT=root_positions[b][FRAME])
                GTX = skeleton_obj.get_node_transforms()
                line_sets[b].points = V3dV(GTX[:, :3, 3])

                for jx, M in enumerate(GTX):
                    apply_M = np.dot(M, inv(GTS[b][jx]))
                    coords[b][jx].transform(apply_M)

                GTS[b] = GTX.copy()
                # assert_same(GTX, all_GT[idx])

            vis.update_geometry()
            vis.update_renderer()

    vis.destroy_window()


class VIBE_Positions_To_Unity_Skeleton_Converter:
    def __init__(self, vibe_skeleton=VIBESkeleton(), unity_skeleton=UnityAvatarSkeleton(use_constraints=True)):
        self.v_to_u_map = [
            0,
            1, 2, 3, 4, -1, # but LToeEnd will be needed for Ltoebase direction/rotation
            5, 6, 7, 8, -1,  # but RToeEnd will be needed for Rtoebase direction/rotation
            12, -1, -1,  # but nose and headtop will be needed for head direction
            15, 16, 17,
            19, 20, 21
        ]
        self.u_to_v_map = [
            0,
            1, 2, 3, 4,
            6, 7, 8, 9,
            -1, -1, -1, 11, -1,
            -1, 14, 15, 16,
            -1, 17, 18, 19
        ]

        self.vibe = vibe_skeleton
        self.usc = unity_skeleton

        assert len(self.v_to_u_map) == len(self.vibe.joint_names)
        assert len(self.u_to_v_map) == len(self.usc.joint_names)

    def solve_LR(self, vibe_positions_3d):
        vibe = self.vibe
        usc = self.usc
        v_to_u_map = self.v_to_u_map
        u_to_v_map = self.u_to_v_map


        total_vibe_joints = len(vibe.joint_names)
        assert vibe_positions_3d.shape[-2:] == (total_vibe_joints, 3)        
        total_usc_joints = len(usc.joint_names)

        usc.reset_to_tpose()

        tpose_GT = usc.get_node_transforms()
        tpose_pos = tpose_GT[:,:3,3]

        # # used for lookat rotation on x-axis (yz plane)
        default_neck_rotation = tpose_GT[12, :3,:3]
        R_forward_z_on_upward_xaxis = e2md([0,0,90]) # np.dot(e2md([90, 0, 0]), e2md([0,0,90]))
        R_forward_z_to_default_neck_rotation = np.dot(inv(R_forward_z_on_upward_xaxis), default_neck_rotation)

        tpose_shoulder_pos = (tpose_pos[18] + tpose_pos[14]) * 0.5
        spine2_to_tpose_shoulder_dist = dist(tpose_shoulder_pos - tpose_pos[11])
        spine2_to_tpose_neck_dist = dist(tpose_pos[12] - tpose_pos[11])
        tpose_shoulder_width = dist(tpose_pos[18] - tpose_pos[14]) * 0.5
        spine2_to_shoulder_vs_neck_ratio = spine2_to_tpose_shoulder_dist / spine2_to_tpose_neck_dist

        mid_up_leg_pos = (tpose_pos[1] + tpose_pos[5]) * 0.5
        hip_to_leg_vector = mid_up_leg_pos - tpose_pos[0]
        hip_to_leg_dist = dist(hip_to_leg_vector)

        pos = vibe_positions_3d.copy()
        # normalize to root (0,0,0)
        posn = pos.copy()
        posn -= posn[0]

        # make left and right UpLegs symmetric
        posn[6] = posn[1] * -1
        for i in range(7,11):
            posn[i] = posn[i-1] + pos[i] - pos[i-1]
        pos = posn

        pos2 = np.zeros((total_usc_joints, 3))

        """first, start with the legs (easiest, since mapping is the same)"""
        v_leg_joints = [[1, 5],[6,10]] # [[L leg start, L Leg end], [R leg start, R leg end]]
        for jj in v_leg_joints:
            leg_joints = v_to_u_map[jj[0]:jj[1]]
            for ux in leg_joints: 
                vx = u_to_v_map[ux]
                local_t = pos[vx] - pos[vibe.parents[vx]]
                pos2[ux] = pos2[usc.parents[ux]] + normalize(local_t) * usc.bone_lengths[ux]

        """
        then project "Hip/Spine" in the opposite direction of the feet
        # subject to being perpendicular to Hips: the line connecting RightUpLeg and LeftUpLeg
        """
        R_upleg_to_foot = pos2[3] - pos2[1]
        L_upleg_to_foot = pos2[7] - pos2[5]
        spine_vector = np.mean([R_upleg_to_foot, L_upleg_to_foot], axis=0) * -1 # opposite
        hip_vector = pos2[1] - pos2[5]  # rightupleg and leftupleg
        # get perpendicular vector of plane formed by hip_vector + spine vector
        perp = np.cross(hip_vector, spine_vector)
        spine_vector = np.cross(hip_vector, -normalize(perp)) # perpendicular to unit perp vector
        spine_vector = normalize(spine_vector) # normalize to unit vector

        """
        offset legs to hip
        """
        usc_leg_joints = [1,2,3,5,6,7]
        if hip_to_leg_dist > 0:
            pos2[usc_leg_joints] -= spine_vector * hip_to_leg_dist

        # project magnitude (bone length) of spine vector to get "Spine" position
        spine_u = 9 # Spine
        spine_pos = spine_vector * usc.bone_lengths[spine_u]
        pos2[spine_u] = pos2[usc.parents[spine_u]] + spine_pos

        """next, Set Neck"""
        hip_u = 0
        neck_u = 12 # Neck
        hip_v = u_to_v_map[hip_u]
        neck_v = u_to_v_map[neck_u]

        spine1_u = 10
        spine2_u = 11

        # for Spine1, Spine2, Neck, linearly interpolate all of them with Hips -> Neck vector (since vibe doesnt have spine information)
        local_t = pos[neck_v] - pos[hip_v] # !! hip_v is the parent of neck_v
        hip_to_neck = normalize(local_t)
        inds = [spine_u, spine1_u, spine2_u, neck_u]
        for ix, ind in enumerate(inds[1:]):
            pos2[ind] = pos2[inds[ix]] + usc.bone_lengths[ind] * hip_to_neck

        """then solve for Head position"""
        ux = 13 # head
        local_t = pos[13] - pos[11] # using HeadTop to Neck vector
        pos2[ux] = pos2[usc.parents[ux]] + normalize(local_t) * usc.bone_lengths[ux]

        """
        then, solve positions for side shoulders
        for side shoulders, use vector between the 2 shoulders
        and make perpendicular to spine vector
        """
        Rshoulder_v = 17
        Lshoulder_v = 14
        shoulder_vector = pos[Rshoulder_v] - pos[Lshoulder_v]  # right to left shoulder
        spine_to_neck_vector = pos2[neck_u] - pos2[spine1_u]
        perp = np.cross(spine_to_neck_vector, shoulder_vector)
        shoulder_vector = np.cross(spine_to_neck_vector, -normalize(perp)) # perpendicular to unit perp vector
        shoulder_vector = normalize(shoulder_vector) # normalize to unit vector

        spine2_to_shoulder_mid = pos2[spine2_u] + (pos2[neck_u] - pos2[spine2_u]) * spine2_to_shoulder_vs_neck_ratio
        pos2[14] = spine2_to_shoulder_mid + tpose_shoulder_width * -shoulder_vector
        pos2[18] = spine2_to_shoulder_mid + tpose_shoulder_width * shoulder_vector

        """
        finally, solve for arms
        make sure shoulder to arm is correct length
        """
        R_neck_to_arm = pos[Rshoulder_v] - pos[neck_v]
        L_neck_to_arm = pos[Lshoulder_v] - pos[neck_v]
        R_neck_to_arm_length = dist(tpose_pos[19] - tpose_pos[neck_u])
        L_neck_to_arm_length = dist(tpose_pos[15] - tpose_pos[neck_u])
        pos2[19] = pos2[neck_u] + normalize(R_neck_to_arm) * R_neck_to_arm_length
        pos2[15] = pos2[neck_u] + normalize(L_neck_to_arm) * L_neck_to_arm_length
        Rv = pos2[19] - pos2[18]
        Lv = pos2[15] - pos2[14]
        # QUICK HACK: Use shoulder_to_arm vector and length as shortcut
        # Otherwise, the solution would involve solving the root of a quadratic equation
        pos2[19] = pos2[18] + normalize(Rv) * usc.bone_lengths[19]
        pos2[15] = pos2[14] + normalize(Lv) * usc.bone_lengths[15]
        for ix in [15, 19]:
            for jx in [1, 2]:
                ux = ix + jx
                vx = u_to_v_map[ux]
                local_t = pos[vx] - pos[vibe.parents[vx]]
                pos2[ux] = pos2[ux-1] + normalize(local_t) * usc.bone_lengths[ux]

        toeBase_joints = [4, 8] # right, left toeEnds
        usc.inverse_kinematics(pos2, ignore_joints=toeBase_joints)
        GT = usc.get_node_transforms()
        LR = usc.get_node_transforms(global_=False)[:,:3,:3]

        # # """
        # # Solve for ToeBase rotation, using ToeEnd positions
        # # """
        # for ux in toeBase_joints:
        #     vx = u_to_v_map[ux]
        #     direction = normalize(pos[vx] - pos[vibe.parents[vx]])
        #     toeLR = usc.solve_node_lookat_R(ux, direction)[0]
        #     LR[ux] = toeLR
        #     usc.set_node_local_pose_by_index(ux, LR[ux])

        """
        then calculate neck/head rotation based on the nose
        here, we will set as Neck rotation
        the neck will rotate on the x-axis (yz-axis plane) to face the nose
        """
        ux = neck_u  # neck
        GT_neck = GT[ux]
        nose_v = 12
        nose_direction = normalize(pos[nose_v] - pos[neck_v])

        # compute angle between the two points on yz plane
        # the angle is such that the z vector is pointing towards the direction
        R_axis = lookat_2D(GT_neck, nose_direction, axis_of_rotation=usc.forward_axis)
        R_axis = np.dot(R_axis, R_forward_z_to_default_neck_rotation)
        # apply the x-axis rotation on the neck local rotation
        neck_LR = np.dot(LR[ux], R_axis)
        if usc.USE_CONSTRAINTS:
            is_constrained, neck_LR = usc.constrain_joint_LR(ux, neck_LR)
        LR[ux] = neck_LR
        usc.set_node_local_pose_by_index(ux, LR[ux])  # second last upper_body link is Neck

        # ## DEBUG: VISUALIZE
        # sphere_radius = 1
        # sphere = open3d.geometry.TriangleMesh.create_sphere(sphere_radius)
        # M = np.eye(4)
        # M[:3,3] = pos2[ux] + normalize(nose_direction) * sphere_radius * 7
        # sphere.transform(M)
        # usc.visualize_transforms(extra_scene_objs=[sphere])

        return LR

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)

    poses_3d_file = "./output/data/processed_bruno_mars.npy"

    # LOAD DATA
    poses_3d = np.load(poses_3d_file)[:100]
    N, J = poses_3d.shape[:2]

    # rotate data to our desired coordinate frame
    R_YZ_180 = euler2mat_deg([0, 180, 180])
    poses_3d = np.dot(R_YZ_180, np.mat(poses_3d.reshape(N*J, 3)).T).T
    poses_3d = np.array(poses_3d).reshape(N, J, 3)

    # init converter
    converter = VIBE_Positions_To_Unity_Skeleton_Converter()
    usc = converter.usc
    vibe = converter.vibe
    total_usc_joints = len(usc.parents)

    # INIT ROTATION OUTPUTS
    all_local_rotations = np.zeros((N, total_usc_joints, 3, 3))

    # COMPUTE TRANSLATION (ROOT POSITION)
    root_j = usc.parents.index(-1)
    # scale root positions 
    vx = 2
    ROOT_MOTION_MULTIPLIER = 1.0 # 0.9
    scale = usc.bone_lengths[converter.v_to_u_map[vx]] / dist(np.mean(poses_3d[:,vx]-poses_3d[:,vibe.parents[vx]], axis=0))
    root_positions = poses_3d[:, root_j] * scale * ROOT_MOTION_MULTIPLIER
    # print(scale)

    for frame in range(0, N):
        pos = poses_3d[frame].copy()

        LR = converter.solve_LR(pos) # J, 3, 3

        all_local_rotations[frame] = LR

        print(frame)


    SAVE_CONSTRAINED_VERSION = 1
    TEST_UNCONSTRAINED = 0
    if TEST_UNCONSTRAINED:
        # # ======== UNCONSTRAINED: FOR COMPARISON PURPOSES ========
        all_local_rotations2 = all_local_rotations.copy()
        converter2 = VIBE_Positions_To_Unity_Skeleton_Converter(vibe, UnityAvatarSkeleton(use_constraints=False))
        for frame in range(0, N):
            pos = poses_3d[frame].copy()

            LR = converter2.solve_LR(pos)
            all_local_rotations2[frame] = LR

            print(frame)
        # # ======== UNCONSTRAINED: FOR COMPARISON PURPOSES ========
        visualize_skeleton_rotations(usc, np.stack((all_local_rotations, all_local_rotations2)), np.stack((root_positions, root_positions)))

        if not SAVE_CONSTRAINED_VERSION:
            all_local_rotations = all_local_rotations2
    else:
        visualize_skeleton_rotations(usc, all_local_rotations, root_positions)

    EXPORT = 0
    if EXPORT:
        FPS = 24.
        usc.reset_to_tpose() # put tpose as first frame
        LR = usc.local_rotations.copy()
        all_local_rotations = np.vstack(([LR], all_local_rotations))
        root_positions = np.vstack((root_positions[0], root_positions))

        frames = len(all_local_rotations)

        sc = usc
        data = {
            "parents": list(sc.parents),
            "joint_names": sc.joint_names,
            "seconds": [],
            "local_rotations": [],
            "local_translations": [],
            "global_translations": []
        }

        seconds = np.linspace(0., frames / FPS, frames)
        data["seconds"] = np.around(seconds, 3).tolist()
        for jx, LR in enumerate(all_local_rotations):
            euler_LR = np.array([m2ed(R) for R in LR])
            euler_LR = np.around(euler_LR, 4).tolist()
            sc.set_node_local_poses(LR)
            LT = sc.local_translations.copy()
            LT[0] = root_positions[jx]
            GT = sc.get_node_transforms()
            GT[:,:3,3] += LT[0]
            GT = np.around(GT[:,:3,3], 3).tolist()
            data["local_rotations"].append(euler_LR)
            data["local_translations"].append(np.around(LT, 4).tolist())
            data["global_translations"].append(GT)

        import json, os
        out_json = os.path.splitext(poses_3d_file)[0].split("/")[-1] + ".json"
        with open(out_json, "w") as f:
            json.dump(data, f)
            print("Saved to %s"%(out_json))
