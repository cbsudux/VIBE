import numpy as np

from transforms3d.euler import mat2euler, euler2mat

inv = np.linalg.inv
norm = np.linalg.norm


R_y_90 = euler2mat(*np.deg2rad([0,90,0]))
R_x_90 = euler2mat(*np.deg2rad([90,0,0]))
R_x_n90 = euler2mat(*np.deg2rad([-90,0,0]))
R_x_180 = euler2mat(*np.deg2rad([0,180,0]))
R_z_90 = euler2mat(*np.deg2rad([0,0,90]))

def normalize(x):
    n = np.linalg.norm(x)
    if n == 0:
        return x*0
    return x / n

def lookAt(center, eye, up):
    """
    Taken from
    http://learnwebgl.brown37.net/07_cameras/camera_introduction.html
    """
    n = eye - center
    n = normalize(n)

    u = np.cross(up, n)
    u = normalize(u)

    v = np.cross(n, u)
    v = normalize(v)

    tx = - np.dot(u,eye)
    ty = - np.dot(v,eye)
    tz = - np.dot(n,eye)

    # Set the camera matrix
    M = np.eye(4)
    M[0, :3] = u
    M[1, :3] = v
    M[2, :3] = n
    M[:3, 3] = [tx, ty, tz]

    return M


def camera_lookAt(center, eye, up, forward_axis=0):
    """
    Make forward_axis face towards 'eye'
    """
    assert 0 <= forward_axis <= 2
    center = np.asarray(center)
    eye = np.asarray(eye)
    up = np.asarray(up)

    M = lookAt(center + 1e-6, eye + 1e-6, up + 1e-6)
    invM = inv(M)

    if forward_axis == 0:
        R = R_y_90
    elif forward_axis == 1:
        R = R_x_n90
    else:
        # R = np.dot(R_z_90, R_x_180)
        raise NotImplementedError()

    invM[:3,:3] = np.dot( invM[:3,:3], R )
    return invM


def lookAt_on_axis(M, pt_to, axis=0):
    """
    Locally rotates the transform on a single axis (x,y,or z) to face a given point (pt_to) in world space
    Outputs the final 4x4 transformation in world space

    M: 4x4 Transformation Matrix (row major) in world space
    pt_to: Translation Vector (size 3) in world space
    axis: x=0, y=1, z=2  (the axis of rotation in local space)
    """
    assert M.shape == (4,4)
    assert pt_to.size == 3
    assert 0 <= axis <= 2

    ax = np.zeros(3, dtype=np.float32)
    ax[axis] = 1
    other_axes = np.where(ax==0)[0]
    a1,a2 = other_axes

    # get the 2 orthogonal vectors to axis of rotation (forming our plane e.g. xy plane -> z axis of rotation)
    forward = M[:3,a1]
    right = M[:3,a2]

    pt_from = M[:3,3]  # get translation

    # compute vector between the two points
    dt = pt_from - pt_to

    # project to forward and right vectors
    dot_f_dt = np.dot(forward, dt)
    dot_r_dt = np.dot(right, dt)

    # compute angle between the two points on plane
    at = np.arctan2(dot_f_dt, dot_r_dt)

    # rotate on plane axis
    ax[axis] = at
    R_axis = euler2mat(*ax)
    R = np.dot(M[:3,:3], R_axis)  # first rotate locally on plane axis, then rotate to original rotation

    out = M.copy()
    out[:3,:3] = R
    return out


if __name__ == '__main__':
    import time
    import open3d

    np.set_printoptions(suppress=True, precision=4)

    # Local coordinate system for the camera:
    #   u maps to the x-axis
    #   v maps to the y-axis
    #   n maps to the z-axis

    M_eye = np.eye(4)
    eye_t = np.array([0., 0., 0])
    M_eye[:3,3] = eye_t

    coord = open3d.create_mesh_coordinate_frame(0.4)
    coord1 = open3d.create_mesh_coordinate_frame(0.3)
    coord2 = open3d.create_mesh_coordinate_frame(0.3)
    mesh_sphere = open3d.geometry.create_mesh_sphere(radius=0.1)
    # open3d.draw_geometries([coord, coord1, coord2, mesh_sphere])#, coord2, coord3])

    line_set = open3d.geometry.LineSet()
    line_set.lines = open3d.Vector2iVector([[0, 1]])
    line_set.points = open3d.Vector3dVector([eye_t, eye_t])

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(line_set)
    vis.add_geometry(coord)
    vis.add_geometry(coord1)
    # vis.add_geometry(coord2)
    vis.add_geometry(mesh_sphere)

    I4 = np.eye(4)
    R_y_90 = euler2mat(*np.deg2rad([0,90,0]))

    start_t = np.array([3.0, 2.0, 0.])
    start_M = np.eye(4)
    start_M[:3,3] = start_t

    max_rotation_theta = 1.0
    forward_axis = 0
    axis_of_rotation = (forward_axis + 1) % 3 # up axis
    # if forward_axis == 1:
    #     M_eye[:3,:3] = euler2mat(*np.radians([0,-90,-90]))
    # elif forward_axis == 2:
    #     M_eye[:3,:3] = np.dot(R_x_90, R_y_90)

    # coordx = open3d.create_mesh_coordinate_frame(0.3)
    # coordx.transform(M_eye)
    # open3d.draw_geometries([coord, coordx])
    for angle in range(-50, 50, 1):
        up = M_eye[:3,axis_of_rotation].copy()

        R = np.eye(4)
        R[:3,:3] = euler2mat(*np.deg2rad([angle,0,0]))

        target = np.dot(R, start_M)[:3,3]

        prev_M = M_eye.copy()

        M_target = np.eye(4)
        M_target[:3,3] = target
        M_eye = camera_lookAt(target, eye_t, up, forward_axis=forward_axis)
        apply_R = np.dot(M_eye, inv(prev_M))
        # eu_angles = mat2euler(apply_R[:3,:3])
        # print(apply_R)
        # print(np.rad2deg(eu_angles))

        if not np.all(np.abs(apply_R - I4) < 1e-4):
            line_set.points = open3d.Vector3dVector([eye_t, target])
            # line_set.colors = open3d.Vector3dVector(colors)
            coord1.transform(apply_R)
            mesh_sphere.transform(M_target)

            vis.update_geometry()

        vis.poll_events()
        vis.update_renderer()

        time.sleep(0.1)

    vis.destroy_window()
