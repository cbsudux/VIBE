import open3d
import copy


OPEN3D_VERSION_MAJOR, OPEN3D_VERSION_MINOR = open3d.__version__.split(".")[:2]

if int(OPEN3D_VERSION_MINOR) > 7:
    open3d.Vector3dVector = open3d.utility.Vector3dVector
    open3d.Vector3iVector = open3d.utility.Vector3iVector
    open3d.Vector2iVector = open3d.utility.Vector2iVector
    open3d.draw_geometries = open3d.visualization.draw_geometries
    open3d.create_mesh_coordinate_frame = open3d.geometry.TriangleMesh.create_coordinate_frame
    open3d.create_mesh_sphere = open3d.geometry.TriangleMesh.create_sphere
    open3d.create_mesh_box = open3d.geometry.TriangleMesh.create_box

PCI = open3d.camera.PinholeCameraIntrinsic
PCP = open3d.camera.PinholeCameraParameters

V3dV = open3d.Vector3dVector
V3iV = open3d.Vector3iVector
V2iV = open3d.Vector2iVector


def get_ctr_intrinsic(ctr):
    ic = get_ctr_intrinsics_class(ctr)
    intrinsics = ic.intrinsic_matrix
    return intrinsics

def get_vis_intrinsic(vis):
    ctr = vis.get_view_control()
    return get_ctr_intrinsic(ctr)

def get_ctr_extrinsic(ctr):
    cam_params = ctr.convert_to_pinhole_camera_parameters() # PinholeCameraParameters
    extrinsics = cam_params.extrinsic
    return extrinsics

def get_vis_extrinsic(vis):
    ctr = vis.get_view_control()
    return get_ctr_extrinsic(ctr)

def get_vis_window_size(vis):
    """
    :param vis: Open3D Visualizer instance
    :return: (Visualizer window height, Visualizer window width)
    """
    ctr = vis.get_view_control()
    ic = get_ctr_intrinsics_class(ctr)
    return (ic.height, ic.width)

def get_ctr_intrinsics_class(ctr):
    cam_params = ctr.convert_to_pinhole_camera_parameters() # PinholeCameraParameters
    return PCI(cam_params.intrinsic)

def set_vis_cam_params(vis, intr_mat=None, extr_mat=None):
    if intr_mat is None and extr_mat is None:
        return

    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params2 = copy.deepcopy(cam_params)
    if intr_mat is not None:
        assert intr_mat.shape == (3,3)
        cpi = cam_params.intrinsic
        W, H = (cpi.width, cpi.height)
        fx = intr_mat[0,0]
        fy = intr_mat[1,1]
        cx = intr_mat[0,2]
        cy = intr_mat[1,2]
        if cx == W / 2:
            cx -= 0.5
        if cy == H / 2:
            cy -= 0.5        
        target_intrinsics_class = PCI(W, H, fx, fy, cx, cy)
        cam_params2.intrinsic = target_intrinsics_class

    if extr_mat is not None:
        assert extr_mat.shape == (4,4)
        cam_params2.extrinsic = extr_mat

    ctr.convert_from_pinhole_camera_parameters(cam_params2)

