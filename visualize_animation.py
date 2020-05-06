import argparse
import numpy as np
import time

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


def get_parent_lineset():
    # lines = []
    # for jx in range(len(parents)):
    #     p = parents[jx]
    #     if p > -1:
    #         lines.append([p, jx])

    line_set = open3d.geometry.LineSet()
    line_set.lines = V2iV(connections)
    return line_set

def get_coords_n_lines(matrix_array, coord_frame_scale=0.3):
    line_set = get_parent_lineset()
    line_set.points = V3dV(matrix_array)
    coords = []
    for M in matrix_array:
        coord = open3d.create_mesh_coordinate_frame(coord_frame_scale)
        # coord.transform(M)
        coords.append(coord)
    return [coords, line_set]

# def get_bone_lengths(positions, parents):
#     assert len(positions) == len(parents)

#     bone_lens = np.empty(len(positions), dtype=np.float32)
#     for ix, pos in enumerate(positions):
#         p = parents[ix]
#         bone_lens[ix] = dist(pos - positions[p]) if p >= 0 else dist(pos)
#     return bone_lens


if __name__ == '__main__':
    import os, os.path as osp
    np.set_printoptions(suppress=True, precision=4)

    data_dir = "output/data"
    animation_file = osp.join(data_dir, "thatswhatilike_2_poses.npy")

    animation = np.load(animation_file)
    # import pdb; pdb.set_trace()
    total_frames = animation.shape[0]

    w,h,d = 10,0.1,10
    mesh_box = open3d.geometry.TriangleMesh.create_box(w,h,d)
    mesh_box.translate([-w/2.,-h,-d/2.])
    origin_coord = open3d.create_mesh_coordinate_frame(0.5)

    out_src = get_coords_n_lines(animation[0], 0.3)  # why animation[0] only?
    lineset = out_src[1]
    coords = out_src[0]

    # FOR SINGLE FRAME VISUALIZATION, USE draw_geometries
    print("Press Q to quit")
    open3d.draw_geometries(coords)

    # FOR SEQUENCE OF 3D POSES, Better to use open3d Visualizer
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    for geometry in [lineset, origin_coord] + coords:# + dst_coords:
        vis.add_geometry(geometry)

    global FRAME, NEXT, QUIT
    QUIT = 0
    FRAME = 0
    NEXT = 0
    def next_callback(vis):
        global NEXT, FRAME
        NEXT = 1
        FRAME += 1
        if FRAME >= total_frames:
            FRAME = 0
    def prev_callback(vis):
        global NEXT, FRAME
        NEXT = 1
        FRAME -= 1
        if FRAME < 0:
            FRAME = total_frames - 1

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

    prev_M = animation[0].copy()

    while 1:
        vis.poll_events()
        time.sleep(0.005)
        if QUIT:
            break
        if NEXT:
            NEXT = 0
            frame = FRAME
            print("FRAME: %d"%frame)
            anim_frame = animation[frame]

            lineset.points = V3dV(anim_frame)
            # lineset.points = V3dV(anim_frame[:, :3, 3])
            
            # for jx, coord in enumerate(coords):
                # apply_M = np.dot(anim_frame[jx], 
                #     np.linalg.inv(prev_M[jx]))
                # coord.transform(apply_M)

            prev_M = anim_frame

            vis.update_geometry()
            vis.update_renderer()

    vis.destroy_window()
