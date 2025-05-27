"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, debug_for_sampler=False):
    import torch
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        if debug_for_sampler:
            mask = gt_boxes[:, -2]
            original_boxes = gt_boxes[mask == 0]
            sampled_boxes = gt_boxes[mask == 1]
            vis = draw_box(vis, original_boxes, (0, 0, 1))
            vis = draw_box(vis, sampled_boxes, (0, 1, 0))
        else:
            vis = draw_box(vis, gt_boxes)
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 1), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()

def draw_scenes_new(points, gt_boxes=None, ref_boxes=None, ref_boxes_2=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, debug_for_sampler=False):
    import torch
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        if debug_for_sampler:
            mask = gt_boxes[:, -2]
            original_boxes = gt_boxes[mask == 0]
            sampled_boxes = gt_boxes[mask == 1]
            vis = draw_box(vis, original_boxes, (0, 0, 1))
            vis = draw_box(vis, sampled_boxes, (0, 1, 0))
        else:
            vis = draw_box(vis, gt_boxes)
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (1, 0, 0), ref_labels, ref_scores)

    if ref_boxes_2 is not None:
        vis = draw_box(vis, ref_boxes_2, (0, 0, 1), ref_labels, ref_scores)


    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def random_beam_range_jittering(points, std = 0.01):

    xyz = points[:, :3]
    # generate noise
    noise = np.random.normal(loc=0.0, scale=std, size=xyz.shape[0])
    # from xyz to range pitch yaw
    range = np.sqrt(xyz[:,0] ** 2 + xyz[:,1] ** 2 + xyz[:,2] ** 2)
    theta = np.arccos(xyz[:,2] / range) #
    phi = np.sign(xyz[:,1]) * np.arccos(xyz[:,0] / np.sqrt(xyz[:,0] ** 2 + xyz[:,1] ** 2)) #theta
    # add noise to range
    range += noise
    # from range pitch yaw to xyz
    x = range * np.sin(theta) * np.cos(phi)
    y = range * np.sin(theta) * np.sin(phi)
    z = range * np.cos(theta)
    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z

    return points

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Open3d visualization tool box')
    parser.add_argument('--pc_path', type=str, help='path to pcd file')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--gt_boxes_path', type=str, default=None, help='path to gt boxes file')
    parser.add_argument('--debug_for_sampler', action='store_true', default=False)
    args = parser.parse_args()

    pc_path = args.pc_path
    gt_boxes_path = args.gt_boxes_path

    if args.dataset == 'simulation':
        points = np.load(pc_path)['data']
        if gt_boxes_path is not None:
            gt_boxes = np.load(gt_boxes_path)
        else:
            gt_boxes = None
    if args.dataset == 'nuscenes':
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
        if gt_boxes_path is not None:
            gt_boxes = np.load(gt_boxes_path)
        else:
            gt_boxes = None
    if args.dataset == 'waymo':
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 6)
        if gt_boxes_path is not None:
            gt_boxes = np.load(gt_boxes_path)
        else:
            gt_boxes = None
    if args.dataset == 'nuscenes_debug':
        points = np.load(pc_path)
        if gt_boxes_path is not None:
            gt_boxes = np.load(gt_boxes_path)
        else:
            gt_boxes = None

    draw_scenes(random_beam_range_jittering(points), gt_boxes, debug_for_sampler=args.debug_for_sampler)


