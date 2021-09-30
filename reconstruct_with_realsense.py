import cv2
import click
import numpy as np
import open3d as o3d
from src.realsense import RealSenseManager
from src.camera_parameter import Intrinsic
from typing import NamedTuple


class Config(NamedTuple):
    depth_max: float
    voxel_size: float
    sdf_trunc: float


close_flag = False


def close_callback(_vis):
    global close_flag
    close_flag = True


def cvt_rgbd_cv2o3d(color_image: np.ndarray, depth_image_aligned2color: np.ndarray):
    depth_image_o3d = o3d.geometry.Image(depth_image_aligned2color)
    color_image_o3d = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False)
    return rgbd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999),
    )
    return result


def refine_registration(source_down, target_down, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down,
        target_down,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30),
    )
    return result_icp


@click.command()
@click.option("-d", "--depth-max", type=float, default=1.0)
@click.option("-v", "--voxel-size", type=float, default=0.005)
@click.option("-s", "--sdf-trunc", type=float, default=0.04)
def main(depth_max, voxel_size, sdf_trunc):
    config = Config(depth_max, voxel_size, sdf_trunc)
    rs_mng = RealSenseManager()  # default image size = (1280, 720)
    rs_mng.laser_turn_on()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("pcd viewer", width=1280, height=720)
    vis.register_key_callback(ord("Q"), close_callback)

    intrinsic_color: Intrinsic = rs_mng.intrinsic_color
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(*intrinsic_color.parameters)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_size,
        sdf_trunc=config.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    count = 0
    first_drawing = True
    source_down = None
    source_fpfh = None

    metadata = [0, 0, 0]
    transform_prev = np.identity(4)
    mesh_drawing = o3d.geometry.TriangleMesh()
    while not close_flag:
        status = rs_mng.update()
        if not status:
            continue

        color_image = cv2.cvtColor(rs_mng.color_frame, cv2.COLOR_BGR2RGB)
        depth_image_aligned2color = rs_mng.depth_frame_aligned2color
        depth_image_aligned2color[depth_image_aligned2color > int(config.depth_max * 1000)] = 0

        rgbd = cvt_rgbd_cv2o3d(color_image, depth_image_aligned2color)
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        if count == 0:
            source_down, source_fpfh = preprocess_point_cloud(point_cloud, config.voxel_size)
            count += 1
            continue

        target_down, target_fpfh = preprocess_point_cloud(point_cloud, config.voxel_size)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, config.voxel_size)
        result_icp = refine_registration(source_down, target_down, result_ransac, config.voxel_size)

        transform = result_icp.transformation
        transform = np.dot(transform, transform_prev)
        transform_prev = transform

        source_down = target_down
        source_fpfh = target_fpfh

        volume.integrate(rgbd, pinhole_camera_intrinsic, transform)
        mesh = volume.extract_triangle_mesh()
        mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # flip!
        mesh.compute_vertex_normals()
        mesh_drawing.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
        mesh_drawing.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors).copy())
        mesh_drawing.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles).copy())
        mesh_drawing.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals).copy())
        count += 1

        if first_drawing:
            vis.add_geometry(mesh_drawing)
            first_drawing = False
        else:
            vis.update_geometry(mesh_drawing)

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
    del vis
    del rs_mng


if __name__ == "__main__":
    main()