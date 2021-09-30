import numpy as np
import open3d as o3d
from src.realsense import RealSenseManager
from src.camera_parameter import Intrinsic


close_flag = False


def close_callback(_vis):
    global close_flag
    close_flag = True


def cvt_rgbd_cv2o3d(color_image: np.ndarray, depth_image_aligned2color: np.ndarray):
    depth_image_o3d = o3d.geometry.Image(depth_image_aligned2color)
    color_image_o3d = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False)
    return rgbd


def main():
    rs_mng = RealSenseManager()  # default image size = (1280, 720)
    rs_mng.laser_turn_on()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("pcd viewer", width=1280, height=720)
    vis.register_key_callback(ord("Q"), close_callback)

    point_cloud = o3d.geometry.PointCloud()
    first_drawing = True
    while not close_flag:
        status = rs_mng.update()
        if not status:
            continue

        color_image = rs_mng.color_frame
        depth_image_aligned2color = rs_mng.depth_frame_aligned2color
        rgbd = cvt_rgbd_cv2o3d(color_image, depth_image_aligned2color)

        intrinsic_color: Intrinsic = rs_mng.intrinsic_color
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(*intrinsic_color.parameters)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        point_cloud.points = pcd.points
        point_cloud.colors = pcd.colors

        if first_drawing:
            vis.add_geometry(point_cloud)
            first_drawing = False
        else:
            vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()
    del vis
    del rs_mng


if __name__ == "__main__":
    main()