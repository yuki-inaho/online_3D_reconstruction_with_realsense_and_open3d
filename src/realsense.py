import pyrealsense2 as rs
import numpy as np
from threading import RLock
from src.camera_parameter import Intrinsic


class RealSenseManager:
    def __init__(self, image_width=1280, image_height=720):
        self._image_width = image_width
        self._image_height = image_height

        self._setting()
        self._set_intrinsic_parameters()
        self._lock: RLock = RLock()
        self._ir_frame_left: np.ndarray = np.array([])
        self._ir_frame_right: np.ndarray = np.array([])
        self._color_frame: np.ndarray = np.array([])
        self._depth_frame: np.ndarray = np.array([])
        self._depth_frame_aligned2color: np.ndarray = np.array([])

    def __del__(self):
        self._pipeline.stop()
        self._device.hardware_reset()

    def _setting(self):
        self._pipeline = rs.pipeline()

        self._config = rs.config()
        self._config.enable_stream(rs.stream.infrared, 1, self._image_width, self._image_height, rs.format.y8, 30)
        self._config.enable_stream(rs.stream.infrared, 2, self._image_width, self._image_height, rs.format.y8, 30)
        self._config.enable_stream(rs.stream.depth, self._image_width, self._image_height, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.color, self._image_width, self._image_height, rs.format.bgr8, 30)

        self._profile = self._pipeline.start(self._config)
        self._device = self._profile.get_device()

        depth_sensor = self._device.first_depth_sensor()
        # preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset,3)  #'Default':1, 'High Accuracy': 3
        depth_sensor.set_option(rs.option.visual_preset, 1)

        align_to = rs.stream.color
        self._align = rs.align(align_to)

    def _set_intrinsic_parameters(self):
        profile_depth = self._profile.get_stream(rs.stream.depth)

        profile_color = self._profile.get_stream(rs.stream.color)
        depth_profile = profile_depth.as_video_stream_profile()
        color_profile = profile_color.as_video_stream_profile()
        color_intrinsic_rs = color_profile.get_intrinsics()
        depth_intrinsic_rs = depth_profile.get_intrinsics()

        profile_left = self._profile.get_stream(rs.stream.infrared, 1)
        profile_right = self._profile.get_stream(rs.stream.infrared, 2)
        ir_left_profile = profile_left.as_video_stream_profile()
        ir_right_profile = profile_right.as_video_stream_profile()
        ir_left_intrinsic_rs = ir_left_profile.get_intrinsics()
        ir_right_intrinsic_rs = ir_right_profile.get_intrinsics()

        self._left_to_right_extrinsic = ir_left_profile.get_extrinsics_to(ir_right_profile)
        self._left_to_color_extrinsic = ir_left_profile.get_extrinsics_to(color_profile)

        self._depth_intrinsic = self._cvt_intrinsics(depth_intrinsic_rs)
        self._ir_left_intrinsic = self._cvt_intrinsics(ir_left_intrinsic_rs)
        self._ir_right_intrinsic = self._cvt_intrinsics(ir_right_intrinsic_rs)
        self._color_intrinsic = self._cvt_intrinsics(color_intrinsic_rs)

    def _cvt_intrinsics(self, rs_intrinsics: rs.pyrealsense2.intrinsics):
        intrinsic = Intrinsic()
        intrinsic.set_image_size(self._image_width, self._image_height)
        intrinsic.set_intrinsic_parameter(rs_intrinsics.fx, rs_intrinsics.fy, rs_intrinsics.ppx, rs_intrinsics.ppy)
        return intrinsic

    def _cvt_ndarray(self, frame: rs.pyrealsense2.video_frame):
        if frame is not None:
            return np.asanyarray(frame.get_data())
        else:
            return None

    def update(self):
        with self._lock:
            frames = self._pipeline.wait_for_frames()
            self._ir_frame_left = self._cvt_ndarray(frames.get_infrared_frame(1))
            self._ir_frame_right = self._cvt_ndarray(frames.get_infrared_frame(2))
            self._depth_frame = self._cvt_ndarray(frames.get_depth_frame())
            self._color_frame = self._cvt_ndarray(frames.get_color_frame())

            aligned_frames = self._align.process(frames)
            self._depth_frame_aligned2color = self._cvt_ndarray(aligned_frames.get_depth_frame())

            if (self._depth_frame is not None) and (self._color_frame is not None):
                return True
            else:
                return False

    def laser_turn_off(self):
        device = self._profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 0)

    def laser_turn_on(self):
        device = self._profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 150)

    @property
    def intrinsic_depth(self):
        return self._depth_intrinsic

    @property
    def intrinsic_color(self):
        return self._color_intrinsic

    @property
    def intrinsic_ir_left(self):
        return self._ir_left_intrinsic

    @property
    def intrinsic_ir_right(self):
        return self._ir_right_intrinsic

    @property
    def color_frame(self):
        return self._color_frame

    @property
    def ir_frame(self):
        return self._ir_frame_left

    @property
    def ir_frame_left(self):
        return self._ir_frame_left

    @property
    def ir_frame_right(self):
        return self._ir_frame_right

    @property
    def depth_frame(self):
        return self._depth_frame

    @property
    def depth_frame_aligned2color(self):
        return self._depth_frame_aligned2color

    @property
    def K_color(self):
        return self._color_intrinsic.K

    @property
    def P_color(self):
        P = np.zeros([3, 4])
        P[:3, :3] = self._color_intrinsic.K
        return P

    @property
    def K_depth(self):
        return self._depth_intrinsic.K

    @property
    def image_size(self):
        return (self._image_width, self._image_height)

    @property
    def rotation_dcm_ir_left2right(self):
        return np.asarray(self._left_to_right_extrinsic.rotation).reshape(3, 3)

    @property
    def translation_ir_left2right(self):
        return np.asarray(self._left_to_right_extrinsic.translation)

    @property
    def rotation_dcm_ir_left2color(self):
        return np.asarray(self._left_to_color_extrinsic.rotation).reshape(3, 3)

    @property
    def translation_ir_left2color(self):
        return np.asarray(self._left_to_color_extrinsic.translation)