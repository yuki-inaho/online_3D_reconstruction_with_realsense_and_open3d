import numpy as np


class Intrinsic:
    def __init__(self):
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

    def set_image_size(self, image_width, image_height):
        self._image_width = image_width
        self._image_height = image_height

    def set_intrinsic_parameter(self, fx, fy, cx, cy):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

    @property
    def parameters(self):
        return [self._image_width, self._image_height, self._fx, self._fy, self._cx, self._cy]

    @property
    def K(self):
        return np.array([[self._fx, 0.0, self._cx], [0.0, self._fy, self._cy], [0.0, 0.0, 1.0]])

    @property
    def center(self):
        return self._cx, self._cy

    @property
    def focal(self):
        return self._fx, self._fy
