import os
import os.path as osp

from typing import Union, List

COLOR_PALETTE = [
    (0, 0, 0),  # Black
    (255, 255, 255),  # White
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 128, 128),  # Gray
    (192, 192, 192),  # Silver
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Green (Dark)
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 64, 128),  # Blue (Dark)
    (64, 0, 128),  # Purple (Dark)
    (128, 64, 0),  # Brown
    (0, 128, 64),  # Green (Light)
    (0, 64, 0),  # Green (Dark)
    (64, 0, 0),  # Red (Dark)
    (255, 128, 128),  # Light Red
    (255, 128, 0),  # Orange
    (255, 255, 128),  # Light Yellow
    (128, 255, 128),  # Light Green
    (128, 255, 255),  # Light Cyan
    (128, 128, 255),  # Light Blue
    (255, 128, 255),  # Light Purple
    (192, 128, 64),  # Tan
    (255, 192, 128),  # Peach
    (255, 192, 192),  # Light Pink
    (128, 128, 64),  # Olive (Dark)
    (128, 64, 64),  # Brown (Dark)
    (128, 64, 128),  # Purple (Medium)
    (128, 0, 64),  # Maroon (Dark)
    (64, 128, 0),  # Olive (Light)
    (192, 64, 0),  # Orange (Dark)
    (192, 192, 0),  # Yellow (Dark)
    (192, 192, 128),  # Khaki
    (192, 192, 64),  # Olive (Medium)
    (128, 192, 64),  # Olive (Medium-Dark)
    (192, 128, 128),  # Rose
    (64, 192, 128),  # Green (Medium-Light)
    (64, 192, 192),  # Teal (Medium)
    (64, 64, 192),  # Blue (Medium)
    (192, 64, 192),  # Purple (Medium-Light)
    (192, 192, 192),  # Light Gray
    (0, 0, 64),  # Blue (Dark)
    (0, 64, 64),  # Teal (Dark)
    (0, 0, 128),  # Blue (Dark)
    (0, 128, 128),  # Teal (Medium)
    (0, 64, 192),  # Blue (Medium-Light)
    (64, 0, 64),  # Purple (Dark)
    (64, 64, 0),  # Olive (Dark)
    (64, 0, 0),  # Maroon (Dark)
    (192, 0, 0),  # Red (Medium-Dark)
    (192, 0, 192),  # Purple (Medium-Dark)
    (192, 128, 0),  # Brown (Medium)
    (128, 128, 192),  # Blue (Medium-Light)
    (128, 0, 192),  # Purple (Medium-Dark)
    (192, 0, 128),  # Magenta (Dark)
    (128, 0, 64),  # Maroon (Medium-Dark)
    (128, 64, 192),  # Purple (Medium-Light)
    (64, 128, 192),  # Blue (Medium-Light)
    (192, 128, 192),  # Pink (Light)
    (192, 0, 64),  # Red (Medium-Dark)
    (192, 64, 128),  # Pink (Medium)
    (64, 192, 0),  # Green (Medium)
    (128, 192, 128),  # Green (Medium-Light)
    (128, 192, 192),  # Cyan (Medium)
    (192, 192, 128),  # Green (Medium-Light)
    (192, 192, 0),  # Yellow (Medium)
    (0, 192, 64),  # Green (Medium-Light)
    (0, 192, 192),  # Teal (Medium)
    (64, 192, 192),  # Teal (Medium-Light)
    (128, 128, 64),  # Olive (Medium-Dark)
    (0, 128, 192),  # Blue (Medium-Light)
    (192, 128, 64),  # Brown (Medium)
    (192, 128, 0),  # Brown (Medium)
    (128, 128, 0),  # Olive (Medium)
    (64, 128, 64),  # Olive (Medium)
    (192, 64, 64),  # Red (Medium-Dark)
    (0, 128, 64),  # Green (Medium)
    (64, 192, 0),  # Green (Medium)
    (128, 64, 64),  # Red (Medium-Dark)
    (64, 64, 192),  # Blue (Medium-Light)
    (128, 0, 192),  # Purple (Medium-Dark)
    (64, 192, 128),  # Teal (Medium-Light)
    (128, 192, 192),  # Cyan (Medium-Light)
]


def assert_file(file_name: str):
    assert osp.exists(file_name), f"{file_name} not exists"


class FolderPath:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        assert_file(root_dir)
        self.file_index = self.__create_file_index()

    def __create_file_index(self):
        file_index = {}
        for root, _, files in os.walk(self.root_dir):
            if files:
                for file in files:
                    file_full_name = osp.join(root, file)
                    file_base_name = osp.basename(file)
                    assert_file(file_full_name)
                    file_index[file_base_name] = file_full_name
        return file_index

    def __getitem__(self, file_base_name: str):
        return self.file_index[file_base_name]


class BoundingBox:
    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        w: Union[int, float],
        h: Union[int, float],
    ):
        self.__x1 = x1
        self.__y1 = y1
        self.__w = w
        self.__h = h
        self.__x2 = self.x1 + w
        self.__y2 = self.y1 + h
        self.__area = w * h

    def __repr__(self):
        return f"xyxy : [{self.x1:.2f}, {self.y1:.2f}, {self.x2:.2f}, {self.y2:.2f}]"

    @property
    def x1(self) -> Union[int, float]:
        return self.__x1

    @property
    def x2(self) -> Union[int, float]:
        return self.__x2

    @property
    def y1(self) -> Union[int, float]:
        return self.__y1

    @property
    def y2(self) -> Union[int, float]:
        return self.__y2

    @property
    def w(self) -> Union[int, float]:
        return self.__w

    @property
    def h(self) -> Union[int, float]:
        return self.__h

    @property
    def area(self) -> Union[int, float]:
        return self.__area

    @property
    def xywh(self) -> List[Union[int, float]]:
        return [self.x1, self.y1, self.w, self.h]

    @property
    def xyxy(self) -> List[Union[int, float]]:
        return [self.x1, self.y1, self.x2, self.y2]

    def get_iou(self, other) -> float:
        _intersection_x1 = max(self.x1, other.x1)
        _intersection_y1 = max(self.y1, other.y1)
        _intersection_x2 = min(self.x2, other.x2)
        _intersection_y2 = min(self.y2, other.y2)

        _intersection = max(_intersection_x2 - _intersection_x1, 0) * max(
            _intersection_y2 - _intersection_y1, 0
        )
        _union = (self.area + other.area) - _intersection
        iou = _intersection / _union
        return iou
