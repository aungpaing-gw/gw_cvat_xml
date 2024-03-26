from typing import Union, List, Dict, Any
from collections import defaultdict


import numpy as np
import cv2

from cvatxml.utils import assert_file, COLOR_PALETTE, BoundingBox, FolderPath
from cvatxml.cvatxml import CVATXML


class BaseVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def draw_transparent_bg(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Union[np.ndarray, List[int]] = [0, 255, 0],
    ):
        if isinstance(color, list):
            color = np.array(color)
        cropped = img[y1:y2, x1:x2]
        cropped = (cropped * 0.6) + (color * 0.4)
        cropped = cropped.astype(np.uint8)
        img[y1:y2, x1:x2] = cropped
        return img

    @staticmethod
    def draw_dotted_line(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: List[int],
        line_thickness: int,
    ):
        dot_space = int(line_thickness * 1.2)
        dot_ratio = int(line_thickness * 1.5)

        h, w = y2 - y1, x2 - x1

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, [0, 0], [w, h], 255, line_thickness * 2)

        # Crop center
        x_s = line_thickness
        y_s = line_thickness
        x_e = w - line_thickness
        y_e = h - line_thickness

        # Mask Lines
        x_l = np.arange(x_s, x_e, dot_space)
        y_l = np.arange(y_s, y_e, dot_space)
        for i in range(len(x_l) // dot_ratio):
            mask[:, x_l[i * dot_ratio] : x_l[i * dot_ratio + 1]] = 0
        for i in range(len(y_l) // dot_ratio):
            mask[y_l[i * dot_ratio] : y_l[i * dot_ratio + 1], :] = 0

        cropped = img[y1:y2, x1:x2]
        cropped[mask == 255] = color
        return img


class XMLVisualizer(BaseVisualizer):
    def __init__(
        self,
        cxml: CVATXML,
        img_dir: str,
        vis_bbox: bool = True,
        vis_txt: bool = True,
        vis_meta: bool = False,
        txt_size: float = 0.5,
        txt_thickness: int = 1,
        vis_txt_bg_color: bool = True,
        txt_above_bbox: bool = True,
        vis_txt_attributes: List[str] = [],
        COLOR_PALETTE: List[List[int]] = COLOR_PALETTE,
    ):
        super().__init__()
        self.cxml = cxml
        self.img_dir = img_dir
        self.folder_path = FolderPath(img_dir)
        self.vis_bbox = vis_bbox
        self.vis_txt = vis_txt
        self.vis_meta = vis_meta
        self.txt_size = txt_size
        self.txt_thickness = txt_thickness
        self.vis_txt_bg_color = vis_txt_bg_color
        self.txt_above_bbox = txt_above_bbox
        self.vis_txt_attributes = vis_txt_attributes
        self.COLOR_PALETTE = COLOR_PALETTE

    def __vis_txt(self, img_arr, txt, x1, y1, color):
        txt_color = (0, 0, 0) if np.mean(color) > 122 else (255, 255, 255)
        __text_size = cv2.getTextSize(
            txt, self.font, self.txt_size, self.txt_thickness
        )[0]
        if self.txt_above_bbox:
            x1, y1 = x1, y1 - 2
            bbox_x1, bbox_y1 = x1, y1 - int(1.4 * __text_size[1])
            bbox_x2, bbox_y2 = x1 + __text_size[0] + 1, y1 - 1
        else:
            bbox_x1, bbox_y1 = x1, y1 + 1
            bbox_x2, bbox_y2 = x1 + __text_size[0] + 1, y1 + int(1.4 * __text_size[1])
            x1, y1 = x1, y1 + __text_size[1]
        if self.vis_txt_bg_color:
            cv2.rectangle(
                img_arr, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color.tolist(), -1
            )

        cv2.putText(
            img_arr,
            txt,
            (x1, y1),
            self.font,
            self.txt_size,
            txt_color,
            self.txt_thickness,
        )
        return img_arr

    def __get_text(self, anno: Dict[str, Any]):
        ret = ""
        for attr in self.vis_txt_attributes:
            if anno["attributes"].get(attr):
                ret += anno["attributes"].get(attr)
        return ret

    def _vis_bbox(
        self, img_arr: np.ndarray, bbox: BoundingBox, catId: int, txt_append: str = ""
    ):
        _color = self.COLOR_PALETTE[catId]
        _color_np = np.array(_color)

        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        cv2.rectangle(img_arr, [x1, y1], [x2, y2], _color, 2)

        text = self.cxml.cats[catId]["name"]
        if txt_append != "":
            text += txt_append
        if text != "":
            img_arr = self.__vis_txt(img_arr, text, x1, y1, _color_np)
        return img_arr

    def _vis_meta(self, img_arr, annIds):
        sample_counts = defaultdict(int)

        for annId in annIds:
            anno = self.cxml.loadAnns(annIds=annId)[0]
            catId = anno["category_id"]
            sample_counts[catId] += 1

        n_rows = len(sample_counts)
        txt_size = cv2.getTextSize(
            "test", self.font, self.txt_size, self.txt_thickness
        )[0]
        bbox_x1 = 0
        bbox_y1 = 0
        bbox_y2 = n_rows * txt_size[1] + 40
        bbox_x2 = 6 * txt_size[0]
        img_arr = self.draw_transparent_bg(
            img_arr, bbox_x1, bbox_y1, bbox_x2, bbox_y2, [255, 255, 0]
        )
        x, y = 10, 20
        for catId, counts in sample_counts.items():
            catName = self.cxml.cats[catId]["name"]
            txt = f"{catName:<10}: {counts}"
            cv2.putText(
                img_arr,
                txt,
                [x, y],
                self.font,
                self.txt_size,
                [0, 0, 0],
                self.txt_thickness,
            )
            y += txt_size[1] + 10

        return img_arr

    def vis(self, imgId):
        annIds = self.cxml.getAnnIds(imgIds=imgId)
        img = self.cxml.loadImgs(imgIds=imgId)[0]
        img_base_name = img["file_name"]
        img_full_name = self.folder_path[img_base_name]
        assert_file(img_full_name)
        img_arr = cv2.imdecode(
            np.fromfile(img_full_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if self.vis_meta:
            img_arr = self._vis_meta(img_arr, annIds)

        for annId in annIds:
            anno = self.cxml.loadAnns(annIds=annId)[0]
            bbox = BoundingBox(*anno["bbox"])
            catId = anno["category_id"]
            txt_append = self.__get_text(anno)

            if self.vis_bbox:
                img_arr = self._vis_bbox(img_arr, bbox, catId, txt_append)

        return img_arr
