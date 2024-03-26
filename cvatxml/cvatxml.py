import os
import os.path as osp
import numpy as np
import cv2
from bs4 import BeautifulSoup as bs
from collections import defaultdict
from typing import Union, List, Dict, Optional


def assert_file(file_name: str):
    assert osp.exists(file_name), f"{file_name} not exists"


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


class CVATXML:
    def __init__(self, anno_file_name: str):
        self.__anno_file_name = anno_file_name
        self.data = self.read_xml()
        self.cats, self.catNmId = self.__get_cats()
        self.imgs = self.__get_imgs()
        self.annos = self.__get_annos()
        self.catId_imgId = self.__get_cat_img_pair()
        self.imgId_annoId = self.__get_img_anno_pair()

    def __repr__(self):
        _str = f"CVAT format XML annotation file\n"
        _str += f"Number of categories \t : {len(self.cats)}\n"
        _str += f"Number of images \t : {len(self.imgs)}\n"
        _str += f"Number of annotations \t : {len(self.annos)}\n"
        return _str

    def read_xml(self):
        return bs(
            open(self.__anno_file_name, "r", encoding="utf-8").read(), features="xml"
        )

    def __get_cats(self):
        labels = self.data.find_all("label")
        catNmId = {}
        catIdNm = {}
        for i, label in enumerate(labels):
            catNm = label.find("name").text
            attributes = label.find_all("attribute")
            _attr = {}
            for attribute in attributes:
                _attr[attribute.find("name").text] = attribute.find("values")

            catNmId[catNm] = i
            catIdNm[i] = {"id": i, "name": catNm, "attributes": _attr}
        return catIdNm, catNmId

    def __get_imgs(self):
        images = self.data.find_all("image")
        imgIdDict = {}
        for image in images:
            img_attrs = image.attrs
            imgId = int(img_attrs["id"])
            imgName = img_attrs["name"]
            width = int(img_attrs["width"])
            height = int(img_attrs["height"])
            imgIdDict[imgId] = {
                "id": imgId,
                "file_name": imgName,
                "width": width,
                "height": height,
            }
        return imgIdDict

    def __get_annos(self):
        images = self.data.find_all("image")
        annos = {}
        annoId = 0
        for image in images:
            imgId = int(image.attrs["id"])
            bboxes = image.find_all("box")
            for bbox in bboxes:
                bbox_attrs = bbox.attrs
                clsName = bbox_attrs["label"]
                catId = self.catNmId[clsName]
                occluded = bbox_attrs["occluded"]
                x1 = bbox_attrs["xtl"]
                y1 = bbox_attrs["ytl"]
                x2 = bbox_attrs["xbr"]
                y2 = bbox_attrs["ybr"]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w = x2 - x1
                h = y2 - y1
                bbox_attributes = bbox.find_all("attribute")
                attr_dict = {}
                for attribute in bbox_attributes:
                    attr_dict[attribute.attrs["name"]] = attribute.text

                annos[annoId] = {
                    "id": annoId,
                    "image_id": imgId,
                    "bbox": [x1, y1, w, h],
                    "category_id": catId,
                    "occluded": occluded,
                    "attributes": attr_dict,
                }
                annoId += 1
        return annos

    def __get_img_anno_pair(self) -> Dict[int, List[int]]:
        ret = defaultdict(list)
        for annoId, anno in self.annos.items():
            ret[anno["image_id"]].append(annoId)
        return ret

    def __get_cat_img_pair(self) -> Dict[int, List[int]]:
        ret = defaultdict(list)
        for annoId, anno in self.annos.items():
            ret[anno["category_id"]].append(anno["image_id"])
        return ret

    def getImgIds(self, catIds: Optional[int] = None) -> List[int]:
        imgIds = []
        if catIds is None:
            imgIds = [imgId for imgId, _ in self.imgs.items()]
        else:
            imgIds = self.catId_imgId[catIds]
        return imgIds

    def getAnnIds(self, imgIds: Union[List[int], int]) -> List[int]:
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        return [annoId for imgId in imgIds for annoId in self.imgId_annoId[imgId]]

    def loadImgs(self, imgIds: Union[List[int], int]):
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        return [self.imgs[imgId] for imgId in imgIds]

    def loadAnns(self, annIds: Union[List[int], int]):
        annIds = annIds if isinstance(annIds, list) else [annIds]
        return [self.annos[annoId] for annoId in annIds]


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


class BaseVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_transparent_bg(
        self,
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

    def draw_dotted_line(
        self,
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
            x1, y1 = x1, y1 + __text_size[1]
            bbox_x1, bbox_y1 = x1, y1 + 1
            bbox_x2, bbox_y2 = x1 + __text_size[0] + 1, y1 + int(1.4 * __text_size[1])
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

    def __get_text(self):
        return ""

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
            txt_append = self.__get_text()

            if self.vis_bbox:
                img_arr = self._vis_bbox(img_arr, bbox, catId, txt_append)

        return img_arr
