import os
import os.path as osp
import numpy as np
import cv2
from bs4 import BeautifulSoup as bs
from collections import defaultdict
from typing import Union, List, Dict, Optional


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
