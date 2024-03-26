import os
import os.path as osp
import cv2


from cvatxml.cvatxml import CVATXML
from cvatxml.cxml_vis import XMLVisualizer


def main():
    xml_file = "/home/aung/Downloads/child_birth/annotations.xml"
    img_dir = "/home/aung/Downloads/child_birth/images"
    vis_dir = "/home/aung/Downloads/child_birth/vis"
    os.makedirs(vis_dir, exist_ok=True)

    vis_bbox = True
    vis_txt = True
    vis_meta = True
    txt_size = 0.5
    txt_thickness = 1
    txt_bg_color = True
    txt_above_bbox = False
    vis_txt_attributes = []

    cxml = CVATXML(xml_file)
    visualizer = XMLVisualizer(
        cxml,
        img_dir,
        vis_bbox,
        vis_txt,
        vis_meta,
        txt_size,
        txt_thickness,
        txt_bg_color,
        txt_above_bbox,
        vis_txt_attributes,
    )

    imgIds = cxml.getImgIds()
    for imgId in imgIds:
        img_arr = visualizer.vis(imgId)

        dst_file = osp.join(vis_dir, cxml.loadImgs(imgId)[0]["file_name"])
        print(dst_file)
        cv2.imwrite(dst_file, img_arr)
        break


if __name__ == "__main__":
    main()
