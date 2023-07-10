#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import shutil
import imgviz
import numpy as np
from tqdm import tqdm
import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

def parse_opt(known=False): # main으로 호출시 default 입력값.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='/home/jml/_workspace/yolov5_Drug_Segment/datasets/drug_V2_seg/images/', help="input annotated directory")
    parser.add_argument("--output_dir", default='result/',help="output dataset directory")
    parser.add_argument("--labels",default='labels.txt', help="labels file")
    parser.add_argument(
        "--noviz", default=True, help="no visualization", action="store_true"
    )
    # args = parser.parse_args()

    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    # Usage: import labelme2coco; labelme2coco.run(input_dir='path_of_jsonfiles', output_dir='path_of_result', labels='labels.txt', noviz=True)
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

def main(args):
    if osp.exists(args.output_dir): # 이미 결과폴더 있을시,
        print("Output directory already exists:", args.output_dir)
        shutil.rmtree(args.output_dir)

        # sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "Images"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))

    bar = tqdm(os.listdir(args.input_dir), total=len(os.listdir(args.input_dir)))
    for fn in bar:
        cnt = 0
        now = datetime.datetime.now()

        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        for i, line in enumerate(open(args.labels).readlines()):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            class_name_to_id[class_name] = class_id
            data["categories"].append(
                dict(
                    supercategory=None,
                    id=class_id,
                    name=class_name,
                )
            )

        label_files = glob.glob(str(osp.join(args.input_dir, fn, "*.json")))
        if 0 < len(label_files):
            out_ann_file = osp.join(args.output_dir, fn+".json")

            for image_id, filename in (enumerate(label_files)):
                cnt+=1
                bar.set_postfix_str(f"Creating dataset from : {fn}  \tstat: {cnt}/{len(label_files)}")
                # print("Generating dataset from:", filename)

                label_file = labelme.LabelFile(filename=filename)

                base = osp.splitext(osp.basename(filename))[0]
                out_img_file = osp.join(args.output_dir, "Images", base + ".jpg")

                img = labelme.utils.img_data_to_arr(label_file.imageData)
                # imgviz.io.imsave(out_img_file, img)
                data["images"].append(
                    dict(
                        license=0,
                        url=None,
                        file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                        height=img.shape[0],
                        width=img.shape[1],
                        date_captured=None,
                        id=image_id,
                    )
                )

                masks = {}  # for area
                segmentations = collections.defaultdict(list)  # for segmentation
                for shape in label_file.shapes:
                    points = shape["points"]
                    label = shape["label"]
                    group_id = shape.get("group_id")
                    shape_type = shape.get("shape_type", "polygon")
                    mask = labelme.utils.shape_to_mask(
                        img.shape[:2], points, shape_type
                    )

                    if group_id is None:
                        group_id = uuid.uuid1()

                    instance = (label, group_id)

                    if instance in masks:
                        masks[instance] = masks[instance] | mask
                    else:
                        masks[instance] = mask

                    if shape_type == "rectangle":
                        (x1, y1), (x2, y2) = points
                        x1, x2 = sorted([x1, x2])
                        y1, y2 = sorted([y1, y2])
                        points = [x1, y1, x2, y1, x2, y2, x1, y2]
                    if shape_type == "circle":
                        (x1, y1), (x2, y2) = points
                        r = np.linalg.norm([x2 - x1, y2 - y1])
                        # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                        # x: tolerance of the gap between the arc and the line segment
                        n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                        i = np.arange(n_points_circle)
                        x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                        y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                        points = np.stack((x, y), axis=1).flatten().tolist()
                    else:
                        points = np.asarray(points).flatten().tolist()

                    segmentations[instance].append(points)
                segmentations = dict(segmentations)

                for instance, mask in masks.items():
                    cls_name, group_id = instance
                    if cls_name not in class_name_to_id:
                        continue
                    cls_id = class_name_to_id[cls_name]

                    mask = np.asfortranarray(mask.astype(np.uint8))
                    mask = pycocotools.mask.encode(mask)
                    area = float(pycocotools.mask.area(mask))
                    bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                    data["annotations"].append(
                        dict(
                            id=len(data["annotations"]),
                            image_id=image_id,
                            category_id=cls_id,
                            segmentation=segmentations[instance],
                            area=area,
                            bbox=bbox,
                            iscrowd=0,
                        )
                    )

                if not args.noviz:
                    viz = img
                    if masks:
                        labels, captions, masks = zip(
                            *[
                                (class_name_to_id[cnm], cnm, msk)
                                for (cnm, gid), msk in masks.items()
                                if cnm in class_name_to_id
                            ]
                        )
                        viz = imgviz.instances2rgb(
                            image=img,
                            labels=labels,
                            masks=masks,
                            captions=captions,
                            font_size=15,
                            line_width=2,
                        )
                    out_viz_file = osp.join(
                        args.output_dir, "Visualization", base + ".jpg"
                    )
                    imgviz.io.imsave(out_viz_file, viz)
            with open(out_ann_file, "w") as f:
                json.dump(data, f)


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
