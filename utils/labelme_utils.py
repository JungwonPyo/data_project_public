import argparse
import base64
import json
import os
import io
import glob
import os.path as osp
import numpy as np
import cv2
import math
import uuid
from pathlib import Path
from tqdm import tqdm

import imgviz
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

def get_json_info(json_path):

    data = json.load(open(json_path))
    
    return data

def get_all_json_lists(base_path):
    return glob.glob(os.path.join(base_path,'*.json'))


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def get_image_from_json(base_path, json_data):

    imageData = json_data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(
            base_path), json_data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = img_b64_to_arr(imageData)


def get_class_lists(class_list_path):
    classes_lists = []
    # f = open('./dataset/bdd_classes.txt', 'r')
    f = open(class_list_path, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        classes_lists.append(line[:-1])
    f.close()

    return classes_lists

def get_mask_and_label_names_from_json(
    json_data, 
    img_shape, 
    predefined_class_path=None
    ):
    
    
    if predefined_class_path is not None:
        label_name_to_value = {}
        predefined_class_lists = get_class_lists(predefined_class_path)
        for each in predefined_class_lists:
            label_value = len(label_name_to_value)
            label_name_to_value[each] = label_value
    else:
        label_name_to_value = {"_background_": 0}
        for shape in sorted(json_data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
    # print(label_name_to_value)
    lbl, _ = shapes_to_label(
        img_shape, json_data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
        
    return lbl, label_names


def make_mask_img(img, lbl, label_names):
    
    lbl_viz = imgviz.label2rgb(
        lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
    )
    
    return lbl_viz


def save_seg_as_png(
    base_path,
    predefined_class_path=None,
    make_split=False,
    train_ratio=0.9
    ):
    
    img_path = os.path.join(base_path, 'images')
    
    all_json_paths = get_all_json_lists(img_path)
    
    for i in tqdm(range(len(all_json_paths))):
        
        temp_json_data = get_json_info(all_json_paths[i])
        
        imageData = temp_json_data.get("imageData")

        img = img_b64_to_arr(imageData)
        
        lbl, label_names = get_mask_and_label_names_from_json(
            temp_json_data,
            img.shape,
            predefined_class_path
        )
        
        save_path = os.path.join(base_path, 'labels', '%s.png' % (temp_json_data["imagePath"][:-4]))
        
        cv2.imwrite(save_path, lbl)
        
    if make_split:
        split_dataset(base_path, train_ratio)
        
    print('Done !')


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
    

def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def split_dataset(base_path, train_ratio):
    # split train/val set randomly
    split_dir = 'splits'
    ann_dir = 'labels'
    mkdir_or_exist(osp.join(base_path, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in scandir(
        osp.join(base_path, ann_dir), suffix='.png')]
    with open(osp.join(base_path, split_dir, 'train.txt'), 'w') as f:
        # select first 4/5 as train set
        train_length = int(len(filename_list) * train_ratio)
        f.writelines(line + '\n' for line in filename_list[:train_length])
    with open(osp.join(base_path, split_dir, 'val.txt'), 'w') as f:
        # select last 1/5 as train set
        f.writelines(line + '\n' for line in filename_list[train_length:])

def get_all_from_json(
    base_path,
    predefined_class_path=None
    ):
    
    img_path = os.path.join(base_path, 'images')

    all_json_paths = get_all_json_lists(img_path)
    
    temp_json_data = get_json_info(all_json_paths[0])
    
    imageData = temp_json_data.get("imageData")
    
    img = img_b64_to_arr(imageData)
    
    # print(imageData)
    print(np.shape(img))
    
    lbl, label_names = get_mask_and_label_names_from_json(
        temp_json_data, 
        img.shape,
        predefined_class_path
        )
    
    print(np.shape(lbl))
    print(np.unique(lbl))
    print(label_names)
    
    lbl_viz = make_mask_img(img, lbl, label_names)
    
    print(np.shape(lbl_viz))
    
    cv2.imshow("test", img[:, :, (2, 1, 0)])
    # cv2.imshow("test", lbl_viz[:, :, (2, 1, 0)])
    cv2.waitKey(10000)
    
if __name__ == "__main__":
    
    base_path = './dataset/20220908_1'
    class_list_path = './dataset/classes_20220908_1.txt'
    
    # get_all_from_json(base_path, class_list_path)
    save_seg_as_png(
        base_path, 
        class_list_path,
        make_split=True,
        train_ratio=0.9
        )
