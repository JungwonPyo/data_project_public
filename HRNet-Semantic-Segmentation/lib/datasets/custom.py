# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import random

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

from core_utils.detectron2_dataloader import *

class Custom(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 random_samples=None,
                 use_json=False,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 resize_shape=[1280, 1920]):

        super(Custom, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.use_json = use_json
        
        self.resize_shape = np.asarray(resize_shape)
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        if random_samples:
            self.files = random.choices(self.files, k=random_samples)

        '''
            class_info = {
                'Wall':                 [134, 187, 113],
                'Driving Area':         [161, 1, 222],
                'Non Driving Area':     [136, 189, 197],
                'Parking Area':         [121, 242, 64],
                'No Parking Area':      [2, 233, 117],
                'Big Notice':           [31, 230, 229],
                'Pillar':               [12, 118, 24],
                'Parking Area Number':  [78, 152, 159],
                'Parking Line':         [153, 69, 64],
                'Disabled Icon':        [3, 116, 111],
                'Women Icon':           [108, 76, 171],
                'Compact Car Icon':     [134, 19, 237],
                'Speed Bump':           [176, 179, 119],
                'Parking Block':        [142, 88, 239],
                'Billboard':            [192, 112, 43],
                'Toll Bar':             [65, 103, 134],
                'Sign':                 [174, 9, 13],
                'No Parking Sign':      [124, 254, 106],
                'Traffic Cone':         [163, 70, 156],
                'Fire Extinguisher':    [28, 75, 236],
                'Undefined Object':     [188, 104, 114],
                'Two-wheeled Vehicle':  [186, 49, 30],
                'Vehicle':              [82, 60, 101],
                'Wheelchair':           [154, 231, 118],
                'Stroller':             [78, 15, 210],
                'Shopping Cart':        [95, 25, 104],
                'Animal':               [125, 11, 145],
                'Human':                [112, 133, 153],
                'Undefined Stuff':      [202, 197, 79],
            }
        '''
        self.label_mapping = {
            0: 0, 
            1: 1, 
            2: 2, 
            3: 3, 
            4: 4, 
            5: 5, 
            6: 6, 
            7: 7, 
            8: 8, 
            9: 9, 
            10: 10, 
            11: 11, 
            12: 12, 
            13: 13, 
            14: 14, 
            15: 15, 
            16: 16, 
            17: 17, 
            18: 18, 
            19: 19, 
            20: 20, 
            21: 21, 
            22: 22, 
            23: 23, 
            24: 24,
            25: 25, 
            26: 26, 
            27: 27, 
            28: 28, 
            }
        # self.class_weights = torch.FloatTensor([
        #     1., 1., 1., 1., 1., # -1, 0, 1, 2, 3
        #     1., 1., 1., 1., 1., # 4, 5, 6, 7, 8
        #     1., 1., 1., 1., 1., # 9, 10, 11, 12, 13
        #     1., 1., 1., 1., 1., # 14, 15, 16, 17, 18
        #     1., 1., 1., 1., 1., # 19, 20, 21, 22, 23
        #     1., 1., 1., 1. # 24, 25, 26, 27
        #     ]).cuda()
        self.class_weights = torch.FloatTensor([
            0.83, 0.9, 1., 1., 1.1, # -1, 0, 1, 2, 3
            1.1, 1., 1., 1.1, 1.2, # 4, 5, 6, 7, 8
            1.2, 1.2, 1.1, 1.1, 1.2, # 9, 10, 11, 12, 13
            1.2, 1., 1.2, 1.1, 1.2, # 14, 15, 16, 17, 18
            1.2, 1.1, 0.9, 1.1, 1.2, # 19, 20, 21, 22, 23
            1.2, 1.2, 1.1, 0.81 # 24, 25, 26, 27
            ]).cuda()
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                if self.use_json:
                    image_path, label_path, json_path = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "json": json_path,
                        "name": name,
                    })
                else:
                    image_path, label_path, _ = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "name": name,
                    })
        else:
            for item in self.img_list:
                if self.use_json:
                    image_path, label_path, json_path = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "json": json_path,
                        "name": name,
                        "weight": 1
                    })
                else:
                    image_path, label_path, _ = item
                    name = os.path.splitext(os.path.basename(label_path))[0]
                    files.append({
                        "img": image_path,
                        "label": label_path,
                        "name": name,
                        "weight": 1
                    })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, item["img"]),
                           cv2.IMREAD_COLOR)
        
        if self.resize_shape.any():
            image = cv2.resize(
                image,
                (self.resize_shape[1], self.resize_shape[0])
            )

        if 'test' in self.list_path:
            
            size = image.shape
            
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            
            if self.use_json:
                label = make_one_gray_mask(
                    os.path.join(self.root, item["label"]), 
                    self.resize_shape, 
                    os.path.join(self.root, item["json"]),
                    )
            else:
                label = make_one_gray_mask(os.path.join(
                    self.root, item["label"]), self.resize_shape)
            label = self.convert_label(label)

            return image.copy(), label.copy(), np.array(size), name

        size = image.shape
        
        if self.use_json:
            label = make_one_gray_mask(
                os.path.join(self.root, item["label"]), 
                self.resize_shape, 
                os.path.join(self.root, item["json"]),
                )
        else:
            label = make_one_gray_mask(os.path.join(
                self.root, item["label"]), self.resize_shape)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int64(self.crop_size[0] * 1.0)
        stride_w = np.int64(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
                
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int64(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int64(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]

            preds = F.interpolate(
                preds, (ori_height, ori_width), 
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )            
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
