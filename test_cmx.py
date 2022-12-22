import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from PIL import Image

from configs.cmx_config_custom import config

cmx_repo_path = './RGBX_Semantic_Segmentation'

sys.path.append(cmx_repo_path)
# from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
sys.path.remove(cmx_repo_path)

logger = get_logger()

class SegEvaluator(Evaluator):

    def __init__(self, *args):
        super().__init__(*args)
        self.palette = self.get_palette(256)

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

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            preds = np.asarray(pred, dtype=np.uint8)
            pred = self.convert_label(preds, inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(self.palette)
            save_img.save(os.path.join(self.save_path+'_color', fn))

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    # 'eval_source': config.eval_source,
                    'eval_source': config.test_source,
                    'class_names': config.class_names}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)

# python test_cmx.py -p ./cmx_log/result_images