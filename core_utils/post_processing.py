import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import copy
import random
import matplotlib.pyplot as plt
import scipy.io as sio

from core_utils.detectron2_dataloader import *


class iou_evaluation(object):
    
    def __init__(
        self,
        iou_thres=0.5,
        save_result_path='./samples',
        save_result_name='result'
    ):
        self.iou_thres = iou_thres
        
        # Will be stored results
        self.ind_mapping = {}
        self.result_set = {}
        for ind, key in enumerate(class_info.keys()):
            self.ind_mapping[ind] = key
            self.result_set[key] = []
        self.save_result_path = save_result_path
        self.save_result_name = save_result_name
        
    def get_iou_between_two_maps(self, est_map, gt_map):
        
        max_run = np.max(gt_map)
        
        for i in range(max_run):
            
            cur_gt_map = np.where(gt_map == i, 1, 0)
            
            # check ground-truth has this class
            if np.sum(cur_gt_map) == 0:
                continue
            
            cur_est_map = np.where(est_map == i, 1, 0)
            
            union_area = np.sum(np.where((cur_gt_map + cur_est_map) > 0, 1, 0))
            intersection_area = np.sum(np.where((cur_gt_map + cur_est_map) == 2, 1, 0))
            
            iou_val = intersection_area / union_area
            
            self.result_set[self.ind_mapping[i]].append(iou_val)
            
    def save_results(self):
        sio.savemat('%s/%s.mat' % (self.save_result_path,
                    self.save_result_name), self.result_set)

    def print_results(self):
        
        print("Object\t\t\tIoU\n")

        total_miou = []
        for ind, key in enumerate(class_info.keys()):
            
            each_mean_iou = np.mean(np.asarray(self.result_set[key]))
            
            print('%s\t\t\t%.2f' % (key, each_mean_iou * 100.))
            
            total_miou.append(each_mean_iou)
            
        total_miou = np.asarray(total_miou)

        print("================================")

        print('2022-09-27 22:24:05\t- Result')

        print('Mean IoU : %s ' % np.mean(total_miou))
        print('Best IoU : %s ' % np.max(total_miou))

        print("================================\n")
        
