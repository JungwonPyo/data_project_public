import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import scipy.io as sio

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

hrnet_repo_path = './HRNet-Semantic-Segmentation'

sys.path.append(hrnet_repo_path)
lib_path = os.path.join(hrnet_repo_path, 'lib')
sys.path.append(lib_path)
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test, testval_custom
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
sys.path.remove(hrnet_repo_path)
sys.path.remove(lib_path)

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
    'Human':                [88, 28, 38],
    'Undefined Stuff':      [202, 197, 79],
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def print_iou(
    iou, 
    mean_pixel_acc, 
    pixel_acc, 
    class_names=None, 
    show_no_back=False, 
    no_print=False
    ):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
            'mean_IoU', mean_IoU * 100, 
            'mean_IU_no_back', mean_IoU_no_back*100,
            'mean_pixel_acc', mean_pixel_acc*100, 
            'pixel_acc',pixel_acc*100
            ))
    else:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
            'mean_IoU', mean_IoU * 100, 
            'mean_pixel_acc', mean_pixel_acc*100, 
            'pixel_acc',pixel_acc*100
            ))
    line = "\n".join(lines)
    if not no_print:
        print(line)

    logging.info(line)

    return line

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logging.info(pprint.pformat(args))
    logging.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth')        
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        resize_shape=config.DATASET.RESIZE_SHAPE,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    if 'val' in config.DATASET.TEST_SET:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(
            config, 
            test_dataset, 
            testloader, 
            model,
            sv_dir='./hrnet_results',
            sv_pred=True
            )
    
        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)
    elif 'test' in config.DATASET.TEST_SET:
        # test(config, 
        #      test_dataset, 
        #      testloader, 
        #      model,
        #      sv_dir=final_output_dir)
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval_custom(
            config, 
            test_dataset, 
            testloader, 
            model,
            sv_dir='./hrnet_results',
            sv_pred=True
            )
    
        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}'.format(mean_IoU, 
            pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)

        print_iou(
            np.array(IoU_array), 
            mean_acc, 
            pixel_acc, 
            class_names=list(class_info.keys()), 
            show_no_back=False, 
            no_print=False
            )
        sio.savemat(
            './hrnet_results/results.mat',
            {
                'mean_IoU': mean_IoU,
                'IoU_array': IoU_array,
                'pixel_acc': pixel_acc,
                'mean_acc': mean_acc,
            }
        )

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int64((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()

# python test_hrnet.py --cfg ./configs/hrnet_custom_train.yaml
# python test_hrnet.py --cfg ./configs/hrnet_custom_test.yaml
