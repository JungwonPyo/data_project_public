import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import copy
import random
import matplotlib.pyplot as plt

# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.structures import BoxMode

from pycocotools import mask as coco_mask

from multiprocessing import Pool
from multiprocessing import Process
import parmap

# class_info = {
#     'Wall':                 [134, 187, 113],
#     'Driving Area':         [161, 1, 222],
#     'Non Driving Area':     [136, 189, 197],
#     'Parking Area':         [121, 242, 64],
#     'No Parking Area':      [2, 233, 117],
#     'Big Notice':           [31, 230, 229],
#     'Pillar':               [12, 118, 24],
#     'Parking Area Number':  [78, 152, 159],
#     'Parking Line':         [153, 69, 64],
#     'Disabled Icon':        [3, 116, 111],
#     'Women Icon':           [108, 76, 171],
#     'Compact Car Icon':     [134, 19, 237],
#     'Speed Bump':           [176, 179, 119],
#     'Parking Block':        [142, 88, 239],
#     'Billboard':            [192, 112, 43],
#     'Toll Bar':             [65, 103, 134],
#     'Sign':                 [174, 9, 13],
#     'No Parking Sign':      [124, 254, 106],
#     'Traffic Cone':         [163, 70, 156],
#     'Fire Extinguisher':    [28, 75, 236],
#     'Undefined Object':     [188, 104, 114],
#     'Two-wheeled Vehicle':  [186, 49, 30],
#     'Vehicle':              [82, 60, 101],
#     'Wheelchair':           [154, 231, 118],
#     'Stroller':             [78, 15, 210],
#     'Shopping Cart':        [95, 25, 104],
#     'Animal':               [125, 11, 145],
#     # 'Human':                [88, 28, 38],
#     'Human':                [112, 133, 153],
#     'Undefined Stuff':      [202, 197, 79],
# }
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

# Select only valid labels, 
# becuase labels are too unbalanced
turncated_class_info = {
    'Wall':                 [134, 187, 113],
    'Driving Area':         [161, 1, 222],
    'No Parking Area':      [2, 233, 117],
    'Pillar':               [12, 118, 24],
    'Parking Area Number':  [78, 152, 159],
    'Vehicle':              [82, 60, 101],
    'Others':               [202, 197, 79],
}

def make_turncated_masks_from_gray_masks(
    mask_path,
    save_path,
):
    all_mask_lists = get_img_list_from_path(
        mask_path, '.png', False)
    
    for_convertion_inds = {}
    for ind, each_key in enumerate(class_info.keys()):
        for_convertion_inds[ind] = len(turncated_class_info) - 1
        for sub_ind, sub_each_key in enumerate(turncated_class_info.keys()):
            if each_key == sub_each_key:
                for_convertion_inds[ind] = sub_ind
    
    for each in tqdm(all_mask_lists):

        each_mask = cv2.imread(
            '%s/%s' % (mask_path, each),
            cv2.IMREAD_GRAYSCALE)
        each_mask_origin = copy.deepcopy(each_mask)

        total_gray_mask = np.zeros(np.shape(each_mask)[:2], dtype=np.uint8)

        for ind, each_key in enumerate(class_info.keys()):

            cur_ind = len(turncated_class_info) - 1
            if each_key in turncated_class_info.keys():
                cur_ind = for_convertion_inds[ind]
            
            cur_ind = np.ones((1, 1), dtype=np.uint8) * cur_ind 

            temp_mask = np.where(
                each_mask == ind, cur_ind, 0
            )
            
            total_gray_mask += temp_mask.astype(np.uint8)

        print(each, np.shape(total_gray_mask),
              np.min(total_gray_mask), np.max(total_gray_mask))
        
        if np.max(total_gray_mask) >= len(class_info):
            fig = plt.figure()
            rows = 1
            cols = 2
            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(cv2.resize(each_mask_origin, (1280, 960)))
            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(cv2.resize(total_gray_mask, (1280, 960)))
            plt.show()
            return False

        cv2.imwrite('%s/%s' % (save_path, each), total_gray_mask)


def make_gray_masks_from_color_masks(
    mask_path,
    save_path,
    ):

    all_mask_lists = get_img_list_from_path(
        mask_path, '.png', False)
    
    for each in tqdm(all_mask_lists):
        
        each_mask = cv2.imread('%s/%s' % (mask_path, each))
        each_mask_origin = copy.deepcopy(each_mask)
        
        total_gray_mask = np.zeros(np.shape(each_mask)[:2], dtype=np.uint8)
        
        for ind, each_key in enumerate(class_info.keys()):
            
            cur_key_color = np.asarray(class_info[each_key])
            cur_key_color = np.expand_dims(cur_key_color, axis=(0, 1))
            
            temp_mask = np.where(
                each_mask == cur_key_color, 1, 0
                )
            temp_mask = np.where(
                np.sum(temp_mask, axis=-1) == 3, ind, 0
                )
            
            total_gray_mask += temp_mask.astype(np.uint8)
            
        print(each, np.shape(total_gray_mask),
              np.min(total_gray_mask), np.max(total_gray_mask))
        if np.max(total_gray_mask) >= len(class_info):
            fig = plt.figure()
            rows = 1
            cols = 2
            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(cv2.resize(each_mask_origin, (1280, 960)))
            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(cv2.resize(total_gray_mask, (1280, 960)))
            plt.show()
            return False
        
        cv2.imwrite('%s/%s' % (save_path, each), total_gray_mask)
            
    
def make_gray_masks_from_color_masks_parallel_process(
    mask_path,
    save_path,
):
    
    all_mask_lists = get_img_list_from_path(
        mask_path, '.png', False)
    
    def for_each_parallel_make_gray(
        each
    ):
        each_mask = cv2.imread('%s/%s' % (mask_path, each))
        each_mask_origin = copy.deepcopy(each_mask)

        total_gray_mask = np.zeros(np.shape(each_mask)[:2], dtype=np.uint8)

        for ind, each_key in enumerate(class_info.keys()):

            cur_key_color = np.asarray(class_info[each_key])
            cur_key_color = np.expand_dims(cur_key_color, axis=(0, 1))

            temp_mask = np.where(
                each_mask == cur_key_color, 1, 0
            )
            temp_mask = np.where(
                np.sum(temp_mask, axis=-1) == 3, ind, 0
            )

            total_gray_mask += temp_mask.astype(np.uint8)

        # print(each, np.shape(total_gray_mask),
        #     np.min(total_gray_mask), np.max(total_gray_mask))
        if np.max(total_gray_mask) >= len(class_info):
            fig = plt.figure()
            rows = 1
            cols = 2
            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(cv2.resize(each_mask_origin, (1280, 960)))
            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(cv2.resize(total_gray_mask, (1280, 960)))
            plt.show()
            return False

        cv2.imwrite('%s/%s' % (save_path, each), total_gray_mask)
    
    '''
        The multiprocessing.Pool is generally used for heterogeneous tasks, whereas multiprocessing.Process is generally used for homogeneous tasks.
    '''
    
    # create all tasks
    processes = [Process(target=for_each_parallel_make_gray, args=(each,))
                 for each in all_mask_lists]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)
    
    
def make_one_gray_mask(
    file_path,
    reshape_size=False
):

    each_mask = cv2.imread(file_path)
    
    if reshape_size.any():
        each_mask = cv2.resize(
            each_mask, 
            (reshape_size[1], reshape_size[0]),
            interpolation=cv2.INTER_NEAREST
            )

    total_gray_mask = np.zeros(np.shape(each_mask)[:2], dtype=np.uint8)

    for ind, each_key in enumerate(class_info.keys()):

        cur_key_color = np.asarray(class_info[each_key])
        cur_key_color = np.expand_dims(cur_key_color, axis=(0, 1))

        temp_mask = np.where(
            each_mask == cur_key_color, 1, 0
        )
        temp_mask = np.where(
            np.sum(temp_mask, axis=-1) == 3, ind, 0
        )

        total_gray_mask += temp_mask.astype(np.uint8)

    if np.max(total_gray_mask) >= len(class_info):
        print('Label something wrong !')
        
    return total_gray_mask
    
def for_each_parallel_make_gray(
    each,
    mask_path,
    save_path
):
    
    each = each[:]
    
    each_mask = cv2.imread('%s/%s' % (mask_path, each))
    each_mask_origin = copy.deepcopy(each_mask)

    total_gray_mask = np.zeros(np.shape(each_mask)[:2], dtype=np.uint8)

    for ind, each_key in enumerate(class_info.keys()):

        cur_key_color = np.asarray(class_info[each_key])
        cur_key_color = np.expand_dims(cur_key_color, axis=(0, 1))

        temp_mask = np.where(
            each_mask == cur_key_color, 1, 0
        )
        temp_mask = np.where(
            np.sum(temp_mask, axis=-1) == 3, ind, 0
        )

        total_gray_mask += temp_mask.astype(np.uint8)

    # print(each, np.shape(total_gray_mask),
    #     np.min(total_gray_mask), np.max(total_gray_mask))
    if np.max(total_gray_mask) >= len(class_info):
        fig = plt.figure()
        rows = 1
        cols = 2
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(cv2.resize(each_mask_origin, (1280, 960)))
        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(cv2.resize(total_gray_mask, (1280, 960)))
        plt.show()
        return False

    cv2.imwrite('%s/%s' % (save_path, each), total_gray_mask)
    
    
def make_gray_masks_from_color_masks_parallel_pool(
    mask_path,
    save_path,
    MAX_WORKERS=16
):

    all_mask_lists = get_img_list_from_path(
        mask_path, '.png', False)

    all_mask_lists = [[each] for each in all_mask_lists]

    '''
        The multiprocessing.Pool is generally used for heterogeneous tasks, whereas multiprocessing.Process is generally used for homogeneous tasks.
    '''

    # with Pool(processes=MAX_WORKERS) as pool:
    #     results = tqdm(
    #         pool.imap_unordered(
    #             for_each_parallel_make_gray, 
    #             all_mask_lists, 
    #             chunksize=CHUNK_SIZE
    #             ),
    #         total=len(all_mask_lists),
    #     )  
    #     # 'total' is redundant here but can be useful
    #     # when the size of the iterable is unobvious
    
    #     for result in results:
    #         ...
    #         # print(result)
    
    # parmap.map(
    #     for_each_parallel_make_gray, 
    #     all_mask_lists, 
    #     pm_pbar=True,
    #     pm_processes=MAX_WORKERS
    #     )

    parmap.starmap(
        for_each_parallel_make_gray, 
        all_mask_lists, 
        mask_path,
        save_path,
        pm_pbar=True,
        pm_processes=MAX_WORKERS
        )
    

def get_parking_dicts(img_path, mask_path):
    
    img_list = get_img_list_from_path(img_path, '.jpg', False)
    # color_mask_list = get_img_list_from_path(color_mask_path, '.png', True)
    # gray_mask_list = get_img_list_from_path(gray_mask_path, '.png', True)

    dataset_dicts = []
    for idx, img_name in enumerate(img_list):
        record = {}

        filename = '%s/%s' % (img_path, img_name)
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        # mask = cv2.imread(mask_paths)
        # coco_mask.encode(np.asarray(mask, order="F"))
        
        record["sem_seg_file_name"] = '%s/%s.png' % (mask_path, img_name[:-4])
        
        segments_info = []
        for ind, each_key in enumerate(class_info.keys()):
            obj = {
                "id": ind,
                "category_id": ind
            }
            segments_info.append(segments_info)
        record["segments_info"] = segments_info
        
        dataset_dicts.append(record)
        
    return dataset_dicts

def register_metadata(
    name, 
    dicts_class,
    class_info
    ):
    DatasetCatalog.register(name, dicts_class)
    
    names = []
    colors = []
    for key in class_info.keys():
        names.append(key)
        colors.append(tuple(class_info[key]))
    
    MetadataCatalog.get(name).stuff_classes = names
    MetadataCatalog.get(name).stuff_colors = colors
    
    # dataset_metadata = MetadataCatalog.get("%s_train" % (name))
    # return dataset_metadata

def get_unique_colors_from_all_data(
    file_list
):

    total_unique = []

    for each in tqdm(file_list):

        # temp_seg = cv2.imread('%s/%s' % (base_path, each))
        temp_seg = cv2.imread(each)

        temp_seg_shape = np.shape(temp_seg)

        temp_seg_reshape = temp_seg.reshape(
            (temp_seg_shape[0] * temp_seg_shape[1], temp_seg_shape[2])
        )

        # print(np.unique(temp_seg_reshape, axis=0))
        temp_seg_reshape_unique = np.unique(temp_seg_reshape, axis=0)
        # print(np.shape(temp_seg_reshape_unique))

        for i in range(np.shape(temp_seg_reshape_unique)[0]):

            total_unique.append(list(temp_seg_reshape_unique[i, :]))
            total_unique = [each for each in sorted(total_unique)]

    return total_unique


def get_img_list_from_path(path, extension='.png', full_path=True):

    return_list = []

    for images in os.listdir(path):
        if (images.endswith(extension)):
            # print(images)
            if full_path:
                return_list.append('%s/%s' % (path, images))
            else:
                return_list.append(images)

    return return_list


if __name__ == '__main__':
    
    ''' Set metadata again '''
    train_img_path = '/media/asura/T7 Shield/for_mid/data/merged/train/img'
    train_color_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/train/mask'
    train_gray_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/train/gray_mask'
    train_turn_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/train/turn_mask'

    val_img_path = '/media/asura/T7 Shield/for_mid/data/merged/val/img'
    val_color_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/val/mask'
    val_gray_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/val/gray_mask'
    val_turn_mask_path = '/media/asura/T7 Shield/for_mid/data/merged/val/turn_mask'

    # make_gray_masks_from_color_masks(train_color_mask_path, train_gray_mask_path)
    # make_gray_masks_from_color_masks(val_color_mask_path, val_gray_mask_path)

    # make_turncated_masks_from_gray_masks(
    #     train_gray_mask_path, train_turn_mask_path)
    # make_turncated_masks_from_gray_masks(
    #     val_gray_mask_path, val_turn_mask_path)

    train_dataset_name = "parking_train"
    val_dataset_name = "parking_val"

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "parking_" + d, lambda d=d: get_parking_dicts(
                "/media/asura/T7 Shield/for_mid/data/merged/" + d + "/img",
                # "/media/asura/T7 Shield/for_mid/data/merged/" + d + "/gray_mask"
                "/media/asura/T7 Shield/for_mid/data/merged/" + d + "/turn_mask"
            )
        )

    names = []
    colors = []
    # for key in class_info.keys():
    for key in turncated_class_info.keys():
        names.append(key)
        # colors.append(tuple(class_info[key]))
        colors.append(tuple(turncated_class_info[key]))

    MetadataCatalog.get(train_dataset_name).stuff_classes = names
    MetadataCatalog.get(train_dataset_name).stuff_colors = colors
    MetadataCatalog.get(train_dataset_name).ignore_label = 20
    MetadataCatalog.get(
        train_dataset_name).evaluator_type = "sem_seg"
    MetadataCatalog.get(val_dataset_name).stuff_classes = names
    MetadataCatalog.get(val_dataset_name).stuff_colors = colors
    MetadataCatalog.get(val_dataset_name).ignore_label = 20
    MetadataCatalog.get(
        val_dataset_name).evaluator_type = "sem_seg"
    ''' End metadata '''
    
    dataset_dicts = get_parking_dicts(
        train_img_path,
        # train_gray_mask_path
        train_turn_mask_path
        )
    
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=MetadataCatalog.get(train_dataset_name), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.imshow(cv2.resize(out.get_image(), (1280, 960)))
        plt.show()
    
    
