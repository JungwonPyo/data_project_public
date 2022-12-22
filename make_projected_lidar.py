import cv2
import numpy as np
import cv2
# import tqdm
from tqdm import tqdm
import open3d as o3d
from utils.preprocessing import preprocessing

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    image_path = '/media/asura/T7_Shield_1/for_mid/20221216/img'
    lidar_path = '/media/asura/T7_Shield_1/for_mid/20221216/lidar'
    projected_lidar_path = '/media/asura/T7_Shield_1/for_mid/20221216/lidar_projected'

    preprocessing_class = preprocessing(
        image_path=image_path,
        image_size=[4032, 3040],
        # image_size=[640, 480],
        lidar_path=lidar_path,
        label_path='',
        list_filename=None,
        separate_by_time=False,
        config_filename=None,
        undistortion=False,
        hp_map_file_path=None,
        hd_map_offset=[302209., 4124130.],
        ignore_link_ids=None,
        lidar_frame_id='custom',
        lidar_publish_name='/custom_points',
        time_interval=0.1,
        index_interval=1,
        # time_interval=0.2,
        # index_interval=2,
        # time_interval=1.0,
        # index_interval=10,
        using_ros=False
    )
    
    for i in tqdm(range(len(preprocessing_class.image_files))):
    
        img, matched_img, each_pcd = preprocessing_class.read_each_from_lists(i)
        
        cv2.imwrite('%s/%s.png' %
                    (projected_lidar_path,
                     preprocessing_class.image_files[i][:-4]),
                    cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
