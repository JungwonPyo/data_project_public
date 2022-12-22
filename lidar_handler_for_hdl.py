import cv2
import numpy as np
import cv2
import tqdm
from core_utils.preprocessing import preprocessing


if __name__ == '__main__':
    
    base_path = '/media/asura/T7_Shield_1/GeoN_location'

    save_mat_path = './samples'

    temp_place = 'hospital_general'

    temp_time = ''
    temp_ind = 2

    if temp_time == '':
        image_path = '%s/%s/camera/camera' % (
            base_path, temp_place)
        lidar_path = '%s/%s/pcd' % (
            base_path, temp_place)
        list_filename = '%s/%s/csv/%s.csv' % (
            base_path, temp_place, temp_place)
        save_mat_name = '%s' % (
            temp_place)
    else:
        image_path = '%s/%s/%s/camera' % (
            base_path, temp_place, temp_time)
        lidar_path = '%s/%s/%s/pcd' % (
            base_path, temp_place, temp_time)
        list_filename = '%s/%s/%s/csv/%s_%02d.csv' % (
            base_path, temp_place, temp_time, temp_place, temp_ind)
        save_mat_name = '%s_%02d_hdl.mat' % (
            temp_place, temp_ind)
    label_path = ''
    
    preprocessing_class = preprocessing(
        image_path=image_path,
        image_size=[4032, 3040],
        # image_size=[640, 480],
        lidar_path=lidar_path,
        label_path=label_path,
        list_filename=list_filename,
        config_filename=None,
        undistortion=True,
        hp_map_file_path=None,
        hd_map_offset=[302209., 4124130.],
        ignore_link_ids=None,
        lidar_frame_id='custom',
        lidar_publish_name='/custom_points',
        imu_publish_name='/imu',
        time_interval=0.1,
        index_interval=1,
        # time_interval=0.2,
        # index_interval=2,
        # time_interval=1.0,
        # index_interval=10,
        using_ros=True
    )
    
    preprocessing_class.pulish_lidar_frame_and_save_tf(
        start_frame=0,
        max_frame=preprocessing_class.list_pd_len,
        tf_from='/odom',
        tf_to='/custom',
        save_mat_path=save_mat_path,
        save_mat_name=save_mat_name,
        get_tf_and_save=True
    )

