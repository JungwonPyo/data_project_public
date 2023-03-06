import numpy as np
import cv2
import os
import sys
import struct
# import scipy.io
import time
import math
import open3d as o3d
import pandas as pd
import copy
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformations import euler_from_matrix
import matplotlib
import matplotlib.cm
from core_utils.camera_model import PinholeCameraModel

if sys.version_info[0] == 2:
    from pyproj import Proj, transform
else:
    from pyproj import Proj, CRS, transform
    from pyproj import Transformer
import shapefile

import rospy
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import NavSatFix, NavSatStatus, PointCloud2, PointField, Imu
from std_msgs.msg import Header
from geographic_msgs.msg import GeoPointStamped
from rosgraph_msgs.msg import *
import tf
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped, WrenchStamped

'''
    Dogugonggan sented in 2022.08.07
    image width 640 height 480

    camera matrix
    441.033860 0.000000 319.671152
    0.000000 439.186728 237.963391
    0.000000 0.000000 1.000000

    distortion
    -0.703113 0.363788 -0.024956 -0.005854 0.000000

    rectification
    1.000000 0.000000 0.000000
    0.000000 1.000000 0.000000
    0.000000 0.000000 1.000000

    projection
    310.364105 0.000000 313.566936 0.000000
    0.000000 318.652924 213.742362 0.000000
    0.000000 0.000000 1.000000 0.000000
    
    camera-lidar extrinsic
    1.01520529e-02 -9.99947504e-01  1.38757728e-03 -0.01396078
    6.82573541e-02 -6.91424795e-04 -9.97667508e-01 -0.09558755
    9.97616093e-01  1.02230857e-02  6.82467515e-02 -0.07756302
    0 0 0 1
'''

'''
    Dogugonggan sented in 2022.08.10
    image width 1920 height 1080

    camera matrix
    1478.453504 0.000000 975.691216
    0.000000 1472.703315 376.876506
    0.000000 0.000000 1.000000

    distortion
    0.081891 -0.179290 0.001540 -0.001314 0.000000

    rectification
    1.000000 0.000000 0.000000
    0.000000 1.000000 0.000000
    0.000000 0.000000 1.000000

    projection
    1482.718506 0.000000 974.054154 0.000000
    0.000000 1488.223145 377.875855 0.000000
    0.000000 0.000000 1.000000 0.000000
    
    camera-lidar extrinsic
    0.01359768 -0.99975766 -0.01731238 -0.03784309
    0.08916464  0.01845737 -0.99584587 -0.11696305
    0.99592408  0.01199754  0.08939401 -0.03077733
    0 0 0 1
'''

'''
    See this to know the basic knowledge about camera calibration:
        https://www.mathworks.com/help/vision/ug/camera-calibration.html
    See this to check how some parameters are generated:
        https://github.com/heethesh/lidar_camera_calibration
'''

class preprocessing(object):
    
    def __init__(
        self,
        image_path,
        image_size, # [w, h] # [1920, 1080]
        lidar_path,
        label_path,
        list_filename,
        separate_by_time=True,
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
        using_ros=False,
        circle_radius=5
        ):
        
        self.image_path = image_path
        self.image_size = image_size
        self.lidar_path = lidar_path
        self.label_path = label_path
        self.list_filename = list_filename
        self.separate_by_time = separate_by_time
        self.config_filename = config_filename
        self.undistortion = undistortion
        self.hp_map_file_path = hp_map_file_path
        self.hd_map_offset = hd_map_offset
        self.ignore_link_ids = ignore_link_ids
        self.lidar_frame_id = lidar_frame_id
        self.lidar_publish_name = lidar_publish_name
        self.imu_publish_name = imu_publish_name
        self.time_interval = time_interval
        self.index_interval = index_interval
        self.duration = rospy.Duration(self.time_interval)
        self.using_ros = using_ros
        self.circle_radius = circle_radius
        
        if self.using_ros:
            
            rospy.init_node('custom_publisher')
            
            self.bridge = CvBridge()
            
            self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=256)
            self.lidar_pub = rospy.Publisher(
                '%s' % (self.lidar_publish_name), PointCloud2, queue_size=32)
            self.imu_pub = rospy.Publisher(self.imu_publish_name, Imu, queue_size=32)
            self.listener = tf.TransformListener()
            # self.rate = rospy.Rate(hz)
        
        if sys.version_info[0] == 2:
            self.proj_UTM52N = Proj(init='epsg:32652')
            self.proj_WGS84 = Proj(init='epsg:4326')
        else:
            self.gps_transformer = Transformer.from_crs(
                4326, 32652, always_xy=True)
        
        if self.image_path != '':
            self.image_files = self.get_filenames_from_folder(self.image_path)
        if self.lidar_path != '':
            self.lidar_files = self.get_filenames_from_folder(self.lidar_path)
        
        
        ## If ther is not list file, get all lists from 
        if self.list_filename is not None:
            self.list_pd = pd.read_csv(self.list_filename)
            self.list_pd_len = len(self.list_pd)
        
        # self.lists_by_time = []
        # if self.separate_by_time:
        #     self.lists_by_time = self.seperate_by_time(self.list_pd)
        
        self.lists_by_dist = []
        if self.separate_by_time:
            self.lists_by_dist = self.seperate_by_dist(self.list_pd)

        # Set custom matrix if there is no config file
        if self.config_filename is None:

            '''
                GeoN sensor parameters:
                >> tform.Translation

                ans =

                    0.0675    0.0864    0.4389

                >> tform.Rotation

                ans =

                    1.0000    0.0010    0.0002
                -0.0002    0.0223    0.9998
                    0.0010   -0.9998    0.0223
                    
                >> tform.Translation

                ans =

                -0.0342    0.3084    0.4439

                >> tform.Rotation

                ans =

                    0.9968   -0.0792   -0.0137
                    0.0129   -0.0108    0.9999
                -0.0794   -0.9968   -0.0098
                
                >> tform.Translation

                ans =

                    0.0861    0.2408    0.3818

                >> tform.Rotation

                ans =

                    0.9998   -0.0039    0.0205
                -0.0206   -0.0279    0.9994
                -0.0033   -0.9996   -0.0280
            '''

            self.camera_intrinsic = np.array([
                [2124.899035, 0.000000, 2016.000000],
                [0.000000, 2124.899035, 1520.000000],
                [0.000000, 0.000000, 1.000000]
            ])
            self.camera_distortion = np.array([
                -0.015349, -0.048668, -0.001330, -0.003238, 0.0
            ])
            self.camera_lidar_extrinsic = np.array([
                [0.9998, -0.0039, 0.0205, 0.0000],
                [-0.0206, -0.0279, 0.9994, 0.0000],
                [-0.0033, -0.9996, -0.0280, 0.0000],
                [0.0861, 0.2408, 0.3818, 1.0000],
            ]).transpose()
            '''
                0.999961802241891	-0.00488760215544343	0.00724606115896803	0
                -0.00721499767082878	0.00634353668543567	0.999953850610582	0
                -0.00493334225037506	-0.999967934929661	0.00630803033838834	0
                0.00208433790457183	0.190056099985926	0.00986245921618803	1
            '''
            temp_eye = np.eye(4, dtype=np.float)
            temp_intrinsic = np.array([
                [2007.6, 0.000000, 2079.9],
                [0.000000, 1994.1, 1503.8],
                [0.000000, 0.000000, 1.000000]
            ])
            
            temp_eye[:3, :3] = temp_intrinsic
            self.projection_matrix = temp_eye

        else:
            ...
            
        self.camera_info_class = PinholeCameraModel()
          
        self.camera_info_class.fromCameraInfo(
            K=self.camera_intrinsic,
            D=self.camera_distortion,
            R=self.camera_lidar_extrinsic[:3, :3],
            P=self.projection_matrix[:3, :],
            # P=self.K_mul_R,
            width=self.image_size[0], 
            height=self.image_size[1]
        )
        
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic, 
            self.camera_distortion, 
            (self.image_size[0], self.image_size[1]), 
            1, 
            (self.image_size[0], self.image_size[1])
            )
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_intrinsic, 
            self.camera_distortion, 
            None, 
            self.newcameramtx, 
            (self.image_size[0], self.image_size[1]), 
            5
            )

    def get_only_min_from_str(self, input):

        return int(input[9:-3])

    def seperate_by_time(self, list_pd):
        
        return list_pd.groupby(list_pd['TIME'].map(
            lambda x: self.get_only_min_from_str(x)).diff().ge(10).cumsum())['TIME'].apply(list).tolist()
        
    def get_euclidean(self, input):

        return np.sqrt(input['EASTING']**2 + input['NORTHING']**2)
        
    def seperate_by_dist(self, list_pd):

        list_pd_euclidean = list_pd[['EASTING', 'NORTHING']].diff().apply(
            lambda x: self.get_euclidean(x), axis=1).rename('EUCLIDEAN')

        list_pd['EUCLIDEAN'] = list_pd_euclidean

        list_pd_sep_by_euclidean = list_pd.groupby(
            list_pd['EUCLIDEAN'].diff().ge(10).cumsum())['EUCLIDEAN'].apply(list).tolist()

        return list_pd_sep_by_euclidean

    def get_depth_colors(self, nums):

        # define colors
        color1 = (0, 0, 255)  # red
        color2 = (0, 165, 255)  # orange
        color3 = (0, 255, 255)  # yellow
        color4 = (255, 255, 0)  # cyan
        color5 = (255, 0, 0)  # blue
        color6 = (128, 64, 64)  # violet
        colorArr = np.array(
            [[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

        # resize lut to 256 (or more) values
        lut = cv2.resize(colorArr, (nums, 1), interpolation=cv2.INTER_LINEAR)
        
        return lut

    def project_point_cloud(
        self,
        lidar,
        img
        ):
        
        if self.undistortion:
            img = cv2.remap(img, self.mapx,self.mapy, cv2.INTER_LINEAR)
        
        points3D = np.asarray(lidar.points)

        max_range = 20

        # Filter points in front of camera
        inrange = np.where((points3D[:, 2] > 0) &
                           (points3D[:, 2] < max_range) &
                           (np.abs(points3D[:, 0]) < max_range) &
                           (np.abs(points3D[:, 1]) < max_range))

        max_intensity = max_range * 1e3
        points3D = points3D[inrange[0]]
        
        points2D = self.camera_info_class.project3dToPixel_based_on_intrinsic(
            points3D)
        inrange = np.where((points2D[:, 0] >= 0) &
                           (points2D[:, 1] >= 0) &
                           (points2D[:, 0] < np.shape(img)[1]) &
                           (points2D[:, 1] < np.shape(img)[0]))
        points2D = points2D[inrange[0]]

        cmap = matplotlib.cm.get_cmap('jet')
        colors = cmap((points2D[:, -1] * 1e3) / max_intensity) * 255

        # Draw the projected 2D points
        result = copy.deepcopy(img)
        for i in range(len(points2D)):
            cv2.circle(result, tuple(points2D[i, :2].round().astype(
                'int')), 10, tuple(colors[i, :]), -1)
         
        return img, result
    
    def project_point_cloud_with_clustering(
        self,
        lidar,
        img,
        camera_lidar_extrinsic,
        clustering_class
    ):

        if self.undistortion:
            img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # Do clustering 
        result_pcd, result_labels = clustering_class.o3d_get_dbscan_result_at_once(lidar)

        max_label = result_labels.max()
        # print(f"point cloud has {max_label + 1} clusters")
        cluster_colors = plt.get_cmap("tab20")(
            result_labels / (max_label if max_label > 0 else 1))
        cluster_colors[result_labels < 0] = 0

        max_intensity = np.max(np.asarray(result_pcd.points))

        points2D = self.camera_info_class.project3dToPixel_based_on_intrinsic(
            np.asarray(copy.deepcopy(result_pcd).transform(camera_lidar_extrinsic).points))
        inrange = np.where((points2D[:, 0] >= 0) &
                           (points2D[:, 1] >= 0) &
                           (points2D[:, 0] < np.shape(img)[1]) &
                           (points2D[:, 1] < np.shape(img)[0]))
        points2D = points2D[inrange[0]]
        cluster_colors = cluster_colors[inrange[0]]

        cmap = matplotlib.cm.get_cmap('jet')
        colors = cmap((points2D[:, -1] * 1e3) / max_intensity) * 255

        # Draw the projected 2D points
        result = copy.deepcopy(img)
        for i in range(len(points2D)):
            cv2.circle(result, tuple(points2D[i, :2].round().astype(
                'int')), self.circle_radius, tuple(colors[i, :]), -1)
            
        result_clustering = copy.deepcopy(img)
        for i in range(len(points2D)):
            cv2.circle(result_clustering, tuple(points2D[i, :2].round().astype(
                'int')), self.circle_radius, tuple(cluster_colors[i, :] * 255), -1)

        return img, result, result_clustering
        
    def read_pcd_file(self, filename):
        return o3d.io.read_point_cloud(filename)
    
    def get_filenames_from_folder(self, path):
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        return files
    
    def check_same_filenames(self, name1, name2):
        if name1[:-4] == name2[:-4]:
            True
        else:
            False
            
    def make_projected_image(
        self, 
        input_rgb,
        input_pcd_path
    ):
        # print('read: ', input_pcd_path)
        input_pcd = self.read_pcd_file(input_pcd_path)
        each_pcd_t = copy.deepcopy(input_pcd).transform(
            self.camera_lidar_extrinsic)

        _, matched_img = self.project_point_cloud(
            each_pcd_t, cv2.resize(np.asarray(input_rgb), tuple(self.image_size)))

        return matched_img
    
    def read_each_from_lists(self, index):
        each_png_name = self.image_files[index]
        
        each_png = o3d.io.read_image('%s/%s' % (self.image_path, each_png_name))
        each_pcd = self.read_pcd_file(
            '%s/%s.pcd' % (self.lidar_path, each_png_name[:-4]))
        each_pcd_t = copy.deepcopy(each_pcd).transform(self.camera_lidar_extrinsic)
        
        img, matched_img = self.project_point_cloud(each_pcd_t, cv2.resize(np.asarray(each_png), tuple(self.image_size)))
        
        # return img, matched_img, each_pcd_t
        return img, matched_img, each_pcd
            
    def read_each(self, index):
        each_png_name = self.list_pd.loc[index]['CAMERA']
        each_pcd_name = self.list_pd.loc[index]['LIDAR']

        each_png = o3d.io.read_image('%s/%s' % (self.image_path, each_png_name))
        each_pcd = self.read_pcd_file(
            '%s/%s' % (self.lidar_path, each_pcd_name))
        each_pcd_t = copy.deepcopy(each_pcd).transform(self.camera_lidar_extrinsic)
        
        img, matched_img = self.project_point_cloud(each_pcd_t, cv2.resize(np.asarray(each_png), tuple(self.image_size)))
        
        # return img, matched_img, each_pcd_t
        return img, matched_img, each_pcd


    def normalize(self, val, min_val, max_val):
        val_between_0_and_1 = (val - min_val) / (abs(min_val) + abs(max_val))
        val_between_minus_1_and_1 = (val_between_0_and_1 * 2.0) - 1.0
        return val_between_minus_1_and_1

    def fix(self, val):
        MAX_VAL = 32767
        MIN_VAL = -32768
        return self.normalize(val, MIN_VAL, MAX_VAL)

    def create_quaternion(self, q1, q2, q3, q4):
        q = Quaternion()
        # Failed attempt
        # q.x = fix(q1)
        # q.y = fix(q2)
        # q.z = fix(q3)
        # q.w = fix(q4)
        # The quaternion IS w,x,y,z
        q.x = self.fix(q2)
        q.y = self.fix(q3)
        q.z = self.fix(q4)
        q.w = self.fix(q1)
        return q

    def create_vector3(self, groll, gpitch, gyaw):
        # print "Creating Vector3 from:"
        # print (groll, gpitch, gyaw)
        v = Vector3()
        # I guess the units will be like that, but I don't know
        v.x = groll / 1000.0
        v.y = gpitch / 1000.0
        v.z = gyaw / 1000.0
        return v

    def read_each_imu_as_quaternion(self, index):

        roll = self.list_pd.loc[index]['ROLL']
        pitch = self.list_pd.loc[index]['PITCH']
        yaw = self.list_pd.loc[index]['HEADING']

        # because these values are degree, change these to radian
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # then, need to change quarternion to make imu data
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        return quaternion
    
    def read_each_with_clustering(
        self, 
        index, 
        clustering_class
        ):
        each_png_name = self.list_pd.loc[index]['CAMERA']
        each_pcd_name = self.list_pd.loc[index]['LIDAR']

        each_png = o3d.io.read_image(
            '%s/%s' % (self.image_path, each_png_name))
        each_pcd = self.read_pcd_file(
            '%s/%s' % (self.lidar_path, each_pcd_name))

        img, matched_img, matched_img_clustering = self.project_point_cloud_with_clustering(
            each_pcd, 
            cv2.resize(np.asarray(each_png), tuple(self.image_size)),
            self.camera_lidar_extrinsic,
            clustering_class
            )

        return img, matched_img, matched_img_clustering, each_pcd
    
    def open3d2pointcloud(self, time, o3d_points):
        
        if self.using_ros:
            cloud = PointCloud2()
            # cloud.header.stamp = rospy.Time.from_sec(time)
            cloud.header.stamp = time
            cloud.header.frame_id = self.lidar_frame_id
            points_np = np.asarray(o3d_points.points, dtype=np.float32)

            cloud = pc2.create_cloud_xyz32(
                cloud.header, points_np)
            
            return cloud
        else:
            return None
    
    def publish_lidar_frames(self, start_frame, max_frame):
        
        # cur_time = rospy.Time.now()
        cur_time = rospy.Time.from_sec(0.0)
        
        # while not rospy.is_shutdown():
        for i in tqdm(range(start_frame, max_frame, self.index_interval)):
        
            print(cur_time)
            
            _, _, points = self.read_each(i)
            print(np.shape(points.points))
            
            cloud_msg = self.open3d2pointcloud(cur_time, points)

            clock_msg = Clock()
            clock_msg.clock = cur_time
            
            # if pub.get_num_connections():
            self.clock_pub.publish(clock_msg)
            self.lidar_pub.publish(cloud_msg)
            
            cur_time += self.duration
            # rospy.sleep(self.duration)
            # self.rate.sleep()
            
    def tf_listener(
        self,
        tf_from='/map',
        tf_to='/custom',
        save_mat_path='./samples',
        save_mat_name='20220819.txt',
        ):
        
        total_tf = []
        
        while not rospy.is_shutdown():
            try:
                now = rospy.Time(0)
                position, quaternion = self.listener.lookupTransform(
                    tf_from, tf_to, now)
                total_tf.append(
                    [
                        now.to_sec(),
                        position[0],
                        position[1], 
                        position[2],
                        quaternion[0],
                        quaternion[1],
                        quaternion[2],
                        quaternion[3],
                    ]
                )
                # print(total_tf)
                mdic = {'tf': np.array(total_tf)}
                sio.savemat('%s/%s' % (save_mat_path, save_mat_name), mdic)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            
            rospy.sleep(self.duration)
            
        # mdic = {"tf": np.array(total_tf)}
        # sio.savemat('%s/%s' % (save_mat_path, save_mat_name), mdic)
            
    def pulish_lidar_frame_and_save_tf(
        self, 
        start_frame, 
        max_frame,
        tf_from='/map',
        tf_to='/custom',
        save_mat_path='./samples',
        save_mat_name='20220926.mat',
        get_tf_and_save=True
    ):
        cur_time = rospy.Time.from_sec(0.0)

        total_tf = []

        for i in tqdm(range(start_frame, max_frame, self.index_interval)):

            # print(cur_time)

            _, _, points = self.read_each(i)
            cur_imu_quaternion = self.read_each_imu_as_quaternion(i)

            clock_msg = Clock()
            clock_msg.clock = cur_time

            imu_msg = Imu()
            imu_msg.header.stamp = cur_time
            imu_msg.header.frame_id = self.lidar_frame_id
            # imu_msg.orientation = cur_imu_quaternion
            imu_msg.orientation.x = cur_imu_quaternion[0]
            imu_msg.orientation.y = cur_imu_quaternion[1]
            imu_msg.orientation.z = cur_imu_quaternion[2]
            imu_msg.orientation.w = cur_imu_quaternion[3]
            imu_msg.angular_velocity = self.create_vector3(
                0., 0., 0.)

            cloud_msg = self.open3d2pointcloud(cur_time, points)

            # if pub.get_num_connections():
            self.imu_pub.publish(imu_msg)
            self.lidar_pub.publish(cloud_msg)
            self.clock_pub.publish(clock_msg)

            # rospy.sleep(self.duration)
            # self.rate.sleep()
            
            cur_time += self.duration
            
            if get_tf_and_save:

                ## Save to mat
                try:
                    time_stamp = self.listener.getLatestCommonTime(
                        tf_from, tf_to)  # This does return the sim time
                    # print('tf time_stamp: ', time_stamp)
                    position, quaternion = self.listener.lookupTransform(
                        tf_from, tf_to, time_stamp)
                    total_tf.append(
                        [
                            time_stamp.to_sec(),
                            position[0],
                            position[1],
                            position[2],
                            quaternion[0],
                            quaternion[1],
                            quaternion[2],
                            quaternion[3],
                        ]
                    )
                    # print(total_tf)
                    mdic = {'tf': np.array(total_tf)}
                    sio.savemat('%s/%s' % (save_mat_path, save_mat_name), mdic)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print('Fail to get TF!')
                    continue
            
            
    def pulish_lidar_frame_and_save_tf_seperated(
        self,
        tf_from='/map',
        tf_to='/custom',
        save_mat_path='./samples',
        save_mat_name='20220926',
    ):

        total_num = 0

        for i in tqdm(range(0, len(self.lists_by_dist))):

            cur_time = rospy.Time.from_sec(0.0)
            
            total_tf = []

            for j in tqdm(
                range(0, len(self.lists_by_dist[i]), self.index_interval),
                leave=False
                ):

                _, _, points = self.read_each(total_num)

                cloud_msg = self.open3d2pointcloud(cur_time, points)

                clock_msg = Clock()
                clock_msg.clock = cur_time

                # if pub.get_num_connections():
                self.clock_pub.publish(clock_msg)
                self.lidar_pub.publish(cloud_msg)

                total_num += self.index_interval
                cur_time += self.duration

                ## Save to mat
                try:
                    time_stamp = self.listener.getLatestCommonTime(
                        tf_from, tf_to)  # This does return the sim time
                    # print('tf time_stamp: ', time_stamp)
                    position, quaternion = self.listener.lookupTransform(
                        tf_from, tf_to, time_stamp)
                    total_tf.append(
                        [
                            time_stamp.to_sec(),
                            position[0],
                            position[1],
                            position[2],
                            quaternion[0],
                            quaternion[1],
                            quaternion[2],
                            quaternion[3],
                        ]
                    )
                    # print(total_tf)
                    mdic = {'tf': np.array(total_tf)}
                    sio.savemat('%s/%s_%d.mat' % (save_mat_path, save_mat_name, i), mdic)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print('Fail to get TF!')
                    continue
            
    def make_mp4_from_frames(
        self, 
        max_frame,
        img_size,
        save_result_path,
        save_name='test.mp4'
        ):
        
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # mp4
        out = cv2.VideoWriter(
            '%s/%s' % (save_result_path, save_name), fourcc, 30.0, img_size)
        
        for i in tqdm(range(max_frame)):
            
            img, _, _ = self.read_each(i)
            # print(np.shape(img))
            
            out.write(img[:, :, :3])
            # cv2.imshow('video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        out.release()
        cv2.destroyAllWindows()

    def adjust_offset(self, input):
        input_array = np.array(input)

        if self.hd_map_offset is not None:
            return input_array - self.hd_map_offset
        else:
            return input_array

