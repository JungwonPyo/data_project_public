import numpy as np
import os
# import six.moves.urllib as urllib

import zipfile

from distutils.version import StrictVersion
from io import StringIO
from PIL import Image
import json

import time
import rospy
import cv2
import pcl_ros
import ros_numpy
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image as Sensor_Image
from sensor_msgs.msg import CompressedImage
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import CameraInfo
import message_filters

# from .vidstab.VidStab import VidStab

class image_receiver:

    def __init__(
        self, 
        camera_type, 
        camera_num=None, 
        camera_parameter_path=None, 
        smoothing_window=3, 
        border_size=0,
        depth_enabled=False
        ):

        self.bridge = CvBridge()

        self.rgb_input = None
        self.rgb_left_input = None
        self.rgb_right_input = None
        self.depth_input = None
        self.cloud = None
        self.camera_p = None

        self._camera_type = camera_type
        self.camera_num = camera_num

        # self.stabilizer = VidStab()

        self.smoothing_window = smoothing_window
        self.border_size = border_size

        self.depth_enabled = depth_enabled

        if self._camera_type == 'zed':
            self.camera_info = rospy.Subscriber(
                "/zed2/zed_node/rgb/camera_info", CameraInfo, self.info_left)
        elif self._camera_type == 'zed2i':
            self.camera_info = rospy.Subscriber(
                "/zed2i/zed_node/rgb/camera_info", CameraInfo, self.info_left)
        elif self._camera_type == 'intel':
            self.camera_info = rospy.Subscriber(
                "/camera/color/camera_info", CameraInfo, self.info_left)
        elif self._camera_type == 'imx390':
            self.camera_info = rospy.Subscriber(
                "/csi_cam_%d/camera_info" % (self.camera_num), CameraInfo, self.info_left)

        if camera_parameter_path is not None:
            self.map1, self.map2 = self.undistortion_maps(camera_parameter_path)

        self.run()

    def undistortion_maps(self, path):
        with open(path, "r") as cal_json:
            cal_json_val = json.load(cal_json)
        dim1 = tuple(cal_json_val['dim1'])
        dim2 = tuple(cal_json_val['dim2'])
        dim3 = tuple(cal_json_val['dim3'])
        K = np.array(cal_json_val['K'])
        D = np.array(cal_json_val['D'])
        new_K = np.array(cal_json_val['new_K'])
        scaled_K = np.array(cal_json_val['scaled_K'])
        balance = cal_json_val['balance']

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

        return map1, map2

    def get_undistorted_image(self, image, resize=None):
        # Undistort Input Image
        temp_image = cv2.remap(image, self.map1, self.map2,
                               interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if resize is not None:
            temp_image = cv2.resize(temp_image, resize)

        return temp_image

    def imx_callback(self, data):
        try:
            raw_input_image = np.frombuffer(
                data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            self.rgb_input = self.get_undistorted_image(
                raw_input_image, None)[:, :, (2, 1, 0)]
            # if self.smoothing_window > 0:
            # 	self.rgb_input = self.stabilizer.stabilize_frame(
            # 		input_frame=self.rgb_input, smoothing_window=self.smoothing_window, border_size=self.border_size)

        except CvBridgeError as e:
              print(e)

    def rgb_callback(self, data):
        self.rgb_input = np.frombuffer(
            data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    # self.rgb_input = self.bridge.imgmsg_to_cv2(data, "rgb8")
    # np_arr = np.fromstring(data.data, np.uint8)
    # self.rgb_input = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    def depth_callback(self, data):
        self.depth_input = np.frombuffer(
            data.data, dtype=np.float32).reshape(data.height, data.width, -1)

    # print(data.height, data.width)
    # self.depth_input = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    def sync_callback_zed(self, rgb_data, depth_data):
        try:
            self.rgb_input = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(
                rgb_data.height, rgb_data.width, -1)
            self.depth_input = np.frombuffer(depth_data.data, dtype=np.float32).reshape(depth_data.height,
                                                                                        depth_data.width, -1)
        except CvBridgeError as e:
            print(e)

    def sync_callback_realsense(self, rgb_data, depth_data):
        try:
            self.rgb_input = self.bridge.imgmsg_to_cv2(rgb_data, "rgb8")
            self.depth_input = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")

        except CvBridgeError as e:
            print(e)

        self.rgb_input = np.array(self.rgb_input, dtype=np.uint8)
        self.depth_input = np.array(self.depth_input, dtype=np.float32)

    def sync_callback_stereo(self, rgb_left_data, rgb_right_data, depth_data):
        try:
            self.rgb_left_input = self.bridge.imgmsg_to_cv2(rgb_left_data, "rgb8")
            self.rgb_right_input = self.bridge.imgmsg_to_cv2(rgb_right_data, "rgb8")
            self.depth_input = self.bridge.imgmsg_to_cv2(depth_data, "passthrough")
        except CvBridgeError as e:
            print(e)

        self.rgb_left_input = np.array(self.rgb_left_input, dtype=np.uint8)
        self.rgb_right_input = np.array(self.rgb_right_input, dtype=np.uint8)
        self.depth_input = np.array(self.depth_input, dtype=np.float32)

    def stereo_cloud(self, data):
        """ Callback to process the point cloud """
        self.cloud = data

    def info_left(self, data):
        """ Camera Info """
        self.camera_p = np.matrix(data.P, dtype='float64')
        self.camera_p.resize((3, 4))
        self.camera_info.unregister()

    def ros_to_pcl(self, ros_cloud):
        """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

            Args:
                ros_cloud (PointCloud2): ROS PointCloud2 message

            Returns:
                pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
        """
        points_list = []

        for data in pc2.read_points(ros_cloud, skip_nans=True):
            points_list.append([data[0], data[1], data[2], data[3]])

        pcl_data = pcl_ros.PointCloud_PointXYZRGB()
        pcl_data.from_list(points_list)

        return pcl_data

    def uv_to_xyz(self, uv, depth):
        ## uv can be [N, 2]
        ## depth has to be [N, 1]
        # print(np.shape(uv), np.shape(depth))
        real_x = (uv[:, 0] - self.camera_p[0, 2]) * depth / self.camera_p[0, 0]
        real_y = (uv[:, 1] - self.camera_p[1, 2]) * depth / self.camera_p[1, 1]
        # print(np.shape(real_x), np.shape(real_y), np.shape(depth))

        return np.concatenate((np.expand_dims(real_x, axis=1),
                               np.expand_dims(real_y, axis=1),
                               np.expand_dims(depth, axis=1)), axis=1)

    def run(self):
        
        if self._camera_type == 'imx390':
            rospy.init_node('image_converter_%d' % (self.camera_num), anonymous=False)
        else:
            rospy.init_node('Image_receiver', anonymous=True)

        image_sub = None
        depth_sub = None

        if self._camera_type == 'zed' or self._camera_type == 'zed2' or self._camera_type == 'zed2i':
            if self.depth_enabled:
                image_sub = message_filters.Subscriber(
                    "/%s/zed_node/rgb/image_rect_color" % (self._camera_type), Sensor_Image)
                depth_sub = message_filters.Subscriber(
                    "/%s/zed_node/depth/depth_registered" % (self._camera_type), Sensor_Image)
            else:
                image_sub = rospy.Subscriber(
                    "/%s/zed_node/rgb/image_rect_color" % (self._camera_type), Sensor_Image, self.rgb_callback, queue_size=10)
            print('%s Selected!' % (self._camera_type))
        elif self._camera_type == 'intel':
            image_sub = message_filters.Subscriber(
                "/camera/color/image_raw", Sensor_Image)
            depth_sub = message_filters.Subscriber(
                "/camera/aligned_depth_to_color/image_raw", Sensor_Image)
            print('Realsens Selected!')
        elif self._camera_type == 'imx390':
            image_sub = rospy.Subscriber(
                "/csi_cam_%d/image_raw" % (self.camera_num), Sensor_Image, self.imx_callback, queue_size=10)
            print('IMX390 Selected!')
        else:
            print('Set camera again!')

        if self._camera_type == 'zed' and self.depth_enabled:
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [image_sub, depth_sub], 10, 0.5, allow_headerless=True)
            self.ts.registerCallback(self.sync_callback_zed)
            # self.ts.registerCallback(self.sync_callback_stereo)
        elif self._camera_type == 'zed2i' and self.depth_enabled:
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [image_sub, depth_sub], 10, 0.5, allow_headerless=True)
            self.ts.registerCallback(self.sync_callback_zed)
        elif self._camera_type == 'intel' and self.depth_enabled:
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [image_sub, depth_sub], 10, 0.5, allow_headerless=True)
            self.ts.registerCallback(self.sync_callback_realsense)

        # rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Sensor_Image, self.rgb_callback)
        # rospy.Subscriber("/zed/zed_node/depth/depth_registered", Sensor_Image, self.depth_callback)
