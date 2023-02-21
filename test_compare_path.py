import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import math

from core_utils.icp import *

class run_icp(object):

    def __init__(
        self,
        mat_filename,
        csv_filename,
        max_iter=100     
        ):

        self.get_tf = sio.loadmat(mat_filename)['tf'][:, 1:3]
        
        self.csv_pd = pd.read_csv(csv_filename)

        self.max_iter = max_iter

        self.gt_x = self.csv_pd['GT_X'].to_numpy()
        self.gt_y = self.csv_pd['GT_Y'].to_numpy()

        self.gt = np.stack((self.gt_x, self.gt_y), axis=1)

        # gt transpose
        self.gt = self.gt - self.gt[0, :]

        # Make these two's shape as same
        num_tf = np.shape(self.get_tf)[0]
        num_gt = np.shape(self.gt)[0]

        if num_tf >= num_gt:
            self.get_tf = self.get_tf[:num_gt, :]
        else:
            self.gt = self.gt[:num_tf, :]
            
        self.get_tf_dir = self.get_tf[-100, :] - self.get_tf[100, :]
        self.gt_dir = self.gt[-100, :] - self.gt[100, :]
        
        self.diff_rad = self.angle_between(
            self.gt_dir, self.get_tf_dir)
        
        self.get_tf = self.rotate_2d_matrix(
            self.get_tf, self.diff_rad)
        
        # self.get_tf[:,0] = -self.get_tf[:,0]
        # self.get_tf[:,1] = -self.get_tf[:,1]

    def rotation_matrix_2d(self, rad):
        
        c, s = np.cos(rad), np.sin(rad)
        R = np.array(((c, -s), (s, c)))
        
        return R
        
    def rotate_2d_matrix(self, target, rad):
        
        rot = self.rotation_matrix_2d(rad)
        print(np.shape(rot), np.shape(target))
        # return np.transpose(np.matmul(rot, np.transpose(target)))
        return np.matmul(target, np.transpose(rot))
    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)


    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def run(self):

        start = time.time()
        T, distances, iterations, mean_error = icp(
            self.get_tf, 
            self.gt, 
            max_iterations=self.max_iter,
            tolerance=1e-15
            )
        total_time = time.time() - start

        print('icp time:', total_time)
        print('break iteration number: ', iterations)
        print('mean_error: ', mean_error)

        # Make a homogeneous representation of GT
        self.trans_tf = np.ones((np.shape(self.get_tf)[0], 3))
        
        self.trans_tf[:,0:2] = np.copy(self.get_tf)

        # Transform 
        self.trans_tf = np.dot(T, self.trans_tf.T).T
        
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title("Generated")
        ax1.plot(self.get_tf[:, 0], self.get_tf[:, 1])
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title("Ground-Truth")
        ax2.plot(self.gt[:, 0], self.gt[:, 1])
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title("Transformed")
        ax3.plot(self.trans_tf[:, 0], self.trans_tf[:, 1])
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title("Comparison")
        ax4.plot(self.gt[:, 0], self.gt[:, 1], color='red')
        ax4.plot(self.trans_tf[:, 0], self.trans_tf[:, 1], color='blue')

        plt.show()




if __name__ == "__main__":
    data_place = 'office_large'
    slam_type = 'loam'
    mat_filename = '/home/kkk/Pictures/SLAM/%s/%s/%s_loam.mat' %(slam_type,data_place, data_place)

    csv_filename = '/media/kkk/T7_Shield/GeoN_location/%s/csv/%s.csv' %(data_place,data_place)    
    # csv_filename = '/media/kkk/T7_Shield/GeoN_location/mart_large/csv/mart_large.csv'  

    icp_class = run_icp(
        mat_filename=mat_filename,
        csv_filename=csv_filename,
        max_iter=300   
    )

    icp_class.run()
