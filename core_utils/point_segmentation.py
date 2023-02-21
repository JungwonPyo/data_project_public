from sklearn.cluster import DBSCAN
import numpy as np
import pandas
import math
from math import pi
import open3d as o3d
import copy
import matplotlib.pyplot as plt

class point_segmentation(object):
    
    def __init__(
        self,
        voxel_size=0.02,
        noise_method='statistical',  # 'statistical' or 'radius'
        nb_neighbors=20,  # for statistical oulier removal
        std_ratio=2.0,  # for statistical oulier removal
        nb_points=16,  # for radius oulier removal
        radius=0.05,  # for radius oulier removal
        x_thres=2.0,
        y_thres=2.0,
        z_thres=2.0,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000,
        total_attept_for_plane=5,
        normal_thres=0.8,
        eps=0.02,
        min_points=10,
        ):
        
        self.voxel_size = voxel_size
        self.noise_method = noise_method
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.nb_points = nb_points
        self.radius = radius
        
        self.x_thres = x_thres,
        self.y_thres = y_thres,
        self.z_thres = z_thres
        
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.total_attept_for_plane = total_attept_for_plane
        self.normal_thres = normal_thres
        
        self.eps = eps
        self.min_points = min_points

    def o3d_noise_filtering(
        self,
        pcd
        ):
        
        voxel_down_pcd = pcd.voxel_down_sample(self.voxel_size)
        
        if self.noise_method == 'statistical':
            print("Statistical oulier removal")
            cl, ind = voxel_down_pcd.remove_statistical_outlier(
                self.nb_neighbors,
                self.std_ratio
                )
        elif self.noise_method == 'radius':
            print("Radius oulier removal")
            cl, ind = voxel_down_pcd.remove_radius_outlier(
                self.nb_points,
                self.radius
                )
        else:
            print("Please select 'statistical' or 'radius'")

        inlier_cloud = voxel_down_pcd.select_by_index(ind)
        # outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)

        return inlier_cloud
        # return self.o3d_remove_points(inlier_cloud)
        
    
    def o3d_remove_points(
        self, pcd, plane_pcd
    ):
        pcd_points = np.asarray(pcd.points)
        # print(np.min(pcd_points[:, 2]), np.max(pcd_points[:, 2]))
        
        plane_pcd_points = np.asarray(plane_pcd.points)
        plane_mean_z = np.mean(plane_pcd_points[:, 2])
        # print(plane_mean_z)
        
        index = np.where(
            (np.absolute(pcd_points[:, 0]) > self.x_thres) &
            (np.absolute(pcd_points[:, 1]) > self.y_thres) &
            (plane_mean_z <= pcd_points[:, 2]) &
            (pcd_points[:, 2] < (plane_mean_z + self.z_thres))
            )
        
        return pcd.select_by_index(index[0])

    def o3d_remove_ground_plane(
        self,
        pcd
        ):
        
        copyed_pcd = copy.deepcopy(pcd)

        for i in range(self.total_attept_for_plane):
            
            # print(np.shape(copyed_pcd.points))

            plane_model, inliers = copyed_pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
                )

            [a, b, c, d] = plane_model
            # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            
            # Check only z axis polynomial 
            if abs(c) >= self.normal_thres:
                # inlier_cloud = copyed_pcd.select_by_index(inliers)
                # inlier_cloud.paint_uniform_color([1.0, 0, 0])
                # outlier_cloud = copyed_pcd.select_by_index(inliers, invert=True)

                # return inlier_cloud, outlier_cloud
                inlier_cloud = copyed_pcd.select_by_index(inliers)
                outlier_cloud = copyed_pcd.select_by_index(inliers, invert=True)
                return inlier_cloud, outlier_cloud
            else:
                copyed_pcd = copyed_pcd.select_by_index(inliers, invert=True)
            
        return False
        
    def o3d_dbscan_clustering(
        self,
        pcd
        ):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(
                    self.eps,
                    self.min_points,
                    print_progress=False
                    ))

        return pcd, labels
    
    def o3d_get_dbscan_result_at_once(self, pcd):
        
        noise_removed_cloud = self.o3d_noise_filtering(pcd)
        
        ground_cloud, ground_removed_cloud = self.o3d_remove_ground_plane(
            noise_removed_cloud)
        
        ground_cell_removed_cloud = self.o3d_remove_points(
            ground_removed_cloud, ground_cloud)

        _, dbscan_labels = self.o3d_dbscan_clustering(
            ground_cell_removed_cloud)
        
        return ground_cell_removed_cloud, dbscan_labels
    
    def o3d_get_dbscan_result_for_non_plane(self, pcd):

        _, dbscan_labels = self.o3d_dbscan_clustering(
            pcd)
        
        return pcd, dbscan_labels
        
    def o3d_get_only_label_points(self, pcd, label, ind):
        
        valid_label_inds = np.where(label == ind)
        
        return pcd.select_by_index(valid_label_inds[0])
        
    def o3d_show_dbscan_result(self, pcd, labels):

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        o3d.visualization.draw_geometries(
            [pcd],
            zoom=0.455,
            front=[-0.4999, -0.1659, -0.8499],
            lookat=[2.1813, 2.0619, 2.0999],
            up=[0.1204, -0.9852, 0.1215]
        )

