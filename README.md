# 52-1 Models for Parking Dataset

##  Used Models

- [RGBX_Semantic_Segmentation](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
- [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [hdl_graph_slam](https://github.com/koide3/hdl_graph_slam)


##  Preparation

### Environment

- Multi-core CPU (at least 8 CPUs, Threading)
- Nvidia GPU (more than 24GB memory)
- 16GB+ system memory
- Ubuntu 20.04

### Software

- Python 3.8
- [Pytorch 1.8+ (with torch vision) for GPU](https://pytorch.org/get-started/previous-versions/)
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [Open3D](https://github.com/isl-org/Open3D)
- [OpenCV 4.6 (with contrib.)](https://github.com/opencv/opencv/releases)


Note that Pytorch and ROS Noetic are neede to be installed seperately. 

## Get This Github

```bash
git clone https://github.com/JungwonPyo/data_project
cd data_project
```

## Install pip requiremments

```shell
pip install -r requirements.txt
```

## Pretraining Model

See this [Google Drive](https://drive.google.com/drive/folders/1Ixb7x1aahbu59bq9WGKosh5oGJAwcJnn?usp=sharing), downlaod all files and put these in 'checkpoints' folder.

Details for HRNet-Semantic-Segmentation, see [this](https://github.com/HRNet/HRNet-Semantic-Segmentation)

Details for RGBX_Semantic_Segmentation, see [this](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5)


## HRNet-Semantic-Segmentation

To change training configs, see [this](configs/hrnet_custom_train.yaml)

### Training

```bash
python train_hrnet.py --cfg ./configs/hrnet_custom_train.yaml
```

To change testing configs, see [this](configs/hrnet_custom_test.yaml)

### Testing

```bash
python test_hrnet.py --cfg ./configs/hrnet_custom_test.yaml
```

## RGBX_Semantic_Segmentation

To change training configs, see [this](configs/cmx_config_custom.yaml)
### Training

```bash
python train_cmx.py
```

To change testing configs, see [this](configs/cmx_config_custom.py)

### Testing

```bash
python test_cmx.py
```


## hdl_graph_slam
### Installation

```bash
# for noetic
sudo apt-get install ros-noetic-geodesy ros-noetic-pcl-ros ros-noetic-nmea-msgs ros-noetic-libg2o

cd catkin_ws/src
git clone https://github.com/koide3/ndt_omp.git
git clone https://github.com/SMRT-AIST/fast_gicp.git --recursive
git clone https://github.com/koide3/hdl_graph_slam

cd .. && catkin_make -DCMAKE_BUILD_TYPE=Release
```

### Run

To perform SLAM and take tf information, 

```bash
roslaunch hdl_graph_slam hdl_graph_slam_custom.launch
python lidar_handler.py
```

## LEGO-LOAM

Referenced problems are [here](https://www.programmersought.com/article/95708989760/)

### Install gtsm

```bash
wget -O ~/Downloads/gtsam.zip https://github.com/borglab/gtsam/archive/4.0.0-alpha2.zip
cd ~/Downloads/ && unzip gtsam.zip -d ~/Downloads/
cd ~/Downloads/gtsam-4.0.0-alpha2/
mkdir build && cd build
cmake ..
make -j
sudo make install
```

### Install libmetics

```bash
# To prevent below error
# .../devel/lib/lego_loam/mapOptmization: error while loading shared libraries: libmetis.so: cannot open shared object file: No such file or directory
sudo apt-get install libmetis-dev
```

### Modifying PCL Headers

```bash
# In PCL Header
pcl/filter/voxel_gird.h
line 340 Eigen::index -> int
line 669 Eigen::index -> int
```

### Change package topic names

```
# For all elements below package
All /camera -> camera
All /camera_init -> camera_init
All /map -> map
```


