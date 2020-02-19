import rosbag
import numpy as np
from PIL import Image
import os
import sensor_msgs.point_cloud2 as pc2
import datetime
import cv2

zed_camera_matrix = [[699.8670043945312, 0.0, 603.5809936523438],
                     [0.0, 699.8670043945312, 332.77801513671875],
                     [0.0, 0.0, 1.0]]
zed_dist = [-0.171875, 0.02449920028448105, 0.0, 0.0, 0.0]
zed_camera_matrix = np.array(zed_camera_matrix)
zed_dist = np.array(zed_dist)

bag = rosbag.Bag('/media/ash/OS/IIIT_Labels/rectified_img_bag/nilgiri_1.bag')
data_dir='/media/ash/OS/IIIT_Labels/playground_folder/file_1/'

# os.makedirs(os.path.join(data_dir,"image"))
os.makedirs(os.path.join(data_dir,"velodyne"))

velodyne_stamps = []
img_stamps = []
index = 0

for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
    velodyne_stamps.append(msg.header.stamp)
    gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z","intensity","ring"))
    gen = list(gen)
    gen = np.asarray(zip(*gen),dtype=np.float32).transpose()
    np.save(os.path.join(data_dir,"velodyne/" + '{0:010d}.npy'.format(index)),gen)
    index += 1

velodyne_stamps = np.array(velodyne_stamps)
index = 0
"""
for topic,msg,t in bag.read_messages(topics=['/zed/zed_node/left/image_rect_color']):
    img_stamps.append(msg.header.stamp)

img_stamps = np.array(img_stamps)
time_diff = np.array([abs(elem-img_stamps) for elem in velodyne_stamps])
synced_img_stamps = img_stamps[np.argmin(time_diff,axis=1)]

for topic,msg,t in bag.read_messages(topics=['/zed/zed_node/left/image_rect_color']):
    if msg.header.stamp in synced_img_stamps:
        img = np.fromstring(msg.data,dtype=np.uint8).reshape((720,1280,4))
        img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
        #Run only for raw images
        ########
        #img = cv2.undistort(img,cameraMatrix=zed_camera_matrix,distCoeffs=zed_dist)
        ########
        img = Image.fromarray(img)
        img.save(os.path.join(data_dir,"image/" + '{0:010d}.png'.format(index)))
        index += 1
"""
bag.close()
