import rosbag
import numpy as np
from PIL import Image
import os
import sensor_msgs.point_cloud2 as pc2
import datetime
import cv2

bag = rosbag.Bag('/media/ash/OS/IIIT_Labels/lego_loam/seq_4.bag')
output_data_dir='/media/ash/OS/IIIT_Labels/test/seq_4/'

os.makedirs(os.path.join(output_data_dir,"odometry"))


def read_odometry():
    index = 1
    odom=[]
    #x,y,z = [],[],[]
    #t_x,t_y,t_z=[],[],[]

    for topic, msg, t in bag.read_messages(topics="/integrated_to_init"):
        time_stamp = msg.header.stamp

        t_vec = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        r_vec = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        #odom.append((t_vec,r_vec))
        #x.append(r_vec[0])
        #y.append(r_vec[1])
        #z.append(r_vec[2])
        #t_x.append(msg.pose.pose.position.x)
        #t_y.append(msg.pose.pose.position.y)
        #t_z.append(msg.pose.pose.position.z)
        odom = [t_vec,r_vec]
        np.save(os.path.join(output_data_dir,"odometry/" + '{0:010d}.npy'.format(index)),odom)
        index += 1


if __name__ == "__main__":
    read_odometry()
    print("Finished")
