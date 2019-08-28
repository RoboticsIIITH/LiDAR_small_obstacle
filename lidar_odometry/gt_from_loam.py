import numpy as np
from PIL import Image
import os
import sensor_msgs.point_cloud2 as pc2
import cv2
import rosbag
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

bag = rosbag.Bag('/media/ash/OS/small_obstacle_bag/raw_data/loam_file1.bag')
gt_path = '/media/ash/OS/small_obstacle_bag/synced_data/seq_2/groundTruth/'
image_path = '/media/ash/OS/small_obstacle_bag/synced_data/seq_2/image/'

transform_matrix = [[0.99961240, 0.00960922,-0.02612872,0.257277],
                    [-0.01086974,0.99876225,-0.04853676,-0.0378583],
                    [0.02562997,0.04880196,0.99847958,-0.0483284],
                    [0, 0, 0,1]]

projection_matrix = [[692.653256 ,0.000000, 629.321381],
                     [0.000,692.653256,330.685425],
                     [0.000000,0.000000, 1.00000]]

distortion_matrix = [-0.163186, 0.026619, 0.000410 ,0.000569 ,0.000000]

transform_matrix=np.array(transform_matrix)
projection_matrix=np.array(projection_matrix)
distortion_matrix=np.array(distortion_matrix)

rot_vec = transform_matrix[:3,:3]
trans_vec= transform_matrix[:3,3]

velodyne_stamps =[]
map_clouds = []

def read_cloud(topic):
    index = 0
    for topic, msg, t in bag.read_messages(topics=[topic]):
        velodyne_stamps.append(msg.header.stamp)
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        gen = list(gen)
        gen = np.asarray(zip(*gen)).transpose()
        index += 1
        map_clouds.append(gen)
    return map_clouds


def read_odometry(topic):
    index = 0
    odom=[]
    x,y,z = [],[],[]
    t_x,t_y,t_z=[],[],[]
    cloud_index= []

    for topic, msg, t in bag.read_messages(topics=[topic]):
        time_stamp = msg.header.stamp

        if len(velodyne_stamps):
            nearest_cloud = [abs(time_stamp - elem) for elem in velodyne_stamps]
            nearest_cloud = np.argmin(nearest_cloud)
            cloud_index.append(nearest_cloud)

        t_vec = [msg.pose.pose.position.x , msg.pose.pose.position.y , msg.pose.pose.position.z]
        r_vec = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w]
        odom.append((t_vec,r_vec))
        x.append(r_vec[0])
        y.append(r_vec[1])
        z.append(r_vec[2])
        t_x.append(msg.pose.pose.position.x)
        t_y.append(msg.pose.pose.position.y)
        t_z.append(msg.pose.pose.position.z)
        index += 1

    return odom,t_x,t_y,t_z,cloud_index


def shift_cloud(points,transf):
    shifted_points = points.copy()
    """"
    for i in range(points.shape[0]):
        if points[i,2] >= 0:
            shifted_points.append(points[i])
    shifted_points = np.array(shifted_points).transpose()
    """
    shifted_points = shifted_points.transpose()
    shifted_points[3,:] = 1                                          # Convert to homogeneous coordinates
    shifted_points = np.matmul(np.linalg.inv(transf),shifted_points)
    shifted_points = shifted_points.transpose()
    return shifted_points


def add_intensity_channel(cloud):
    cloud[:,3] = np.sqrt(pow(cloud[:,0],2) + pow(cloud[:,1],2) + pow(cloud[:,2],2))
    return cloud


if __name__ == "__main__":

    cloud_map = read_cloud('/laser_cloud_surround')
    odom,x,y,z,cloud_indexes = read_odometry('/integrated_to_init')
    index = 0
    img_files = sorted(os.listdir(image_path))[10:len(odom)+10]
    assert len(img_files) == len(odom)
    for t_vec,quat in odom:
        r_mat = R.from_quat(quat)
        r_mat = r_mat.as_dcm()
        t_vec = np.array(t_vec)
        t_vec = t_vec[:, np.newaxis]

        trans_matrix = np.concatenate((r_mat, t_vec), axis=1)
        trans_matrix = np.concatenate((trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)

        point_cloud = cloud_map[-1]
        point_cloud = shift_cloud(point_cloud, trans_matrix)
        point_cloud = add_intensity_channel(point_cloud)

        np.save(os.path.join(gt_path, '{0:010d}.npy'.format(index)), point_cloud)
        index += 1

    #cv2.destroyAllWindows()
    bag.close()
    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(range(len(x)),x)
    ax.set_xlabel('X Label')
    ax = fig.add_subplot(132)
    ax.plot(range(len(y)),y)
    ax.set_xlabel('Y Label')
    ax = fig.add_subplot(133)
    ax.plot(range(len(z)),z)
    ax.set_xlabel('Z Label')
    plt.show()
    """