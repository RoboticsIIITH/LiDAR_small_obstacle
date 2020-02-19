import rosbag
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import TransformStamped
import os
import rospy
import numpy as np
import time
import tf_conversions
import tf2_ros
from lidar_odometry.gt_from_loam import read_odometry
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge, CvBridgeError


path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_2/groundTruth"
img_path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_2/image/"
bag = rosbag.Bag('gt_seq2.bag','w')


def talker():
    rospy.init_node("lidar_data",disable_signals=True)
    pub = rospy.Publisher("velodyne_points", PointCloud2,queue_size = 10)
    img_pub = rospy.Publisher("image", Image,queue_size=10)

    msg = PointCloud2()
    odom,a,b,c,_ = read_odometry('/integrated_to_init')
    files = sorted(os.listdir(path))
    img_files = sorted(os.listdir(img_path))[10:len(files)+10]
    assert len(img_files) == len(files), "File size doesn't match"

    for i in range(0,len(files)):
        """Publishing odometry info"""
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "/start"
        t.child_frame_id = "/velodyne"

        trans, rot = odom[i]
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]


        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        br.sendTransform(t)

        """Publishing Point Cloud """
        data = np.load(os.path.join(path, files[i]))

        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "/velodyne"
        msg.header.seq = i

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)]

        msg.height = 1
        msg.width = data.shape[0]
        msg.point_step = 16
        msg.row_step = 16 * data.shape[0]
        data = np.asarray(data, np.float32)
        msg.data = data.tostring()
        pub.publish(msg)

        """Publishing corresponding Image messages"""
        img = cv2.imread(os.path.join(img_path, img_files[i]))
        img = CvBridge().cv2_to_imgmsg(img,encoding="bgr8")
        img_pub.publish(img)

        #bag.write("velodyne_points",msg)
        #bag.write("image_raw",img_msg)
        time.sleep(0.05)
    #bag.close()

if __name__=="__main__":

    try:
        talker()
    except rospy.ROSInterruptException:
        pass







