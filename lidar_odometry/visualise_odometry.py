import rosbag
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import os
import rospy
import numpy as np
import time
import tf_conversions
import tf2_ros
from lidar_odometry.gt_from_loam import read_odometry,read_cloud,add_intensity_channel
from scipy.spatial.transform import Rotation as R

def talker():
	rospy.init_node("lidar_data",disable_signals=True)
	pub = rospy.Publisher("velodyne_points", PointCloud2, queue_size=100)
	msg = PointCloud2()
	rate = rospy.Rate(10)
	odom,a,b,c = read_odometry('/integrated_to_init')
	data = read_cloud('/laser_cloud_surround')[-1]
	data = add_intensity_channel(data)

	while not rospy.is_shutdown():
		for i in range(0,len(odom)):

			br = tf2_ros.TransformBroadcaster()
			t = TransformStamped()
			t.header.stamp = rospy.Time.now()
			t.header.frame_id = "/start"
			t.child_frame_id = "/odometry"

			trans, quat = odom[i]
			t.transform.translation.x = trans[0]
			t.transform.translation.y = trans[1]
			t.transform.translation.z = trans[2]

			rot = quat
			t.transform.rotation.x = rot[0]
			t.transform.rotation.y = rot[1]
			t.transform.rotation.z = rot[2]
			t.transform.rotation.w = rot[3]
			br.sendTransform(t)


			msg.header.stamp = rospy.Time.now()
			msg.header.frame_id = "/start"
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

			data = np.asarray(data[:,:4],np.float32)
			msg.data = data.tostring()
			pub.publish(msg)
			rate.sleep()


if __name__=="__main__":
	try:
		talker()
	except rospy.ROSInterruptException:
		pass







