import sys
sys.path.append('../')
from PIL import Image
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
from lidar_odometry.gt_from_loam import read_odometry
from matplotlib import pyplot as plt

path = '/Users/aditya/Documents/Code/aditya/iiit_research/iiit_data/debug/'
img_path = os.path.join(path,"image")
velo_path = os.path.join(path,"velodyne")
dense_velo_path = os.path.join(path,"groundTruth_denser")
sparse_velo_path = os.path.join(path,"depth")
ring_info = os.path.join(path, "rings")


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

"""Rotating velodyne point cloud to same axis as Odometry  """
hacky_trans_matrix = R.from_euler('xyz',[1.57,-1.57,0]).as_dcm()
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.zeros(3)[:,np.newaxis]),axis=1)
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.array([[0,0,0,1]])),axis=0)

rot_vec = transform_matrix[:3,:3]
trans_vec= transform_matrix[:3,3]

"""
# For file_1 and file_4
# we modify the RnT matrix manually below

dump_vec = R.from_dcm(rot_vec)
dump_vec = dump_vec.as_rotvec()
print("Rot before",dump_vec)
dump_vec[0] += 0.01
print("Rot after",dump_vec)
new_rot_vec = R.from_rotvec(dump_vec)
new_rot_vec = new_rot_vec.as_dcm()
print("Trans before",trans_vec)
trans_vec[0] += 0.25
print("Trans after",trans_vec)
"""


def read_velodyne(src_path):
    frames = []
    for file in sorted(os.listdir(src_path)):
        points = np.load(os.path.join(velo_path, file))
        points = points.transpose()
        points[3,:] = 1                         # Convert to homogeneous coordinates
        points = points[:4,:]
        points = np.dot(hacky_trans_matrix,points)
        points = points.transpose()[:,:4]
        frames.append(points)
    return frames


def get_transf_mat(odom1,odom2):
    transf1 = make_transf(odom1)
    transf2 = make_transf(odom2)
    transf_mat = np.matmul(np.linalg.inv(transf1),transf2)
    return transf_mat


def make_transf(odom):
    trans,quat = odom
    r_mat = R.from_quat(quat)
    r_mat = r_mat.as_dcm()
    trans = np.array(trans)
    t_vec = trans[:, np.newaxis]

    trans_matrix = np.concatenate((r_mat, t_vec), axis=1)
    trans_matrix = np.concatenate((trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)
    return trans_matrix


def shift_cloud(points,transf):
    shifted_points = points.copy()
    shifted_points = shifted_points.transpose()
    shifted_points = np.matmul(transf,shifted_points)
    shifted_points = shifted_points.transpose()
    return shifted_points


def densify_cloud(frames,odom,num_frames):
    dense_frames = []

    for i in range(num_frames,len(frames)-num_frames):
        acc_cloud = frames[i]
        for j in range(-num_frames, num_frames+1):
            if i == j:
                continue
            transf_mat = get_transf_mat(odom[i], odom[i + j])
            proj_cloud = shift_cloud(frames[i + j], transf_mat)
            acc_cloud = np.concatenate((acc_cloud, proj_cloud), axis=0)

        dense_frames.append(acc_cloud)
    return dense_frames


def vis_calib(frames,start,end,num_frames):
    img_files = sorted(os.listdir(img_path))[start+num_frames:end-num_frames]
    assert len(img_files) == len(frames)
    for k in range(len(frames)):
        img = cv2.imread(os.path.join(img_path,img_files[k]))
        dense_cloud = frames[k]
        dense_cloud = dense_cloud[:,:3]
        rot, _ = cv2.Rodrigues(rot_vec)
        proj_points, _ = cv2.projectPoints(dense_cloud, rot, trans_vec, projection_matrix, distortion_matrix)

        for i in range(proj_points.shape[0]):
            x = int(proj_points[i][0][0])
            y = int(proj_points[i][0][1])
            depth = np.sqrt(pow(dense_cloud[i][0], 2) + pow(dense_cloud[i][1], 2) + pow(dense_cloud[i][2], 2))
            if (y < 720 and x < 1280 and x > 0 and y > 0 and depth <= 10 and dense_cloud[i][2] >= 0 and depth > 2.5):
                hsv = np.zeros((1, 1, 3)).astype(np.uint8)
                hsv[:, :, 0] = int((depth)/(10) * 159)
                hsv[0, 0, 1] = 255
                hsv[0, 0, 2] = 200
                hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.circle(img, (x, y), 2, color=(int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])), thickness=1)
        cv2.imshow("window", img[50:512,280:1000])
        cv2.waitKey(10)
    cv2.destroyAllWindows()


def save_frames(frames,start,num_frames):
    index = start
    for k in range(len(frames)):
        img = np.zeros((720,1280,1),dtype=np.uint16)
        ring = np.zeros((720, 1280, 1), dtype=np.uint8)
        point_cloud = frames[k]
        dense_cloud = point_cloud[:,:3]
        rot, _ = cv2.Rodrigues(rot_vec)
        proj_points, _ = cv2.projectPoints(dense_cloud, rot, trans_vec, projection_matrix, distortion_matrix)

        for i in range(proj_points.shape[0]):
            x = int(proj_points[i][0][0])
            y = int(proj_points[i][0][1])
            depth = np.sqrt(pow(dense_cloud[i][0], 2) + pow(dense_cloud[i][1], 2) + pow(dense_cloud[i][2], 2))
            depth = np.clip(depth,a_min=0,a_max=100)
            if (y < 720 and x < 1280 and x > 0 and y > 0 and dense_cloud[i][2] >= 0 and depth > 2.5):
                depth = int(depth/100*65535)
                img[y,x] = depth
                ring[y,x] = point_cloud[i, 4]

        cv2.imwrite(os.path.join(sparse_velo_path,'{0:010d}.png'.format(index)),img)
        cv2.imwrite(os.path.join(ring_path, '{0:010d}.png'.format(index)), ring)
        index += 1


if __name__ == "__main__":

    """
    
    Densify Point Cloud by accumulating frames
    Frames accumulated for any Nth frame: [N-num_frame,N+num_frame]
    Hence densified cloud has (2*num_frames + 1) point clouds  
    """

    if not os.path.exists(sparse_velo_path):
        os.mkdir(sparse_velo_path)
    num_frames = 5
    odometry,a,b,c,_ = read_odometry('/integrated_to_init')

    len_odom = len(odometry)
    start = 10                                               # Frames discarded for which odometry not available
    end = len_odom + 10

    sparse_frames = read_velodyne(velo_path)
    sparse_frames = sparse_frames[start:end]
    #assert len(sparse_frames) == len_odom

    #dense_frames = densify_cloud(sparse_frames,odometry,num_frames)
    #print(len_odom, len(sparse_frames),len(dense_frames))
    # vis_calib(dense_frames,start,end,num_frames)
    save_frames(sparse_frames,start,num_frames)

    """
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(range(len(a)), a)
    ax.set_xlabel('X Label')
    ax = fig.add_subplot(132)
    ax.plot(range(len(b)), b)
    ax.set_xlabel('Y Label')
    ax = fig.add_subplot(133)
    ax.plot(range(len(c)), c)
    ax.set_xlabel('Z Label')
    plt.show()
    """
