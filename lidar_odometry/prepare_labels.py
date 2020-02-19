import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import time


def read_txt(path):
    with open(path, 'r') as f:
        rows = f.read().split('\n')[:-1]
        values = [row.split(' ')[:-1] for row in rows]
        transform_matrix = np.array(values, dtype=np.float)
        return transform_matrix


def rotate_axis(inp):
    hacky_trans_matrix = R.from_euler('xyz', [1.57, -1.57, 0]).as_dcm()
    hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.zeros(3)[:, np.newaxis]), axis=1)
    hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)
    return np.dot(hacky_trans_matrix, inp.transpose())


def project_lid_on_img(lid_pt, T, p):
    tran_pt = np.dot(T, lid_pt)
    proj_lid_pt = np.dot(p, tran_pt).reshape(3, -1)
    pix = np.array([proj_lid_pt[0] / proj_lid_pt[2], proj_lid_pt[1] / proj_lid_pt[2]]).reshape(2, -1)
    return pix


projection_matrix = np.array([[692.653256, 0.000000, 629.321381, 0.000],
                            [0.000, 692.653256, 330.685425, 0.000],
                            [0.000000, 0.000000, 1.00000, 0.000]])

dir_path = "/media/ash/OS/IIIT_Labels/val/vindhya_1"
labels_path = os.path.join(dir_path,"labels")
odom_path = os.path.join(dir_path,"odometry")
img_path = os.path.join(dir_path,"image")
pointcloud_path = os.path.join(dir_path,"velodyne")
transform_matrix = read_txt('../combined_transf_3.txt')


def get_breakpoints(pts):
    diff_log = []
    length = pts.shape[0]
    pred = np.zeros(length)
    new_pred = np.zeros(length)

    for i in range(2,length):
        d_i_1 = np.linalg.norm(pts[i-1,:3])
        d_i_2 = np.linalg.norm(pts[i-2,:3])
        d_i = np.linalg.norm(pts[i,:3])
        gamma_1 = np.dot(pts[i-1,:3],pts[i-2,:3])/(d_i_1*d_i_2)
        gamma_2 = np.dot(pts[i - 1, :3], pts[i, :3]) / (d_i_1 * d_i)
        gamma = (gamma_1 + gamma_2)/2
        d_p = (d_i_1*d_i_2)/(2*d_i_1*gamma-d_i_2)
        diff = d_i-d_p

        if 0.4 < diff < 1:
            pred[i] = 1
        elif -1 < diff < -0.4:
            pred[i] = -1
        diff_log.append(diff)

    min_segment = 1
    segments = []
    for i in range(length):
        if pred[i] == -1:
            obs_start = i
            obs_end = 0
            end_range = i+11 if i+11 < length else length
            for j in range(i+1,end_range):
                if pred[j] == 1 and j-i > min_segment:
                    obs_end = j
                    break
            if obs_start != 0 and obs_end != 0:
                segments.append((obs_start,obs_end))

    for start,end in segments:
        new_pred[start:end] = -1
    return new_pred


def get_odometry(path):
    try:
        odom = np.load(path)
    except IOError:
        return None
    t_vec, quat = np.array(odom[0]), np.array(odom[1])
    r_mat = R.from_quat(quat)
    r_mat = r_mat.as_dcm()
    t_vec = np.array(t_vec)
    t_vec = t_vec[:, np.newaxis]
    RnT = np.concatenate((r_mat, t_vec), axis=1)
    RnT = np.concatenate((RnT, np.array([[0, 0, 0, 1]])), axis=0)
    return RnT


if __name__ == "__main__":

    files = sorted(os.listdir(pointcloud_path))
    for i,file_name in enumerate(files):
        if not os.path.exists(os.path.join(labels_path,file_name.split('.npy')[0]+'.png')):
            continue

        img = cv2.imread(os.path.join(img_path, file_name.split('.npy')[0]+'.png'))
        label = Image.open(os.path.join(labels_path,file_name.split('.npy')[0]+'.png'))
        label = np.array(label)
        label[label >= 2] = 2

        trans_1_matrix = get_odometry(os.path.join(odom_path, file_name))
        context_points = np.array([])
        context_pred = np.array([])
        depth = np.zeros((720,1280),dtype=np.float16)

        for j in range(i-5,i+5):
            # Loading up everything
            if 0 <= j < len(files):
                points = np.load(os.path.join(pointcloud_path, files[j]))
            else:
                continue
            # Transforming Point-Cloud to NED frame through rotation
            ring_num = points[:,4]
            points = points[:, :4]
            points[:, 3] = 1.0
            points = rotate_axis(points).transpose()
            front_points = points[:,2] > 0
            points = points[front_points]
            ring_num = ring_num[front_points]
            points = points.transpose()

            if j != i:
                trans_2_matrix = get_odometry(os.path.join(odom_path, files[j]))
                if trans_1_matrix is not None and trans_2_matrix is not None:
                    trans_matrix = np.matmul(np.linalg.inv(trans_1_matrix), trans_2_matrix)
                    points = np.matmul(trans_matrix, points)
                else:
                    continue

            project_points = project_lid_on_img(points, transform_matrix, projection_matrix)
            project_points = project_points.transpose()
            points = points.transpose()

            for ring_id in range(8):
                proj_pts = project_points[ring_num == ring_id]
                ring_pts = points[ring_num == ring_id]
                valid_indexes = []
                for k, pt in enumerate(proj_pts):
                    x, y = int(pt[0]), int(pt[1])
                    if (0 < x < 1280) and (0 < y < 720):
                        depth[y,x] = np.linalg.norm(ring_pts[k,:3])
                        valid_indexes.append(k)

                ring_pts = ring_pts[valid_indexes]
                proj_pts = proj_pts[valid_indexes]
                # pred = get_breakpoints(ring_pts)
                if context_points.shape[0]:
                    context_points = np.concatenate((context_points,proj_pts),axis=0)
                    # context_pred = np.append(context_pred,pred,axis=0)
                else:
                    context_points = proj_pts
                    # context_pred = pred

        label_mask = (label == 2).astype(np.float32)
        # depth = depth.astype(int)
        # label_mask = label_mask & depth
        label_mask = label_mask.astype(np.float32)
        label_mask = cv2.GaussianBlur(label_mask,(13,13),5)
        # label_mask = cv2.blur(label_mask,(17,17))
        valid_regions = label_mask > 0
        label_mask[valid_regions] = 1
        label_mask = 100*label_mask
        # label_mask[valid_regions] = 1 - label_mask[valid_regions]
        # label_mask = label_mask / np.max(label_mask)
        # label_mask[label==2] = 1
        depth_mask = depth>0
        label_mask[depth_mask] = depth[depth_mask]
        plt.imshow(label_mask)
        plt.show()
        # label_mask[label_mask>0] = 255
        # label_mask = label_mask.astype(np.uint8)
        # for k, pt in enumerate(context_points):
        #     x, y = int(pt[0]), int(pt[1])
        #     if context_pred[k] == -1 and label[y,x] != 0:
        #         pt_color = (0, 0, 255)
        #     else:
        #         pt_color = (255,0,0)
        #     cv2.circle(img, (x, y), 1, pt_color, thickness=1)

        # region_prop = np.clip(region_prop,0,1)
        # region_prop = 255*region_prop
        # region_prop = region_prop.astype(np.uint8)
        # region_prop = cv2.applyColorMap(region_prop,colormap=cv2.COLORMAP_JET)
        # cv2.imshow('feed',region_prop)
        # root_path = os.path.join(labels_path.split('labels')[0],'context_full')
        # if not os.path.exists(root_path):
        #     os.makedirs(root_path)
        # np.save(os.path.join(root_path,file_name),region_prop)

        # cv2.imshow("label",label_mask)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break
        # time.sleep(0.05)
        # cv2.waitKey(0)