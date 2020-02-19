import numpy as np
from PIL import Image
import os
import cv2
from scipy.ndimage import label as make_label
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,HuberRegressor,RANSACRegressor,TheilSenRegressor
import time
from tqdm import tqdm

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

dir_path = "/media/ash/OS/IIIT_Labels/val/vindhya_2/"
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


def get_mask(inp, span=5):
    instance_id, instance_num = make_label(inp)
    obs_centroids = {}
    mask = np.zeros((inp.shape[0], inp.shape[1]))
    for i in range(instance_num):
        x, y = np.where(instance_id == i + 1)
        min_x = np.min(x) - span
        min_y = np.min(y) - span
        max_x = np.max(x) + span
        max_y = np.max(y) + span
        cx = int(np.mean(x))
        cy = int(np.mean(y))
        obs_centroids[i + 1] = [cx, cy]
        mask[min_x:max_x, min_y:max_y] = 1

    return mask


def get_odometry(path):
    try:
        odom = np.load(path,allow_pickle=True)
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

    files = sorted(os.listdir(labels_path))
    all_points = []
    all_pred = []

    for i,file_name in enumerate(tqdm(files)):

        context_points = np.array([])
        context_pred = np.array([])

        img = cv2.imread(os.path.join(img_path,file_name))
        points = np.load(os.path.join(pointcloud_path, file_name.split('.png')[0] + '.npy'),allow_pickle=True)
        label = np.array(Image.open(os.path.join(labels_path,file_name)))

        ring_num = points[:, 4]
        points = points[:, :4]
        points[:, 3] = 1.0
        # Transforming Point-Cloud to NED frame through rotation
        points = rotate_axis(points).transpose()
        front_points = points[:, 2] > 0
        points = points[front_points]
        ring_num = ring_num[front_points]
        points = points.transpose()

        project_points = project_lid_on_img(points, transform_matrix, projection_matrix)
        project_points = project_points.transpose()
        points = points.transpose()

        for ring_id in range(7):
            proj_pts = project_points[ring_num == ring_id]
            ring_pts = points[ring_num == ring_id]
            valid_indexes = []
            for k, pt in enumerate(proj_pts):
                x, y = int(pt[0]), int(pt[ 1])
                if (0 < x < 1280) and (0 < y < 720) and label[y,x] >= 1:
                    valid_indexes.append(k)

            ring_pts = ring_pts[valid_indexes]
            proj_pts = proj_pts[valid_indexes]
            pred = get_breakpoints(ring_pts)
            for k, pt in enumerate(proj_pts):
                x_0, y_0 = int(pt[1]), int(pt[0])
                if pred[k] == -1:
                    pt_color = (0, 0, 255)
                else:
                    pt_color = (0, 255, 0)
                cv2.circle(img, (y_0, x_0), 1, pt_color, 1)
        cv2.imshow("image",img)
        cv2.waitKey(0)

        """
            if context_points.shape[0]:
                context_points = np.concatenate((context_points,ring_pts), axis=0)
                context_pred = np.append(context_pred, pred, axis=0)
            else:
                context_points = ring_pts
                context_pred = pred

        all_points.append(context_points)
        all_pred.append(context_pred)
        """

    """
    for frame_num in range(len(files)):
        # img = cv2.imread(os.path.join(img_path, files[frame_num]))
        region_prop = np.zeros((720, 1280), dtype=np.float16)
        sigma = 5
        span = 15
        for near_frames in range(frame_num,frame_num+1):

            if 0 <= near_frames < len(files):

                label = Image.open(os.path.join(labels_path, files[near_frames]))
                label = np.array(label)
                label[label >= 2] = 2
                class_mask = label == 2
                label = get_mask(class_mask,span=3)

                frame_pts = all_points[near_frames]
                frame_pred = all_pred[near_frames]
                if near_frames != frame_num:
                    trans_1_matrix = get_odometry(os.path.join(odom_path,files[frame_num].split('.png')[0]+'.npy'))
                    trans_2_matrix = get_odometry(os.path.join(odom_path,files[near_frames].split('.png')[0]+'.npy'))
                    # Select only context points
                    indexes = frame_pred == -1
                    frame_pts = frame_pts[indexes]
                    frame_pred = frame_pred[indexes]
                    frame_proj = project_lid_on_img(frame_pts.transpose(), transform_matrix,
                                                    projection_matrix).transpose()
                    valid_indexes = []
                    for k,pt in enumerate(frame_proj):
                        x,y = int(pt[0]),int(pt[1])
                        if 0 < x < 1280 and 0 < y < 720 and label[y,x] == 1:
                            valid_indexes.append(k)

                    frame_pts = frame_pts[valid_indexes]
                    frame_pred = frame_pred[valid_indexes]
                    if trans_1_matrix is not None and trans_2_matrix is not None:
                        trans_matrix = np.matmul(np.linalg.inv(trans_1_matrix),trans_2_matrix)
                        frame_pts = np.matmul(trans_matrix,frame_pts.transpose())
                        frame_pts = frame_pts.transpose()
                    else:
                        continue

                frame_proj = project_lid_on_img(frame_pts.transpose(), transform_matrix,
                                                projection_matrix).transpose()

                for k,pt in enumerate(frame_proj):
                    x_0, y_0 = int(pt[1]), int(pt[0])
                    if 0 < x_0 < 720 and 0 < y_0 < 1280:
                        if frame_pred[k] == -1:
                        #     pt_color = (0, 0, 255)
                        # else:
                            pt_color = (0,255,0)
                        # cv2.circle(img, (y_0,x_0), 1, pt_color, 1)
                            for x in range(x_0-span,x_0+span+1):
                                for y in range(y_0-span,y_0+span+1):
                                    if 0 < x < 720 and 0 < y < 1280:
                                        region_prop[x,y] += np.exp(-0.5*((x-x_0)**2 + (y-y_0)**2)/sigma**2)

            else:
                continue

        # cv2.imshow("window",img)
        region_prop = np.clip(region_prop,0,1)
        # region_prop = 255*region_prop
        # region_prop = region_prop.astype(np.uint8)
        # region_prop = cv2.applyColorMap(region_prop,colormap=cv2.COLORMAP_JET)
        # cv2.imshow('feed',region_prop)
        root_path = os.path.join(labels_path.split('labels')[0],'context_temporal_new')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        np.save(os.path.join(root_path,files[frame_num].split('.png')[0]+'.npy'),region_prop)
        # cv2.waitKey(0)
        """
