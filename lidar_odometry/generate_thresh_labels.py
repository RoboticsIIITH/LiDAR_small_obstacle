import numpy as np
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.ndimage as sp
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

root_path = "/media/ash/OS/IIIT_Labels/val/"

if __name__ == "__main__":

    sequences = os.listdir(root_path)
    for seq_name in sequences:
        print("Sequence: ",seq_name)
        labels_path = os.path.join(root_path,seq_name,"labels")
        img_path = os.path.join(root_path,seq_name,"image")
        pointcloud_path = os.path.join(root_path,seq_name,"velodyne")
        thresh_label_path = os.path.join(root_path,seq_name,"thresh_label_20m")
        if not os.path.exists(thresh_label_path):
            os.makedirs(thresh_label_path)

        if seq_name in ["seq_1","seq_2"]:
            transform_matrix = read_txt('../best_transf_mat.txt')
        elif seq_name in ["seq_3","seq_4","seq_5","seq_6"]:
            transform_matrix = read_txt('../best_transf_mat_2.txt')
        elif seq_name in ["file_3","file_5"]:
            transform_matrix = read_txt('../file_3_transf.txt')
        elif seq_name in ["file_1","file_2"]:
            transform_matrix = read_txt('../file_1_transf.txt')
        elif seq_name in ["stadium_1","stadium_3","stadium_4","vindhya_1","vindhya_2"]:
            transform_matrix = read_txt('../combined_transf_3.txt')
        else:
            raise FileNotFoundError("No transf matrix file found for sequence:".format(seq_name))

        files = sorted(os.listdir(pointcloud_path))

        for i, file_name in enumerate(files):
            if not os.path.exists(os.path.join(labels_path, file_name.split('.npy')[0] + '.png')):
                continue

            # img = cv2.imread(os.path.join(img_path, file_name.split('.npy')[0] + '.png'))
            label = Image.open(os.path.join(labels_path, file_name.split('.npy')[0] + '.png'))
            label = np.array(label)
            label[label >= 2] = 2

            points = np.load(os.path.join(pointcloud_path, files[i]), allow_pickle=True)
            # Transforming Point-Cloud to NED frame through rotation
            ring_num = points[:, 4]
            points = points[:, :4]
            points[:, 3] = 1.0
            points = rotate_axis(points).transpose()
            front_points = points[:, 2] > 0
            points = points[front_points]
            ring_num = ring_num[front_points]
            points = points.transpose()

            project_points = project_lid_on_img(points, transform_matrix, projection_matrix)
            project_points = project_points.transpose()
            points = points.transpose()
            dist_thresh = 720

            for ring_id in range(7):
                proj_pts = project_points[ring_num == ring_id]
                ring_pts = points[ring_num == ring_id]
                valid_indexes = []
                for k, pt in enumerate(proj_pts):
                    x, y = int(pt[0]), int(pt[1])
                    depth = np.linalg.norm(ring_pts[k, :3])
                    if (0 < x < 1280) and (0 < y < 720) and depth < 20 and label[y, x] >= 1:
                        if y < dist_thresh:
                            dist_thresh = y
                        # cv2.circle(img, (x, y), 2, (0, 255, 0), thickness=1)

            dist_thresh = dist_thresh + 5  # Slight leeway considering projection error
            obstacle_mask = label == 2
            obstacle_mask[:dist_thresh, :] = 0  # Remove obstacles beyond distance threshold
            new_label = label.copy() + obstacle_mask
            new_label[new_label == 2] = 1
            new_label[new_label == 3] = 2
            new_label = new_label.astype(np.uint8)
            new_label =  Image.fromarray(new_label)
            new_label.save(os.path.join(thresh_label_path,file_name.split('.npy')[0] + '.png'))

            """"
            # region_id, num_regions = sp.label(obstacle_mask)
            for regions in range(1, num_regions + 1):
                x, y = np.where(region_id == regions)
                c_x, c_y = int(np.mean(x)), int(np.mean(y))
                cv2.circle(img, (c_y, c_x), 3, (255, 0, 0), 2)

            cv2.imshow("img", img)
            new_label = (50*new_label).astype(np.uint8)
            cv2.imshow("label",new_label)
            # cv2.imshow("orig",50*label.astype(np.uint8))
            if cv2.waitKey(10) == ord('q'):
                print('Quitting....')
                break
            cv2.waitKey(0)
            """