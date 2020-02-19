import numpy as np
import cv2
import os
from PIL import Image
import scipy
import scipy.ndimage as sp
from matplotlib import pyplot as plt
methods = ["cv2.TM_CCOEFF_NORMED"]
from copy import deepcopy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import copy


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


def get_mask(inp, span=10):
    instance_id, instance_num = sp.label(inp)
    mask = np.zeros((inp.shape[0], inp.shape[1]))
    for i in range(instance_num):
        x, y = np.where(instance_id == i + 1)
        min_x = np.min(x) - span
        min_y = np.min(y) - span
        max_x = np.max(x) + span
        max_y = np.max(y) + span
        mask[min_x:max_x, min_y:max_y] = 1
    return mask


def get_crop_bounds(b_x, b_y, size, h, w):
    bound_left = b_x - size if b_x - size > 0 else 0
    bound_right = b_x + size if b_x - size < h else h
    bound_down = b_y - size if b_y - size > 0 else 0
    bound_up = b_y + size if b_y + size < w else w
    return (bound_left, bound_right), (bound_down, bound_up)


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

        if 0.5 < diff < 1:
            pred[i] = 1
        elif -1 < diff < -0.5:
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


def temporal_prop(image_list,region_list,labels_list,file_paths):
    sigma = 7
    span = 21

    new_region_list = deepcopy(region_list)
    new_image_list = deepcopy(image_list)

    for i in tqdm(range(len(region_list))):
        img = image_list[i]
        img_height, img_width = img.shape[0], img.shape[1]

        # orig_region = region_list[i]
        # orig_mask = orig_region != 0
        # region_id_1, num_region_1 = sp.label(orig_mask)

        # print("file_name:",file_paths[i],num_region_1)

        # for regions in range(1,num_region_1+1):
        #     x, y = np.where(region_id_1 == regions)
        #     c_x, c_y = int(np.mean(x)), int(np.mean(y))
        #     cv2.circle(img, (c_y, c_x), 3, (255,0,0), 2)

        for j in range(-3,0):
            if j == 0 or i + j < 0 or i + j > len(image_list) - 1:
                continue

            frame_img = new_image_list[i + j]
            frame_region = new_region_list[i + j]

            frame_mask = frame_region != 0
            region_id, num_region = sp.label(frame_mask)
            # print("history file name",file_paths[i+j],num_region)

            for k in range(1, num_region + 1):
                x, y = np.where(region_id == k)
                c_x, c_y = int(np.mean(x)), int(np.mean(y))
                # cv2.circle(frame_label, (c_y, c_x), 2, (255), 2)

                # Get crop around region
                (bound_left, bound_right), (bound_down, bound_up) = get_crop_bounds(c_x, c_y,15,img_height,img_width)
                template = frame_img[bound_left:bound_right, bound_down:bound_up]
                # src_region = frame_region[bound_left:bound_right, bound_down:bound_up]

                h, w = template.shape[0], template.shape[1]
                method = eval(methods[0])

                # Get area region to search in current image
                (left_margin, right_margin), (down_margin, up_margin) = get_crop_bounds(c_x, c_y,75,img_height,img_width)
                dest_img = new_image_list[i]
                dest_template = dest_img[left_margin:right_margin, down_margin:up_margin]

                # Apply template Matching
                try:
                    res = cv2.matchTemplate(dest_template, template, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                except:
                    print("Template match error")
                    continue

                # cv2.imshow("template", template)
                # cv2.imshow("dest_template", dest_template)
                # cv2.waitKey(0)

                if max_val >= 0.90:
                    center_point = (int((top_left[0] + bottom_right[0]) / 2) + down_margin,
                                    int((top_left[1] + bottom_right[1]) / 2) + left_margin)
                    x_0, y_0 = center_point[1], center_point[0]
                    left_corner = [0, 0]
                    right_corner = [0, 0]
                    left_corner[0] = top_left[0] + down_margin
                    left_corner[1] = top_left[1] + left_margin
                    right_corner[0] = bottom_right[0] + down_margin
                    right_corner[1] = bottom_right[1] + left_margin

                    # Check if a region is already there
                    if new_region_list[i][x_0, y_0] < 0.6:
                        # new_region_list[i][left_corner[1]:right_corner[1],left_corner[0]:right_corner[0]] += src_region
                        for x in range(x_0 - span, x_0 + span + 1):
                            for y in range(y_0 - span, y_0 + span + 1):
                                if 0 < x < img_height and 0 < y < img_width:
                                    new_region_list[i][x, y] += np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)

                        # cv2.circle(img, (y_0, x_0), 3, (0, 255, 0), 2)
                        # cv2.rectangle(img, tuple(left_corner), tuple(right_corner), 255, 2)

                # else:
                #     print("not detected",max_val)
                    # cv2.imshow("template",template)
                    # cv2.imshow("dest_template",dest_template)
                    # cv2.imshow("src_image",frame_img)
                    # cv2.imshow("dest_image",new_image_list[i])
                    # cv2.waitKey(0)

        new_region_list[i] = np.clip(new_region_list[i], 0, 1)
        new_region_list[i] = new_region_list[i].astype(np.float16)
        np.save(file_paths[i],new_region_list[i])
        # to_plot = new_region_list[i].copy()
        # to_plot = 255*to_plot
        # to_plot = to_plot.astype(np.uint8)
        # to_plot = cv2.applyColorMap(to_plot,colormap=cv2.COLORMAP_JET)
        # cv2.imshow('feed',to_plot)

        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # if cv2.waitKey(10) == ord('q'):
        #     print('Quitting....')
        #     break


if __name__ == "__main__":

    root_path = "/media/ash/OS/IIIT_Labels/val/stadium_3/"
    transform_matrix = read_txt('../combined_transf_3.txt')
    sigma = 5
    span = 15

    labels_path = os.path.join(root_path,"labels")
    img_path = os.path.join(root_path,"image")
    pointcloud_path = os.path.join(root_path, "velodyne")
    dest_dir = os.path.join(root_path,"context_temporal_label_prior")
    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)
    files = sorted(os.listdir(labels_path))

    image_list = []
    region_list = []
    labels_list = []
    file_paths = []

    files = sorted(os.listdir(labels_path))
    pred = []

    for i, file_name in enumerate(tqdm(files)):

        img = cv2.imread(os.path.join(img_path, file_name))
        label = np.array(Image.open(os.path.join(labels_path, file_name)))

        points = np.load(os.path.join(pointcloud_path, file_name.split('.png')[0] + '.npy'), allow_pickle=True)
        obstacle_mask = get_mask(label >= 2,span=5)
        region = np.zeros((img.shape[0],img.shape[1]),dtype=np.float16)

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

        for ring_id in range(8):
            total = []
            proj_pts = project_points[ring_num == ring_id]
            ring_pts = points[ring_num == ring_id]
            valid_indexes = []
            for k, pt in enumerate(proj_pts):
                x, y = int(pt[0]), int(pt[1])

                # Only points lying on road
                if (0 < x < 1280) and (0 < y < 720) and label[y, x] >= 1:
                    valid_indexes.append(k)

            ring_pts = ring_pts[valid_indexes]
            proj_pts = proj_pts[valid_indexes]
            pred = get_breakpoints(ring_pts)
            for k, pt in enumerate(proj_pts):
                x_0, y_0 = int(pt[1]), int(pt[0])
                if pred[k] == -1:
                    if obstacle_mask[x_0,y_0] == 1:            # Only predictions lying on labels
                        # pt_color = (0, 0, 255)
                        # cv2.circle(img, (y_0, x_0), 4, pt_color, 1)
                        for x in range(x_0 - span, x_0 + span + 1):
                            for y in range(y_0 - span, y_0 + span + 1):
                                if 0 < x < img.shape[0] and 0 < y < img.shape[1]:
                                    region[x, y] += np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)
                    # else:
                    #     pt_color = (0, 255, 0)
                    #     cv2.circle(img, (y_0, x_0), 2, pt_color, 1)

        region = np.clip(region,0,1)
        # region = 255*region
        # region = region.astype(np.uint8)
        # region = cv2.applyColorMap(region,colormap=cv2.COLORMAP_JET)
        # cv2.imshow('feed',region)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)

        #region = np.load(os.path.join(dest_dir, file_name.split('.png')[0] + '.npy'))

        image_list.append(img[50:562,280:1000])
        region_list.append(region[50:562,280:1000])
        labels_list.append(label[50:562,280:1000])
        file_paths.append(os.path.join(dest_dir, file_name.split('.png')[0] + '.npy'))

    temporal_prop(image_list, region_list, labels_list, file_paths)
    # region = region.astype(np.float16)
    # np.save(os.path.join(dest_dir,file_name.split('.png')[0]+'.npy'))