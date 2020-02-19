import numpy as np
import cv2
import os
from PIL import Image
import scipy
import scipy.ndimage as sp
from matplotlib import pyplot as plt
methods = ["cv2.TM_CCOEFF_NORMED"]
from copy import deepcopy


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


def temporal_prop(image_list,region_list,labels_list,file_paths):
    new_region_list = deepcopy(region_list)
    # sigma = 7
    # span = 19
    for i in range(len(region_list)):

        img = cv2.cvtColor(image_list[i], cv2.COLOR_RGB2BGR)
        # img = img[50:562,280:1000]
        img_height, img_width = img.shape[0], img.shape[1]

        orig_region = region_list[i] #[50:562,280:1000]
        orig_label = labels_list[i]
        orig_label = (orig_label >= 2).astype(int)
        # orig_label = get_mask(orig_label).astype(int)

        orig_region = orig_region != 0
        # orig_region = (orig_region & orig_label)
        region_id, num_region = sp.label(orig_region)

        for regions in range(1,num_region+1):
            x, y = np.where(region_id == regions)
            c_x, c_y = int(np.mean(x)), int(np.mean(y))
            cv2.circle(img, (c_y, c_x), 3, (255,0,0), 2)
        """
        for j in range(-10, 0):
            if j == 0 or i + j < 0 or i + j > len(image_list) - 1:
                continue
            # Select contexts lying on road
            frame_region = region_list[i + j]
            # frame_region = frame_region[50:562,280:1000]

            frame_label = labels_list[i + j]
            # frame_label = frame_label[50:562,280:1000]
            road_mask = (frame_label >= 1).astype(int)
            # road_mask = get_mask(road_mask).astype(int)

            region_mask = frame_region != 0
            valid_region = (region_mask & road_mask)
            region_id, num_region = sp.label(valid_region)

            frame_img = cv2.cvtColor(image_list[i + j], cv2.COLOR_RGB2BGR)
            # frame_img = frame_img[50:562,280:1000]

            for k in range(1, num_region + 1):
                x, y = np.where(region_id == k)
                c_x, c_y = int(np.mean(x)), int(np.mean(y))
                # cv2.circle(frame_label, (c_y, c_x), 2, (255), 2)

                (bound_left, bound_right), (bound_down, bound_up) = get_crop_bounds(c_x, c_y, 20, img_height,
                                                                                    img_width)
                template = frame_img[bound_left:bound_right, bound_down:bound_up]
                src_region = frame_region[bound_left:bound_right, bound_down:bound_up]

                h, w = template.shape[0], template.shape[1]
                method = eval(methods[0])
                (left_margin, right_margin), (down_margin, up_margin) = get_crop_bounds(c_x, c_y, 150, img_height,
                                                                                        img_width)
                dest_template = img[left_margin:right_margin, down_margin:up_margin]
                # cv2.imshow("template",template)
                # disp = 255*src_region.astype(np.uint8)
                # cv2.imshow("template_region",disp)
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
                    # if new_region_list[i][x_0, y_0] < 0.6:
                        # cv2.rectangle(img, tuple(left_corner), tuple(right_corner), 255, 2)
                    cv2.circle(img, (y_0, x_0), 3, (0, 255, 0), 2)
                        # new_region_list[i][left_corner[1]:right_corner[1],left_corner[0]:right_corner[0]] += src_region
                        # for x in range(x_0-span,x_0+span+1):
                        #     for y in range(y_0-span,y_0+span+1):
                        #         if 0<x<img_height and 0<y<img_width:
                        #             new_region_list[i][x,y] += np.exp(-0.5*((x-x_0)**2 + (y-y_0)**2)/sigma**2)

            # cv2.imshow("label",50*frame_label.astype(np.uint8))
            # cv2.waitKey(0)
        """
        # new_region_list[i] = np.clip(new_region_list[i], 0, 1)
        # new_region_list[i] = new_region_list[i].astype(np.float16)
        # np.save(file_paths[i],new_region_list[i])
        cv2.imshow("image", img)
        cv2.waitKey(0)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break


if __name__ == "__main__":

    root_path = "/media/ash/OS/IIIT_Labels/val/"
    # folders = os.listdir(root_path)
    folders = ["stadium_3"]
    # labels_list = np.load('../../pred_2.npy')
    # length = labels_list.shape[0]
    # labels_list = labels_list[:length-209]

    for folder in folders:
        labels_path = os.path.join(root_path,folder,"labels")
        img_path = os.path.join(root_path,folder,"image")
        region_path = os.path.join(root_path,folder,'region_prop_updated')
        dest_dir = os.path.join(root_path,folder,"context_temporal_template_labels")
        # if not os.path.exists(dest_dir):
        #     os.makedirs(dest_dir)
        files = sorted(os.listdir(labels_path))

        image_list = []
        region_list = []
        labels_list = []
        file_paths = []
        print("Sequence :",folder)

        for i, file_name in enumerate(files):
            img = cv2.imread(os.path.join(img_path, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            region = np.load(os.path.join(region_path, file_name.split('.png')[0] + '.npy'))
            label = np.array(Image.open(os.path.join(labels_path,file_name)))

            image_list.append(img)
            region_list.append(region)
            labels_list.append(label)
            file_paths.append(os.path.join(dest_dir,file_name.split('.png')[0] + '.npy'))

        temporal_prop(image_list, region_list, labels_list,file_paths)