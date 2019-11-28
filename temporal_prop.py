import numpy as np
import cv2
import os
import scipy
from matplotlib import pyplot as plt
methods = ['cv2.TM_CCOEFF_NORMED']
from copy import deepcopy


def temporal_prop(image_list,region_list,file_paths):
    new_region_list = deepcopy(region_list)
    for i,region in enumerate(region_list):
        if len(np.unique(region)) == 1:
            continue
        valid_proposals = region != 0
        region_id, num_region = scipy.ndimage.label(valid_proposals)
        print("File name:{}, Regions : {}".format(file_paths[i].split('labels')[1],num_region))
        num_frames = 3
        for j in range(-num_frames,num_frames+1):
            if j == 0 or i+j < 0 or i+j > len(image_list)-1:
                continue
            near_img = cv2.cvtColor(image_list[i+j], cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(image_list[i],cv2.COLOR_RGB2BGR)
            for k in range(1, num_region + 1):
                x, y = np.where(region_id == k)
                template = img[np.min(x) - 5:np.max(x) + 5, np.min(y) - 5:np.max(y) + 5]
                if template.shape[0] < 5 or template.shape[1] < 5:
                    continue
                h, w = template.shape[0], template.shape[1]
                method = eval(methods[0])
                # Apply template Matching
                try:
                    res = cv2.matchTemplate(near_img, template, method)
                except:
                    print("Template match error")
                    continue

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                if max_val >= 0.90:
                    center_point = (int((top_left[0] + bottom_right[0])/2),int((top_left[1]+bottom_right[1])/2))
                    x_0,y_0 = center_point[1],center_point[0]
                    span=13
                    sigma=7
                    # Check if a region is already there
                    if new_region_list[i+j][x_0,y_0] == 0:
                        # print("Energy given to image:",file_paths[i+j].split('labels')[1])
                        for x in range(x_0 - span, x_0 + span + 1):
                            for y in range(y_0 - span, y_0 + span + 1):
                                if x < 720 and y < 1280:
                                    new_region_list[i+j][x, y] += np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)
                        # cv2.rectangle(near_img, top_left, bottom_right, 255, 2)
                        # cv2.circle(near_img,center_point,3,(0,255,0),2)
                        # cv2.imshow("window", near_img)
                        # cv2.waitKey(0)

    dir_path = os.path.join(file_paths[0].split('labels')[0],"region_prop_updated")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for num,file_name in enumerate(file_paths):
        file_name = dir_path + file_name.split('labels')[1].split('.')[0] + '.npy'
        new_region = new_region_list[num]
        new_region = new_region.astype(np.float32)
        new_region = np.clip(new_region,0,1)
        np.save(file_name,new_region)