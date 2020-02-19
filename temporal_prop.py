import numpy as np
import cv2
import os
import scipy
import scipy.ndimage as sp
methods = ["cv2.TM_CCOEFF_NORMED"]
from copy import deepcopy


def temporal_prop(image_list,region_list,file_paths):
    new_region_list = deepcopy(region_list)
    for i,region in enumerate(region_list):
        if len(np.unique(region)) == 1:
            continue
        valid_proposals = region != 0
        region_id, num_region = sp.label(valid_proposals)
        # print("File name:{}".format(file_paths[i].split('labels')[1]))
        # print("Total Regions",num_region)
        num_frames = 10
        for j in range(-4,):
            if j == 0 or i+j < 0 or i+j > len(image_list)-1:
                continue
            near_img = cv2.cvtColor(image_list[i+j], cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(image_list[i],cv2.COLOR_RGB2BGR)
            for k in range(1, num_region + 1):
                x, y = np.where(region_id == k)
                c_x,c_y = int(np.mean(x)),int(np.mean(y))
                template = img[np.min(x) - 5: np.max(x) + 5, np.min(y) - 5: np.max(y) + 5]

                if template.shape[0] < 15 or template.shape[1] < 15:
                    continue

                h, w = template.shape[0], template.shape[1]
                method = eval(methods[0])
                left_margin = c_x-100 if c_x-100>0 else 0
                right_margin = c_x + 100 if c_x+100<720 else 720
                down_margin = c_y - 100 if c_y-100>0 else 0
                up_margin = c_y + 100 if c_y+100<1280 else 1280
                cut_image = near_img[left_margin:right_margin,down_margin:up_margin]

                # Apply template Matching
                try:
                    res = cv2.matchTemplate(cut_image,template, method)
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
                    center_point = (int((top_left[0] + bottom_right[0])/2)+down_margin,int((top_left[1]+bottom_right[1])/2)+left_margin)
                    x_0,y_0 = center_point[1],center_point[0]
                    left_corner = [0,0]
                    right_corner = [0,0]
                    left_corner[0] = top_left[0] + down_margin
                    left_corner[1] = top_left[1] + left_margin
                    right_corner[0] = bottom_right[0] + down_margin
                    right_corner[1] = bottom_right[1] + left_margin
                    span=19
                    sigma=7
                    # Check if a region is already there
                    if new_region_list[i+j][x_0,y_0] <= 0.36:
                        # print("Energy given to image:",file_paths[i+j].split('labels')[1])
                        # for x in range(x_0 - span, x_0 + span + 1):
                        #     for y in range(y_0 - span, y_0 + span + 1):
                        #         if x < 720 and y < 1280:
                        #             new_region_list[i+j][x, y] += np.exp(-0.5 * ((x - x_0) ** 2 + (y - y_0) ** 2) / sigma ** 2)
                        # cv2.rectangle(near_img,tuple(left_corner),tuple(right_corner),255, 2)
                        # cv2.circle(region,(c_y,c_x),3,(0,255,0),2)
                        # cv2.imshow("window",near_img)
                        # if cv2.waitKey(10) == ord('q'):
                        #     print('Quitting....')
                        #     break

                        new_region_list[i+j][left_corner[1]:right_corner[1],left_corner[0]:right_corner[0]] += region[c_x-int(h/2):c_x+int(h/2)+1,c_y-int(w/2):c_y+int(w/2)+1]
                        cv2.imshow("image",new_region_list[i+j])
                        # cv2.waitKey(0)

    # dir_path = os.path.join(file_paths[0].split('labels')[0],"region_prop_updated")
    # if not os.path.exists(dir_path):
    #     os.mkdir(dir_path)
    # for num,file_name in enumerate(file_paths):
    #     file_name = dir_path + file_name.split('labels')[1].split('.')[0] + '.npy'
    #     new_region = new_region_list[num]
    #     new_region = new_region.astype(np.float32)
    #     new_region = np.clip(new_region,0,1)
    #     np.save(file_name,new_region)

# odom_path = os.path.join(dir_path,"odometry")
# pointcloud_path = os.path.join(dir_path,"velodyne")
# transform_matrix = read_txt('../combined_transf_3.txt')


if __name__ == "__main__":

    root_path = "/media/ash/OS/IIIT_Labels/val/"
    # folders = os.listdir(root_path)
    folders = ["stadium_3"]
    for folder in folders:
        labels_path = os.path.join(root_path,folder,"labels")
        img_path = os.path.join(root_path,folder,"image")
        region_path = os.path.join(root_path,folder,'context_full')
        files = sorted(os.listdir(labels_path))
        image_list = []
        region_list = []
        file_paths = []
        print("Doing sequence :",folder)
        for i, file_name in enumerate(files):
            img = cv2.imread(os.path.join(img_path, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            region = np.load(os.path.join(region_path, file_name.split('.png')[0] + '.npy'))
            image_list.append(img)
            region_list.append(region)
            file_paths.append(os.path.join(labels_path, file_name))
        temporal_prop(image_list, region_list, file_paths)