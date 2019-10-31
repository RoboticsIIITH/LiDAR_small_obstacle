import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
from sklearn.ensemble import IsolationForest
from multiprocessing import Pool
import cv2
from matplotlib import pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")


class ProjSet:

    def __init__(self, dir_path, class_num):

        self.root_path = dir_path
        self.class_num = class_num
        label_paths = []
        lidar_paths = []
        for folder in os.listdir(self.root_path):
            path = os.path.join(self.root_path,folder,"labels")
            for file in sorted(os.listdir(path)):
                label_paths.append(os.path.join(path,file))
                lidar_paths.append(os.path.join(path.split("labels")[0],"velodyne",file.split('.')[0]+'.npy'))

        img_paths = [file.split('labels')[0] + 'image' + file.split('labels')[1] for file in label_paths]

        p = Pool(8)
        self.images = p.map(self.load_file,img_paths)
        self.lidar = p.map(self.load_file,lidar_paths)
        self.labels = p.map(self.load_file,label_paths)

        self.proj_matrix = np.array([[692.653256 ,0.000000, 629.321381,0.000],
                                    [0.000,692.653256,330.685425,0.000],
                                    [0.000000,0.000000, 1.00000,0.000]])
        self.transf_matrix = []

    def __len__(self):
        return len(self.images)

    @staticmethod
    def load_file(path):
        if path.split('.')[1] == "npy":
            return np.load(path)
        elif path.split('.')[1] == 'png':
            return np.asarray(Image.open(path))

    @property
    def transf(self):
        return self.transf_matrix

    @transf.setter
    def transf(self,value):
        self.transf_matrix = value

    @staticmethod
    def rotate_axis(inp):
        hacky_trans_matrix = R.from_euler('xyz', [1.57, -1.57, 0]).as_dcm()
        hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.zeros(3)[:, np.newaxis]), axis=1)
        hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)
        return np.dot(hacky_trans_matrix, inp)

    @staticmethod
    def project_lid_on_img(lid_pt, T, p):
        tran_pt = np.dot(T, lid_pt)
        proj_lid_pt = np.dot(p, tran_pt).reshape(3, -1)
        pix = np.array([proj_lid_pt[0] / proj_lid_pt[2], proj_lid_pt[1] / proj_lid_pt[2]]).reshape(2, -1)
        return pix

    def road_lidar_pts(self, points, ring_num, label, T, P):
        """Select ring less than 6 """
        points = points[ring_num < 6]
        ring_num = ring_num[ring_num < 6]
        valid_indexes = []

        road_x, road_y = np.where(label == 1)
        max_road_x, min_road_x = np.max(road_x), np.min(road_x)
        max_road_y, min_road_y = np.max(road_y), np.min(road_y)

        proj_pts = self.project_lid_on_img(points.transpose(), T, P).transpose()
        for index, pt in enumerate(proj_pts):
            y, x = int(pt[0]), int(pt[1])
            if x > min_road_x and x < max_road_y and y > min_road_y and y < max_road_y and label[x, y] != 0:
                valid_indexes.append(index)

        points = points[valid_indexes]  # valid lidar points lying on road
        proj_pts = proj_pts[valid_indexes]

        return proj_pts,points

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        lidar = self.lidar[index]

        ring_num = lidar[:,4]
        lidar[:,3] = 1.0
        lidar = lidar[:,:4]
        lidar = self.rotate_axis(lidar.transpose()).transpose()
        x,y = np.where(label == self.class_num)
        reduced_index = np.random.choice(range(x.shape[0]),size=20000)
        x = x[reduced_index]
        y = y[reduced_index]
        class_mask = np.c_[x,y]
        # class_mask = (label == self.class_num).astype(np.uint8)
        # im2, contours, _ = cv2.findContours(class_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = np.array(contours[0]).squeeze()
        # contours = contours[:, [1, 0]]
        proj_pts,valid_lidar_pts = self.road_lidar_pts(lidar,ring_num,label,self.transf_matrix,self.proj_matrix)
        return img, valid_lidar_pts,class_mask,proj_pts


if __name__ == '__main__':
    dataset = ProjSet(dir_path='/media/ash/OS/small_obstacle_bag/synced_data/',class_num=1)
    def read_txt(path):
        with open(path, 'r') as f:
            rows = f.read().split('\n')[:-1]
            values = [row.split(' ')[:-1] for row in rows]
            transform_matrix = np.array(values, dtype=np.float)
            print("Transform matrix :")
            print(transform_matrix)
            return transform_matrix

    dataset.transf = read_txt('best_transf_mat.txt')
    for data in dataset:
        image, pointCloud,target,_ = data
        im2,contours,_ = cv2.findContours(target,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        contours = np.array(contours[0]).squeeze()
        contours = contours[:,[1,0]]
        print(np.max(contours[:,0]))
        for i in range(contours.shape[0]):
            cv2.circle(image, (contours[i,1],contours[i,0]), 1, color=(0, 255, 0))
        cv2.imshow("feed",image)
        cv2.waitKey(0)
        break