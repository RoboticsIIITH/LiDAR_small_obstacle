import os
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
from sklearn.ensemble import IsolationForest
import cv2
import warnings
warnings.filterwarnings("ignore")


class ProjSet:

    def __init__(self, dir_path, class_num):

        self.root_path = dir_path
        self.class_num = class_num
        self.label_paths = []
        self.lidar_paths = []
        for folder in os.listdir(self.root_path):
            path = os.path.join(self.root_path,folder,"labels")
            for file in sorted(os.listdir(path)):
                self.label_paths.append(os.path.join(path,file))
                self.lidar_paths.append(os.path.join(path.split("labels")[0],"velodyne",file.split('.')[0]+'.npy'))

        self.img_paths = [file.split('labels')[0] + 'image' + file.split('labels')[1] for file in self.label_paths]
        self.proj_matrix = np.array([[692.653256 ,0.000000, 629.321381,0.000],
                                    [0.000,692.653256,330.685425,0.000],
                                    [0.000000,0.000000, 1.00000,0.000]])
        self.transf_matrix = []

    def __len__(self):
        return len(self.label_paths)

    @property
    def transf(self):
        return self.transf_matrix

    @transf.setter
    def transf(self,value):
        quat,transl = value
        self.transf_matrix = R.from_quat(quat).as_dcm()
        self.transf_matrix = np.c_[self.transf_matrix, transl]
        self.transf_matrix = np.r_[self.transf_matrix, [[0, 0, 0, 1]]]

    @staticmethod
    def rotate_axis(inp):
        hacky_trans_matrix = R.from_euler('xyz', [1.57, -1.57, 0]).as_dcm()
        hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.zeros(3)[:, np.newaxis]), axis=1)
        hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)
        return np.dot(hacky_trans_matrix, inp)

    @staticmethod
    def get_mask(inp,span=16):
        instance_id, instance_num = label(inp)
        obs_centroids = {}
        for i in range(instance_num):
            x, y = np.where(instance_id == i + 1)
            cx = int(np.mean(x))
            cy = int(np.mean(y))
            obs_centroids[i + 1] = [cx, cy]

        mask = np.zeros((inp.shape[0], inp.shape[1]))
        for key in obs_centroids.keys():
            x, y = obs_centroids[key]
            mask[x - span:x + span, y - span:y + span] = 1
        return mask,obs_centroids

    @staticmethod
    def project_lid_on_img(lid_pt, T, p):
        tran_pt = np.dot(T, lid_pt)
        proj_lid_pt = np.dot(p, tran_pt).reshape(3, -1)
        pix = np.array([proj_lid_pt[0] / proj_lid_pt[2], proj_lid_pt[1] / proj_lid_pt[2]]).reshape(2, -1)
        return pix

    @staticmethod
    def clustering(pts):
        pred = []
        if pts.shape[0]:
            model = IsolationForest(contamination=0.1).fit(pts[:, :3])
            pred = model.predict(pts[:, :3])
        return pred

    def correspond_lidar_pts(self,points, ring_num, label, mask, T, P):
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
        ring_num = ring_num[valid_indexes]

        for i in range(6):
            pred = self.clustering(points[ring_num == i])
            proj_pts = self.project_lid_on_img(points[ring_num == i].transpose(), T, P)
            if i == 0:
                proj_pts_global = np.array(proj_pts)
                pred_global = np.array(pred)
                points_global = np.array(points[ring_num == i])
            else:
                proj_pts_global = np.concatenate((proj_pts_global, proj_pts), axis=1)
                pred_global = np.concatenate((pred_global, pred))
                points_global = np.concatenate((points_global, points[ring_num == i]))

        proj_pts_global = proj_pts_global.transpose()
        for i in range(len(pred_global)):
            if pred_global[i] == -1 and mask[int(proj_pts_global[i, 1]), int(proj_pts_global[i, 0])] == 1:
                continue
            else:
                pred_global[i] = 1

        """Return only detected Outlier Points"""
        proj_pts_global = proj_pts_global[pred_global == -1]
        points_global = points_global[pred_global == -1]
        pred_global = pred_global[pred_global == -1]
        return proj_pts_global, pred_global, points_global

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.img_paths[index]))
        label = np.asarray(Image.open(self.label_paths[index]))
        lidar = np.load(self.lidar_paths[index])

        ring_num = lidar[:,4]
        lidar[:,3] = 1.0
        lidar = lidar[:,:4]
        lidar = self.rotate_axis(lidar.transpose()).transpose()
        class_mask = (label == self.class_num).astype(np.float)     # Target class mask
        span_window,centroids = self.get_mask(class_mask)           # Only to be used for small obstacles

        proj_pts,pred,valid_lidar_pts = self.correspond_lidar_pts(lidar,ring_num,label,span_window,
                                                                  self.transf_matrix,self.proj_matrix)

        return img, valid_lidar_pts, class_mask, proj_pts, centroids,pred


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    dataset = ProjSet(dir_path='/media/ash/OS/small_obstacle_bag/synced_data/',class_num=2)
    quat = [0.0170,-0.0063,-0.0024,1.0017]
    transl = [0.257277,-0.0378583,-0.0483284]
    dataset.transf = (quat,transl)
    for data in dataset:
        image, pointCloud, projection, _ = data
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for i in range(projection.shape[0]):
            cv2.circle(image, (int(projection[i, 0]), int(projection[i, 1])), 3, color=(0, 255, 0))
        cv2.imshow("feed", image)
        cv2.waitKey(0)