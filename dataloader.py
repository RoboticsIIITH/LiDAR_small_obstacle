import os
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,HuberRegressor,RANSACRegressor,TheilSenRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import time
import cv2
import warnings
import random
warnings.filterwarnings("ignore")


class ProjSet:

    def __init__(self, dir_path, class_num):

        self.root_path = dir_path
        self.class_num = class_num
        self.label_paths = []
        lidar_paths = []
        # for folder in os.listdir(self.root_path):
        for folder in ["vindhya_2"]:
            path = os.path.join(self.root_path, folder, "labels")
            for file in sorted(os.listdir(path)):
                self.label_paths.append(os.path.join(path, file))
                lidar_paths.append(os.path.join(path.split("labels")[0], "velodyne", file.split('.')[0] + '.npy'))

        img_paths = [file.split('labels')[0] + 'image_full' + file.split('labels')[1] for file in self.label_paths]

        start = time.time()
        p = Pool(6)
        self.images = p.map(self.load_file, img_paths)
        self.lidar = p.map(self.load_file, lidar_paths)
        self.labels = p.map(self.load_file, self.label_paths)
        p.close()
        p.join()
        print("Took :{} secs to load data".format(time.time()-start))
        self.proj_matrix = np.array([[692.653256, 0.000000, 629.321381, 0.000],
                                     [0.000, 692.653256, 330.685425, 0.000],
                                     [0.000000, 0.000000, 1.00000, 0.000]])
        self.transf_matrix = []

    def __len__(self):
        return len(self.labels)

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
    def get_mask(inp,span):
        instance_id, instance_num = label(inp)
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
            mask[min_x:max_x,min_y:max_y] = 1

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

    @staticmethod
    def fit_poly(pts):
        pred = []
        model = make_pipeline(PolynomialFeatures(1),RANSACRegressor())
        try:
            model.fit(np.c_[pts[:,0],pts[:,2]],pts[:,1][:,np.newaxis])
        except:
            return [1]*pts.shape[0]
        y_hat = model.predict(np.c_[pts[:,0],pts[:,2]])
        error = [(y_hat[i] - pts[i, 1]) ** 2 for i in range(len(y_hat))]
        mean_error = np.mean(error)
        for term in error:
            if term > 10*mean_error:
                pred.append(-1)
            else:
                pred.append(1)
        return pred

    @staticmethod
    def get_breakpoints(pts):
        diff_log = []
        length = pts.shape[0]
        pred = np.zeros(length)
        new_pred = np.ones(length)

        for i in range(2, length):
            d_i_1 = np.linalg.norm(pts[i - 1, :3])
            d_i_2 = np.linalg.norm(pts[i - 2, :3])
            d_i = np.linalg.norm(pts[i, :3])
            # gamma_1 = np.dot(pts[i - 1, :3], pts[i - 2, :3]) / (d_i_1 * d_i_2)
            # gamma_2 = np.dot(pts[i - 1, :3], pts[i, :3]) / (d_i_1 * d_i)
            gamma_l = np.cos(2 * np.pi / 360 * 0.1)
            # gamma_h = np.cos(2 * np.pi / 360 * 0.5)

            # if gamma_h <= gamma_1 and gamma_h <= gamma_2:  # Continuous Points
            d_p = (d_i_1 * d_i_2) / (2 * d_i_1 * gamma_l - d_i_2)
            diff = d_i - d_p
            if 0.4 < diff < 1:
                pred[i] = 1
                # print("blue",d_p,d_i)
            elif -1 < diff < -0.4:
                # print("Red point",d_p,d_i)
                pred[i] = -1
            diff_log.append(diff)

        min_segment = 1
        segments = []
        for i in range(length):
            if pred[i] == -1:
                obs_start = i
                obs_end = 0
                end_range = i + 11 if i + 11 < length else length
                for j in range(i + 1, end_range):
                    if pred[j] == 1 and j - i > min_segment:
                        obs_end = j
                        break
                d_start = np.linalg.norm(pts[obs_start, :3])
                d_end = np.linalg.norm(pts[obs_end, :3])
                resolution = np.degrees(np.arccos(np.dot(pts[obs_start, :3], pts[obs_end, :3]) / (d_start * d_end)))
                if obs_start != 0 and obs_end != 0 and resolution < 2:
                    # print("segment", resolution)
                    segments.append((obs_start, obs_end))

        for start, end in segments:
            new_pred[start:end] = -1
        return new_pred

    def correspond_lidar_pts(self,points, ring_num, label, mask, T, P):
        """Select ring less than 6 """
        points = points[ring_num < 10]
        ring_num = ring_num[ring_num < 10]
        valid_indexes = []

        proj_pts = self.project_lid_on_img(points.transpose(), T, P).transpose()
        for index, pt in enumerate(proj_pts):
            y, x = int(pt[0]), int(pt[1])
            if 0 < x < 720 and 0 < y < 1280 and label[x, y] >= 1 and points[index,2] > 0:
                valid_indexes.append(index)

        points = points[valid_indexes]  # valid lidar points lying on road
        ring_num = ring_num[valid_indexes]

        for i in range(7):
            # pred = self.clustering(points[ring_num == i])
            pred = self.fit_poly(points[ring_num == i])
            # pred = [-1]*len(points[ring_num==i])
            # pred = self.get_breakpoints(points[ring_num==i])
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
        img = self.images[index]
        label = self.labels[index]
        lidar = self.lidar[index]

        ring_num = lidar[:,4]
        lidar[:,3] = 1.0
        lidar = lidar[:,:4]
        lidar = self.rotate_axis(lidar.transpose()).transpose()
        # class_mask = (label == self.class_num).astype(np.float)     # Target class mask
        class_mask = label >= 2
        class_mask = class_mask.astype(np.float)
        span_window,centroids = self.get_mask(class_mask,span=10)           # Only to be used for small obstacles
        if len(centroids.keys()):
            proj_pts,pred,valid_lidar_pts = self.correspond_lidar_pts(lidar,ring_num,label,span_window,self.transf_matrix,self.proj_matrix)
        else:
            valid_lidar_pts,pred,proj_pts = np.array([]),[],np.array([])

        return img, valid_lidar_pts, class_mask, proj_pts, centroids,pred


if __name__ == '__main__':

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