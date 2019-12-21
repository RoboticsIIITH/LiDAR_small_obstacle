import os
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
# from sklearn.ensemble import IsolationForest
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge, HuberRegressor, RANSACRegressor, TheilSenRegressor
from matplotlib import pyplot as plt
import torch
import time
import cv2
import warnings
warnings.filterwarnings("ignore")


def read_txt(path):
    with open(path, 'r') as f:
        rows = f.read().split('\n')[:-1]
        values = [row.split(' ')[:-1] for row in rows]
        transform_matrix = np.array(values, dtype=np.float)
        return transform_matrix


class ProjSet:

    def __init__(self, dir_path, class_num, split):

        self.root_path = dir_path
        self.class_num = class_num
        self.label_paths = []
        self.lidar_paths = []
        self.split = split

        for folder in os.listdir(self.root_path):
            path = os.path.join(self.root_path, folder, "labels")
            for file in sorted(os.listdir(path)):
                self.label_paths.append(os.path.join(path, file))
                self.lidar_paths.append(os.path.join(path.split("labels")[0], "velodyne", file.split('.')[0] + '.npy'))

        if split == "test":
            start = time.time()
            p = Pool(4)
            img_paths = [file.split('labels')[0] + 'image' + file.split('labels')[1] for file in self.label_paths]
            self.images = p.map(self.load_file, img_paths)
            self.labels = p.map(self.load_file, self.label_paths)
            self.labels = p.map(self.correct_label, self.labels)
            p.close()
            p.join()
            print("Took: {} secs to load data".format(time.time() - start))

        self.proj_matrix = np.array([[692.653256, 0.000000, 629.321381, 0.000],
                                     [0.000, 692.653256, 330.685425, 0.000],
                                     [0.000000, 0.000000, 1.00000, 0.000]])

        self.transf_matrix = []
        print("Length of {} dataset: {}".format(split,len(self.label_paths)))

    def __len__(self):
        return len(self.label_paths)

    @staticmethod
    def load_file(path):
        if path.split('.')[1] == "npy":
            return np.load(path)
        elif path.split('.')[1] == 'png':
            return np.asarray(Image.open(path))

    @staticmethod
    def correct_label(label):
        h,w = label.shape[0],label.shape[1]
        label = label.flatten()
        label[label>=2] = 2
        label = np.array(label, dtype=np.float32).reshape(h,w)
        return label

    @property
    def transf(self):
        return self.transf_matrix

    @transf.setter
    def transf(self, value):
        quat, transl = value
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
    def get_mask(inp, span=0):
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
            mask[min_x:max_x, min_y:max_y] = 1

        return mask, obs_centroids

    @staticmethod
    def project_lid_on_img(lid_pt, T, p):
        tran_pt = np.dot(T, lid_pt)
        proj_lid_pt = np.dot(p, tran_pt).reshape(3, -1)
        pix = np.array([proj_lid_pt[0] / proj_lid_pt[2], proj_lid_pt[1] / proj_lid_pt[2]]).reshape(2, -1)
        return pix


    def get_ring_labels(self, pointcloud, ring_num, reflectivity, label, T, P):
        ringwise_points = []
        ringwise_labels = []
        ringwise_proj_points = []
        pad_seq_len = 600
        seq_lengths = []

        range_image = np.zeros((16,256,5),dtype=np.float)
        range_label = np.zeros((16,256))
        for ring_id in range(1,17):
            # Get lidar points corresponding to each ring
            temp_points = pointcloud[ring_num == ring_id]
            refl_ring_wise = reflectivity[ring_num == ring_id]
            proj_pts = self.project_lid_on_img(temp_points.transpose(), T, P).transpose()
            # Select only points lying in image FOV
            # valid_indexes = []
            # point_label = []
            avg_count = np.ones(256)
            label_count = np.zeros((256,3))
            label_count = np.concatenate((label_count,100*np.ones((256,1))),axis=1)

            for index, pt in enumerate(proj_pts):
                y, x = int(pt[0]), int(pt[1])
                if x < 720 and y < 1280 and x > 0 and y > 0 and temp_points[index, 2] > 0:
                    bin_index = int(y/1280*256)
                    depth = np.linalg.norm(temp_points[index,:3],ord=2)

                    if range_image[16-ring_id,bin_index,3] == 0:
                        range_image[16-ring_id,bin_index,:3] = temp_points[index,:3]
                        range_image[16-ring_id,bin_index,3] = depth
                        range_image[16-ring_id,bin_index,4] = refl_ring_wise[index]
                        label_count[bin_index,3] = 0

                    else:
                        range_image[16 - ring_id, bin_index, :3] += temp_points[index, :3]
                        range_image[16 - ring_id, bin_index, 3] += depth
                        range_image[16 - ring_id, bin_index, 4] += refl_ring_wise[index]
                        avg_count[bin_index] += 1

                    label_count[bin_index, int(label[x, y])] += 1
                    # valid_indexes.append(index)
                    # point_label.append(label[x, y])

            avg_count = avg_count[:,np.newaxis]
            avg_count = np.repeat(avg_count,repeats=5,axis=1)
            range_image[16-ring_id] /= avg_count
            range_label[16-ring_id] = np.argmax(label_count,axis=1)

            """
            proj_pts = proj_pts[valid_indexes]
            temp_points = temp_points[valid_indexes]
            temp_points = temp_points[:, :3]                    # Discard homogeneous coordinate dim
            refl_ring_wise = refl_ring_wise[valid_indexes]
            point_label = np.array(point_label)
            ring_values = ring_id * np.ones(temp_points.shape[0])


            # Sort points from left to right for each ring
            sorted_indexes = np.argsort(proj_pts[:, 0])
            proj_pts = proj_pts[sorted_indexes]
            temp_points = temp_points[sorted_indexes]
            point_label = point_label[sorted_indexes]
            refl_ring_wise = refl_ring_wise[sorted_indexes]

            # Pad each sequence to have consistent length = pad_seq_len
            if temp_points.shape[0] >= pad_seq_len:
                temp_points = temp_points[:pad_seq_len]
                point_label = point_label[:pad_seq_len]
                proj_pts = proj_pts[:pad_seq_len]
                refl_ring_wise = refl_ring_wise[:pad_seq_len]
                ring_values = ring_values[:pad_seq_len]
                seq_lengths.append(pad_seq_len)
            else:
                seq_lengths.append(temp_points.shape[0])
                pad_pts = np.zeros((pad_seq_len - temp_points.shape[0], 3))
                temp_points = np.append(temp_points, pad_pts, axis=0)
                refl_ring_wise = np.append(refl_ring_wise,np.zeros(pad_seq_len-refl_ring_wise.shape[0]))
                pad_labels = -100 * np.ones(pad_seq_len - point_label.shape[0])
                point_label = np.append(point_label, pad_labels)
                proj_pts = np.append(proj_pts,np.zeros((pad_seq_len-proj_pts.shape[0],2)),axis=0)
                ring_values = np.append(ring_values,np.zeros(pad_seq_len-ring_values.shape[0]),axis=0)

            # Concatenate x,y,z coordinates of pointCloud and their reflectivity
            # temp_points = np.concatenate((temp_points,refl_ring_wise[:,np.newaxis]),axis=1)
            temp_points = np.concatenate((temp_points,ring_values[:,np.newaxis]),axis=1)

            ringwise_points.append(temp_points)
            ringwise_labels.append(point_label)
            ringwise_proj_points.append(proj_pts)

        plt.imshow(range_image[:,:,3],cmap='plasma')
        plt.show()
        
        return np.array(ringwise_points), np.array(ringwise_labels),\
               np.array(seq_lengths), np.array(ringwise_proj_points)
        """
        range_label[range_label==3] = -100
        return range_image,range_label


    def transform_train(self,sample):
        pt_cloud = torch.from_numpy(sample['point_cloud']).float()
        labels = torch.from_numpy(sample['labels']).float()
        ring_len = torch.from_numpy(sample['ring_lengths']).float()

        return {'point_cloud':pt_cloud,
                'labels':labels,
                'ring_lengths':ring_len}

    def transform_test(self,sample):
        pt_cloud = torch.from_numpy(sample['point_cloud']).float()
        labels = torch.from_numpy(sample['labels']).float()
        ring_len = torch.from_numpy(sample['ring_lengths']).float()
        img = torch.from_numpy(sample['image'])
        proj_points = torch.from_numpy(sample['proj_points'])

        return {'point_cloud': pt_cloud,
                'labels': labels,
                'ring_lengths': ring_len,
                'image':img,
                'proj_points':proj_points}

    def __getitem__(self, index):
        # label = self.labels[index]
        label = self.load_file(self.label_paths[index])
        label = self.correct_label(label)
        lidar = self.load_file(self.lidar_paths[index])

        """Hacky way to read transform matrix for now """
        seq_name = self.lidar_paths[index].split('/')[-3]
        if seq_name in ["seq_1","seq_2"]:
            self.transf_matrix = read_txt('best_transf_mat.txt')
        elif seq_name in ["seq_3","seq_4","seq_5","seq_6"]:
            self.transf_matrix = read_txt('best_transf_mat_2.txt')
        elif seq_name in ["file_3","file_5"]:
            self.transf_matrix = read_txt('file_3_transf.txt')
        elif seq_name in ["file_1","file_2"]:
            self.transf_matrix = read_txt('file_1_transf.txt')
        elif seq_name in ["stadium_1","stadium_3","stadium_4","vindhya_1","vindhya_2"]:
            self.transf_matrix = read_txt('combined_transf_3.txt')
        else:
            raise FileNotFoundError("No transf matrix file found for sequence:".format(seq_name))

        ring_num = lidar[:, 4]
        # Make ring_num between 1-16
        ring_num = ring_num + 1

        reflectivity = lidar[:,3].copy()
        lidar[:, 3] = 1.0
        lidar = lidar[:, :4]

        lidar = self.rotate_axis(lidar.transpose()).transpose()

        # inp_points, out_labels, seq_lengths, proj_points = self.get_ring_labels(lidar, ring_num,reflectivity,label,
        #                                                                         self.transf_matrix, self.proj_matrix)

        range_img,range_label = self.get_ring_labels(lidar, ring_num,reflectivity,label,self.transf_matrix, self.proj_matrix)

        if self.split == "test":
            img = self.images[index]
            sample = {'image':img,'point_cloud':inp_points,'labels':out_labels,
                      'ring_lengths':seq_lengths,'proj_points':proj_points}

            return self.transform_test(sample)
        else:
            sample = {'point_cloud': inp_points, 'labels': out_labels,
                      'ring_lengths': seq_lengths}

            return self.transform_train(sample)
