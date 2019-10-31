import cv2
import numpy as np
import os
import kornia
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import torch
import time
from geomloss import SamplesLoss
from dataloader_road import ProjSet

device = torch.device('cpu')
projection_matrix = [[692.653256 ,0.000000, 629.321381,0.000],
                    [0.000,692.653256,330.685425,0.000],
                    [0.000000,0.000000, 1.00000,0.000]]

def read_txt(path):
    with open(path,'r') as f:
        rows = f.read().split('\n')[:-1]
        values = [row.split(' ')[:-1] for row in rows]
        transform_matrix = np.array(values,dtype=np.float)
        return transform_matrix

transform_matrix = read_txt('best_transf_mat.txt')
projection_matrix = np.array(projection_matrix)


class Optimizer:
    def __init__(self,dataset,quat,transl,proj,num_epochs=200,lr=1e-4,cuda=True):
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.quat = self.to_tensor(quat)
        self.quat.requires_grad = False
        self.transl = self.to_tensor(transl)
        self.transl.requires_grad = True
        self.proj = self.to_tensor(proj)
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.lr = lr
        self.optim = torch.optim.Adam([self.transl],lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=20,gamma=0.5)

    def to_tensor(self,inp):
        return torch.as_tensor(inp, dtype=torch.float, device=self.device)

    @staticmethod
    def compute_dist(x,y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist

    def loss_function(self,quat, trans, proj, lid_pts,target_pts):
        R = kornia.quaternion_to_rotation_matrix(quat)
        lid_pts = lid_pts.transpose(0, 1)
        cam_pts = torch.matmul(R, lid_pts)
        cam_pts = cam_pts.transpose(0, 1)
        cam_pts += trans
        pixel_index = kornia.geometry.project_points(cam_pts, proj[:3, :3])
        pixel_index = pixel_index[:, torch.LongTensor([1, 0])]  # Swap columns 1 and 0]
        pixel_index = pixel_index.unsqueeze(1)
        loss = torch.norm(pixel_index - target_pts, dim=2, p=2)
        # loss = self.compute_dist(pixel_index,target_pts)
        loss, _ = torch.min(loss, dim=1)
        loss = torch.max(loss)
        # criterion = SamplesLoss(loss="hausdorff",p=2,blur=1e-4)
        # loss = criterion(pixel_index,target_pts)
        return loss

    def train(self):
        last_best = np.inf
        for epoch in range(self.num_epochs):
            train_loss = 0.0

            for data in self.dataset:
                img,points,target,_ = data
                if points.shape[0] != 0:
                    points = self.to_tensor(points[:, :3])
                    target_points = self.to_tensor(target)
                    self.optim.zero_grad()
                    loss = self.loss_function(self.quat, self.transl, self.proj, points, target_points)
                    train_loss += loss.item()
                    loss.backward()
                    # print("Before",self.quat.data,self.transl.data)
                    # print("Loss",loss.item())
                    self.optim.step()
                    # print("After",self.quat.data,self.transl.data)

            print("Epoch : {}, Loss: {}".format(epoch,train_loss))
            self.lr_scheduler.step()
            if train_loss < last_best:
                self.save_best()
                last_best = train_loss

    def eval(self,transf):
        test_loss = 0.0
        for data in self.dataset:
            img, points,target,proj_points = data
            if points.shape[0] != 0:
                points = self.to_tensor(points[:, :3])
                target_points = self.to_tensor(target)
                # loss = self.loss_function(self.quat, self.transl, self.proj, points, target_points)
                # test_loss += loss.item()

            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            for i in range(proj_points.shape[0]):
                cv2.circle(img, (int(proj_points[i, 0]), int(proj_points[i, 1])), 3, color=(0,255,0))

            cv2.imshow('feed', img)
            if cv2.waitKey(33) == ord('q'):
                break
        print("Projection Loss across dataset: {}".format(test_loss))
        return test_loss

    def save_best(self,save=True):
        transf = R.from_quat(self.quat.clone().cpu().data.numpy()).as_dcm()
        transf = np.c_[transf, self.transl.clone().cpu().data.numpy()]
        transf = np.r_[transf, [[0, 0, 0, 1]]]
        if save:
            with open('road_transf_mat_2.txt', 'w+') as f:
                for i in range(transf.shape[0]):
                    for j in range(transf.shape[1]):
                        f.write(str(transf[i, j]) + ' ')
                    f.write('\n')
        else:
            return transf


if __name__ == "__main__":

    previous_best = read_txt('road_transf_mat_2.txt')
    quat = R.from_dcm(previous_best[:3,:3])
    quat = quat.as_quat()
    transl = previous_best[:3,3]
    # quat = np.array([0,0,0,1],dtype=np.float)
    # transl = np.array([0,0,0],dtype=np.float)
    dataset = ProjSet(dir_path='/scratch/ash/iiit_data/train/', class_num=1)
    dataset.transf = transform_matrix
    print("Dataset found with {} samples".format(len(dataset)))
    calib_opt = Optimizer(dataset,quat,transl,projection_matrix)
    calib_opt.train()