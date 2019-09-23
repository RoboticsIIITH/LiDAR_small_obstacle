import cv2
import numpy as np
import os
import kornia
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import torch
from dataloader import ProjSet

device = torch.device('cpu')
transform_matrix = [[0.99961240, 0.00960922,-0.02612872,0.257277],
                    [-0.01086974,0.99876225,-0.04853676,-0.0378583],
                    [0.02562997,0.04880196,0.99847958,-0.0483284],
                    [0, 0, 0,1]]
projection_matrix = [[692.653256 ,0.000000, 629.321381,0.000],
                    [0.000,692.653256,330.685425,0.000],
                    [0.000000,0.000000, 1.00000,0.000]]
transform_matrix = np.array(transform_matrix)
projection_matrix = np.array(projection_matrix)


class Optimizer:
    def __init__(self,dataset,quat,transl,proj,num_epochs=30,cuda=False):
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.quat = self.to_tensor(quat)
        self.quat.requires_grad = True
        self.transl = self.to_tensor(transl)
        # self.transl.requires_grad = True
        self.proj = self.to_tensor(proj)
        self.num_epochs = num_epochs
        self.dataset = dataset

    def to_tensor(self,inp):
        return torch.as_tensor(inp, dtype=torch.float, device=self.device)

    @staticmethod
    def loss_function(quat, trans, proj, lid_pts, seg_mask):
        R = kornia.quaternion_to_rotation_matrix(quat)
        lid_pts = lid_pts.transpose(0, 1)
        cam_pts = torch.matmul(R, lid_pts)
        cam_pts = cam_pts.transpose(0, 1)
        cam_pts += trans
        pixel_index = kornia.geometry.project_points(cam_pts, proj[:3, :3])
        pixel_index = pixel_index[:, torch.LongTensor([1, 0])]  # Swap columns 1 and 0]

        target_pts = torch.nonzero(seg_mask).float()
        pixel_index = pixel_index.unsqueeze(1)
        loss = torch.norm(pixel_index - target_pts, dim=2, p=2)
        loss, _ = torch.min(loss, dim=1)
        loss = torch.max(loss)
        return loss

    def train(self):
        for iter in range(self.num_epochs):
            epoch_loss = 0.0

            for data in self.dataset:
                img,points,target,_,centroids,__ = data
                if len(centroids.keys()) != 0 and points.shape[0] != 0:
                    points = self.to_tensor(points[:, :3])
                    target_mask = self.to_tensor(target)
                    loss = self.loss_function(self.quat, self.transl, self.proj, points, target_mask)
                    epoch_loss += loss.item()
                    # loss.backward()
                    # quat.grad = torch.clamp(quat.grad, -100, 100)
                    # print("Before",quat.data)
                    # quat.data -= 5e-7 * quat.grad.data
                    # quat.grad.data.zero_()
                    # print("After",quat.data)

    def eval(self):
        self.dataset.transf = (self.quat.data.numpy(),self.transl.data.numpy())
        test_loss = 0.0
        for data in self.dataset:
            img, points,target,proj_points,centroids,pred = data
            if len(centroids.keys()) != 0 and points.shape[0] != 0:
                points = self.to_tensor(points[:, :3])
                target_mask = self.to_tensor(target)
                loss = self.loss_function(self.quat, self.transl, self.proj, points, target_mask)
                test_loss += loss.item()

            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            colors = [(0, 0, 255) if elem == -1 else (255, 0, 0) for elem in pred]
            for i in range(proj_points.shape[0]):
                cv2.circle(img, (int(proj_points[i, 0]), int(proj_points[i, 1])), 3, color=colors[i])

            cv2.imshow('feed', img)
            if cv2.waitKey(33) == ord('q'):
                break
        print("Projection Loss across dataset: {}".format(test_loss))
        return test_loss


if __name__ == "__main__":

    quat = R.from_dcm(transform_matrix[:3,:3])
    quat = quat.as_quat()
    # quat = [0.0170,-0.0063,-0.0024,1.0017]
    # quat.data = torch.Tensor([0.0161,-0.0060,-0.0040,1.0018])
    transl = transform_matrix[:3,3]
    dataset = ProjSet(dir_path='/media/ash/OS/small_obstacle_bag/synced_data/', class_num=2)
    calib_opt = Optimizer(dataset,quat,transl,projection_matrix)
    calib_opt.eval()
