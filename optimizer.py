import cv2
import numpy as np
import os
import kornia
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import torch
import time
import matplotlib
import scipy
from PIL import Image
from dataloader import ProjSet
import imageio
from tqdm import tqdm
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

projection_matrix = [[692.653256 ,0.000000, 629.321381,0.000],
                    [0.000,692.653256,330.685425,0.000],
                    [0.000000,0.000000, 1.00000,0.000]]


def read_txt(path):
    with open(path,'r') as f:
        rows = f.read().split('\n')[:-1]
        values = [row.split(' ')[:-1] for row in rows]
        transform_matrix = np.array(values,dtype=np.float)
        return transform_matrix


transform_matrix = read_txt('best_transf_mat_2.txt')
projection_matrix = np.array(projection_matrix)


class Optimizer:
    def __init__(self,dataset,quat,transl,proj,num_epochs=50,lr=1e-4,cuda=True):
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.quat = self.to_tensor(quat)
        # self.quat.requires_grad = True
        self.transl = self.to_tensor(transl)
        # self.transl.requires_grad = False
        self.proj = self.to_tensor(proj)
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.dataset.transf = (self.quat.cpu().data.numpy(), self.transl.cpu().data.numpy())
        self.lr = lr
        # self.optim = torch.optim.Adam([self.quat,self.transl],lr=self.lr)

    def to_tensor(self,inp):
        return torch.as_tensor(inp, dtype=torch.float, device=self.device)

    @staticmethod
    def compute_dist(x,y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist

    def loss_function(self, quat, trans, proj, lid_pts, seg_mask):
        R = kornia.quaternion_to_rotation_matrix(quat)
        lid_pts = lid_pts.transpose(0, 1)
        cam_pts = torch.matmul(R, lid_pts)
        cam_pts = cam_pts.transpose(0, 1)
        cam_pts += trans
        pixel_index = kornia.geometry.project_points(cam_pts, proj[:3, :3])
        pixel_index = pixel_index[:, torch.LongTensor([1, 0])]  # Swap columns 1 and 0]

        target_pts = torch.nonzero(seg_mask).float()
        # pixel_index = pixel_index.unsqueeze(1)
        # loss = torch.norm(pixel_index - target_pts, dim=2, p=2)
        loss = self.compute_dist(pixel_index,target_pts)
        loss, _ = torch.min(loss, dim=1)
        loss = torch.max(loss)
        return loss

    def train(self):
        # loss_log = []
        last_best = np.inf
        batch_count = 0
        batch_loss = self.to_tensor([0.0])
        for epoch in range(self.num_epochs):
            if epoch < 25:
                self.quat.requires_grad = True
                self.transl.requires_grad = False
                self.optim = torch.optim.Adam([self.quat], lr=self.lr)
            else:
                self.quat.requires_grad = False
                self.transl.requires_grad = True
                self.optim = torch.optim.Adam([self.transl], lr=self.lr)

            train_loss = 0.0
            start_time = time.time()
            batch_count = 0
            for data in self.dataset:
                img,points,target,_,centroids,__ = data
                if len(centroids.keys()) != 0 and points.shape[0] != 0:
                    points = self.to_tensor(points[:, :3])
                    target_mask = self.to_tensor(target)
                    loss = self.loss_function(self.quat, self.transl, self.proj, points, target_mask)
                    train_loss += loss.item()
                    batch_loss += loss
                    batch_count += 1
                    if batch_count % 4 == 0 and batch_count != 0:
                        self.optim.zero_grad()
                        batch_loss = batch_loss/4
                        batch_loss.backward()
                        self.optim.step()
                        # batch_count = 0
                        batch_loss = self.to_tensor([0.0])

            print("Epoch : {}, Loss: {}, Took :{} secs".format(epoch,train_loss,time.time()-start_time))
            self.dataset.transf = (self.quat.cpu().data.numpy(), self.transl.cpu().data.numpy())
            if train_loss < last_best:
                self.save_best()
                last_best = train_loss
            checkpt = {'loss': train_loss,'transf_mat': self.dataset.transf,'epoch': epoch}
            out_path = os.path.join('/home/ash/Small-Obs-Project/Image_lidar_fusion/calib_mat_4/checkpoints/')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.save(os.path.join(out_path,'{0:010d}.npy'.format(epoch)),checkpt)

    def eval(self):
        self.dataset.transf = (self.quat.cpu().data.numpy(), self.transl.cpu().data.numpy())
        test_loss = 0.0
        total_valid_region_images = 0
        total_valid_label_images = 0
        region_img_array=[]
        for index,data in enumerate(tqdm(self.dataset)):
            img, points,target,proj_points,centroids,pred = data
            if len(centroids.keys()) != 0 and points.shape[0] != 0:
                points = self.to_tensor(points[:, :3])
                target_mask = self.to_tensor(target)
                with torch.no_grad():
                    loss = self.loss_function(self.quat, self.transl, self.proj, points, target_mask)
                print("Loss",loss.item())
                test_loss += loss.item()
                # self.plot_loss_surface(points,target_mask)

            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            colors = [(0, 0, 255) if elem == -1 else (0, 0, 0) for elem in pred]
            size = [1 if elem == -1 else 1 for elem in pred]
            thickness = [5 if elem == -1 else 1 for elem in pred]
            for i in range(proj_points.shape[0]):
                cv2.circle(img, (int(proj_points[i, 0]), int(proj_points[i, 1])), size[i], color=colors[i],thickness=thickness[i])

            cv2.imshow('feed',img)
            if cv2.waitKey(33) == ord('q'):
                break
            cv2.waitKey(0)
            # if index == 118:
            #     print(index)
            #     cv2.imwrite('/home/ash/Desktop/pre_calib.png',overlayed_seg)
            #     print("done")

        print("Projection Loss across dataset: {}".format(test_loss))
        return test_loss

    def save_best(self):
        transf = R.from_quat(self.quat.clone().cpu().data.numpy()).as_dcm()
        transf = np.c_[transf, self.transl.clone().cpu().data.numpy()]
        transf = np.r_[transf, [[0, 0, 0, 1]]]
        out_path = os.path.join('/home/ash/Small-Obs-Project/Image_lidar_fusion/calib_mat_4/')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path,'iros_best_transf_vindhya_2.txt'),'w+') as f:
            for i in range(transf.shape[0]):
                for j in range(transf.shape[1]):
                    f.write(str(transf[i,j])+' ')
                f.write('\n')

    def plot_loss_surface(self,points,target):
        quat = self.quat.data.numpy()
        angles = R.from_quat(quat)
        angles = angles.as_euler('zyx',degrees=True)
        X = np.arange(angles[0]+angles[0]*0.3,angles[0]-angles[0]*0.3, 0.0025)
        Y = np.arange(angles[1]+angles[1]*0.3,angles[1]-angles[1]*0.3, 0.0025)
        # Z = np.arange(angles[2]-angles[2]*0.1,angles[2]+angles[2]*0.1,0.05)
        X,Y = np.meshgrid(X,Y)
        XX = np.ravel(X)
        YY = np.ravel(Y)
        # ZZ = np.ravel(Z)
        loss_vec = []
        for i in range(len(XX)):
            inp_angle = R.from_euler('zyx',[XX[i],YY[i],angles[2]],degrees=True)
            inp_angle = inp_angle.as_quat()
            quat_new = self.to_tensor(inp_angle)
            loss = self.loss_function(quat_new,self.transl,self.proj,points,target)
            loss_vec.append(loss.item())

        loss_vec = np.array(loss_vec).reshape(X.shape)
        plt.contourf(Y,X,loss_vec,cmap='plasma')
        plt.show()


if __name__ == "__main__":

    quat = R.from_dcm(transform_matrix[:3,:3])
    quat = quat.as_quat()
    transl = transform_matrix[:3,3]
    dataset = ProjSet(dir_path='/media/ash/OS/IIIT_Labels/val/', class_num=2)
    print("Dataset found with {} samples".format(len(dataset)))
    calib_opt = Optimizer(dataset,quat,transl,projection_matrix)
    calib_opt.train()