import cv2
import numpy as np
from PIL import Image
import os
import kornia
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import scipy.cluster as cluster
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,HuberRegressor,RANSACRegressor,TheilSenRegressor
from scipy.ndimage import label
import scipy
from sklearn.metrics import mean_squared_error
from scipy import interpolate
import torch
from torch import autograd
import warnings
warnings.filterwarnings("ignore")


wk = 33
device = torch.device('cpu')
path = '/media/ash/OS/small_obstacle_bag/synced_data/seq_1/'
image_path = os.path.join(path, 'image')
label_path = os.path.join(path, 'labels')
pointCloud_path = os.path.join(path,'velodyne')

images = [os.path.join(image_path, i) for i in sorted(os.listdir(label_path))]
labels = [os.path.join(label_path, i) for i in sorted(os.listdir(label_path))]
ptClouds = [os.path.join(pointCloud_path, i.split('.')[0] + '.npy') for i in sorted(os.listdir(label_path))]

transform_matrix = [[0.99961240, 0.00960922,-0.02612872,0.257277],
                    [-0.01086974,0.99876225,-0.04853676,-0.0378583],
                    [0.02562997,0.04880196,0.99847958,-0.0483284],
                    [0, 0, 0,1]]

projection_matrix = [[692.653256 ,0.000000, 629.321381,0.000],
                     [0.000,692.653256,330.685425,0.000],
                     [0.000000,0.000000, 1.00000,0.000]]

transform_matrix = np.array(transform_matrix)
projection_matrix = np.array(projection_matrix)


def project_lid_on_img(lid_pt,T,p):
    tran_pt =  np.dot(T,lid_pt)
    proj_lid_pt = np.dot(p,tran_pt).reshape(3,-1)
    pix = np.array([proj_lid_pt[0]/proj_lid_pt[2],proj_lid_pt[1]/proj_lid_pt[2]]).reshape(2,-1)
    return pix


def get_hacky_transf(inp):
    hacky_trans_matrix = R.from_euler('xyz', [1.57, -1.57, 0]).as_dcm()
    hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.zeros(3)[:, np.newaxis]), axis=1)
    hacky_trans_matrix = np.concatenate((hacky_trans_matrix, np.array([[0, 0, 0, 1]])), axis=0)
    return np.dot(hacky_trans_matrix,inp)


def clustering(pts):
    pred = []
    if pts.shape[0]:
        model = IsolationForest(contamination=0.1).fit(pts[:,:3])
    # model = make_pipeline(PolynomialFeatures(2), RANSACRegressor())
    # feature_vect = np.array([pts[:,0],pts[:,2]]).transpose()
    # model.fit(feature_vect,pts[:,1])
        pred = model.predict(pts[:,:3])
    # deviation = np.sqrt(np.mean(abs(pred - pts[:,1]) ** 2))
    # result =[]
    # for i in range(len(pred)):
    #     if abs(pred[i] - pts[i,1]) > 2.5 * deviation:
    #         result.append(-1)
    #     else:
    #         result.append(1)
    return pred


def valid_lidar_pts(points, ring_num, label, mask, T, P):
    """Select ring less than 6 """
    points = points[ring_num<6]
    ring_num = ring_num[ring_num<6]
    valid_indexes = []

    road_x,road_y = np.where(label == 1)
    max_road_x,min_road_x  = np.max(road_x),np.min(road_x)
    max_road_y,min_road_y = np.max(road_y),np.min(road_y)

    proj_pts = project_lid_on_img(points.transpose(),T,P).transpose()
    for index,pt in enumerate(proj_pts):
        y,x = int(pt[0]),int(pt[1])
        if x > min_road_x and x < max_road_y and y > min_road_y and y < max_road_y and label[x,y] != 0:
            valid_indexes.append(index)

    points = points[valid_indexes]                  # valid lidar points lying on road
    ring_num = ring_num[valid_indexes]

    for i in range(6):
        pred = clustering(points[ring_num == i])
        # pred = spline_fit(points[ring_num==i])
        proj_pts = project_lid_on_img(points[ring_num==i].transpose(),T,P)
        if i==0:
            proj_pts_global = np.array(proj_pts)
            pred_global = np.array(pred)
            points_global = np.array(points[ring_num==i])
        else:
            proj_pts_global = np.concatenate((proj_pts_global,proj_pts),axis=1)
            pred_global = np.concatenate((pred_global,pred))
            points_global = np.concatenate((points_global,points[ring_num==i]))
        # unique_clusters = np.unique(pred)
        # for elem in unique_clusters:
            # print("Mean of cluster:{} = {}".format(elem, np.mean(points[ring_num==i, 1][pred==elem])))

    proj_pts_global = proj_pts_global.transpose()
    for i in range(len(pred_global)):
        if pred_global[i] == -1 and mask[int(proj_pts_global[i, 1]), int(proj_pts_global[i, 0])] == 1:
            continue
        else:
            pred_global[i] = 1

    """Select only detected Outlier Points"""
    proj_pts_global = proj_pts_global[pred_global == -1]
    points_global = points_global[pred_global == -1]
    pred_global = pred_global[pred_global==-1]

    return proj_pts_global,pred_global,points_global


def to_tensor(inp):
    return torch.as_tensor(inp,dtype=torch.float,device=device)


def loss_function(quat,trans,proj,lid_pts,seg_mask):

    R = kornia.quaternion_to_rotation_matrix(quat)
    lid_pts = lid_pts.transpose(0,1)
    cam_pts = torch.matmul(R,lid_pts)
    cam_pts = cam_pts.transpose(0,1)
    cam_pts += trans
    pixel_index = kornia.geometry.project_points(cam_pts,proj[:3,:3])
    pixel_index = pixel_index[:,torch.LongTensor([1,0])]           # Swap columns 1 and 0]

    target_pts = torch.nonzero(seg_mask).float()
    pixel_index = pixel_index.unsqueeze(1)
    loss = torch.norm(pixel_index-target_pts,dim=2,p=2)
    loss,_ = torch.min(loss,dim=1)
    loss = torch.max(loss)
    return loss


if __name__ == "__main__":

    span = 16
    quat = R.from_dcm(transform_matrix[:3,:3])
    quat = quat.as_quat()
    quat = autograd.Variable(to_tensor(quat),requires_grad=True)
    # quat.data = torch.Tensor([0.0170,-0.0063,-0.0024,1.0017])
    quat.data = torch.Tensor([0.0161,-0.0060,-0.0040,1.0018])
    trans = transform_matrix[:3,3]
    trans = autograd.Variable(to_tensor(trans),requires_grad=False)

    proj = to_tensor(projection_matrix)
    epoch = 1
    print("Starting Quat:",quat.data)

    for num in range(epoch):
        epoch_loss = 0.0

        for i, image in enumerate(images):
            path = {}
            path['image'] = images[i]
            path['label'] = labels[i]
            path['ptCloud'] = ptClouds[i]

            label_template = np.asarray(Image.open(labels[i]))
            mask_template = (label_template == 2).astype(np.uint8)
            img_template = cv2.imread(image)

            pointCloud = np.load(path['ptCloud'])  # Shape Nx5 (0-2 channel holds (x,y,z), 3rd channel Intensity)
            ring_info = pointCloud[:, 4]  # Ring number
            pointCloud[:, 3] = 1  # Convert to Homogeneous coordinates
            pointCloud = pointCloud[:, :4].transpose()  # Shape 4xN
            pointCloud = get_hacky_transf(pointCloud)
            pointCloud = pointCloud.transpose()

            instance_id, instance_num = label(mask_template)
            obs_centroids = {}
            for i in range(instance_num):
                x, y = np.where(instance_id == i + 1)
                cx = int(np.mean(x))
                cy = int(np.mean(y))
                obs_centroids[i + 1] = [cx, cy]
                cv2.circle(img_template, (cy, cx), 3, color=(0, 255, 0))

            mask = np.zeros((img_template.shape[0], img_template.shape[1]))
            for key in obs_centroids.keys():
                x, y = obs_centroids[key]
                mask[x - span:x + span, y - span:y + span] = 1

            transf = quat.clone().data.numpy()
            transf = R.from_quat(transf)
            transf = transf.as_dcm()
            transf = np.c_[transf, trans.clone().data.numpy()]
            transf = np.r_[transf, [[0, 0, 0, 1]]]

            proj_pts, pred, valid_points = valid_lidar_pts(pointCloud, ring_info, label_template, mask, transf,
                                                           projection_matrix)

            """Loss function evaluation"""
            if len(obs_centroids.keys()) != 0 and valid_points.shape[0] != 0:
                valid_points = to_tensor(valid_points[:, :3])
                # centroids = to_tensor(list(obs_centroids.values()))
                mask_template = to_tensor(mask_template)
                loss = loss_function(quat, trans, proj, valid_points, mask_template)
                epoch_loss += loss.item()
                # loss.backward()
                # quat.grad = torch.clamp(quat.grad, -100, 100)
                # print("Before",quat.data)
                # quat.data -= 5e-7 * quat.grad.data
                # quat.grad.data.zero_()
                # print("After",quat.data)

            colors = [(0, 0, 255) if elem == -1 else (255, 0, 0) for elem in pred]
            for i in range(proj_pts.shape[0]):
                cv2.circle(img_template, (int(proj_pts[i, 0]), int(proj_pts[i, 1])), 3, color=(0, 0, 255))

            cv2.imshow('feed', img_template)
            if cv2.waitKey(wk) == ord('q'):
                break
            # cv2.waitKey(0)

        print("Epoch: {} Loss: {}".format(num,epoch_loss))
        print("Quat : {}".format(quat.data))







