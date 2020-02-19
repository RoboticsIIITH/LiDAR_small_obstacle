import os
import numpy as np
import shutil
import time
from scipy.spatial.transform import Rotation as R
from sklearn import preprocessing
from matplotlib import pyplot as plt
import warnings
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lstm_model import LSTM,Conv1d,Linear
from lstm_dataloader import ProjSet
from lstm_utils import Evaluator
from tensorboardX import SummaryWriter
from focal_loss import FocalLoss,MSELoss
from u_net import UNet
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge,HuberRegressor,RANSACRegressor,TheilSenRegressor
from multiprocessing import Pool

warnings.filterwarnings("ignore")
torch.set_num_threads(1)


def calculate_weights_batch(sample,num_classes):
    z = np.zeros((num_classes,))
    y = sample.cpu().numpy()
    mask = (y >= 0) & (y < num_classes)
    labels = y[mask].astype(np.uint8)
    count_l = np.bincount(labels, minlength=num_classes)
    z += count_l
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    return ret


def run_epoch(model,optim,dataloader,evaluator,writer,epoch,mode,is_cuda=False):

    accum_loss = 0.0
    batch_count = 0
    len_dataset = len(dataloader)
    evaluator.reset()

    if mode == "train":
        model.train()
    else:
        model.eval()

    for sample in dataloader:

        points, labels, seq_lengths = sample['point_cloud'], sample['labels'], sample['ring_lengths']

        points = points.view(points.shape[0] * points.shape[1], points.shape[2], -1)
        labels = labels.view(labels.shape[0] * labels.shape[1], -1)
        seq_lengths = seq_lengths.view(seq_lengths.shape[0] * seq_lengths.shape[1])

        # Shuffle batch
        shuffle_indexes = torch.randperm(points.shape[0])
        points,labels,seq_lengths = points[shuffle_indexes],labels[shuffle_indexes],seq_lengths[shuffle_indexes]

        # Two class label: Obstacle and Road
        # small_obstacle_mask = labels == 2
        # small_obstacle_mask = small_obstacle_mask.cpu().numpy()

        # For Two class
        labels[labels == 2] = 0
        # print(np.unique(labels.numpy()))

        class_weights = calculate_weights_batch(labels, num_classes=2)
        class_weights = torch.from_numpy(class_weights).float()
        # print("Class weights", class_weights)

        points = points[:,:,:4]
        points = points.permute(0,2,1)

        if is_cuda:
            points, labels, seq_lengths = points.cuda(), labels.cuda(), seq_lengths.cuda()
            class_weights = class_weights.cuda()

        # Focal loss
        loss_function = FocalLoss(gamma=0, alpha=class_weights)

        if mode == "train":
            optim.zero_grad()
            pred = model.forward(points,seq_lengths)
            loss = loss_function.forward(pred,labels)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                pred = model.forward(points, seq_lengths)
                loss = loss_function.forward(pred,labels)

        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()

        evaluator.add_batch(pred_label,labels.cpu().numpy())

        accum_loss += loss.item()
        batch_count += 1
        writer.add_scalar('{}/Loss/iter'.format(mode), loss.item(), epoch * len_dataset + batch_count)

    recall,iou = evaluator.get_metrics(class_num=0)
    writer.add_scalar('{}/Loss/epoch'.format(mode), accum_loss / batch_count, epoch)
    writer.add_scalar('{}/IOU/small_obstacle'.format(mode),iou, epoch)
    writer.add_scalar('{}/Recall/small_obstacle'.format(mode),recall,epoch)

    print("Mode : {},Epoch : {}".format(mode,epoch))
    print("Recall: {}, IOU : {}".format(recall,iou))
    return recall,[]


def get_breakpoints(pts):
    diff_log = []
    length = pts.shape[0]
    pred = np.zeros(length)
    new_pred = np.zeros(length)

    for i in range(2,length):
        d_i_1 = np.linalg.norm(pts[i-1,:3])
        d_i_2 = np.linalg.norm(pts[i-2,:3])
        d_i = np.linalg.norm(pts[i,:3])
        gamma_1 = np.dot(pts[i-1,:3],pts[i-2,:3])/(d_i_1*d_i_2)
        gamma_2 = np.dot(pts[i - 1, :3], pts[i, :3]) / (d_i_1 * d_i)
        gamma = (gamma_1 + gamma_2)/2
        d_p = (d_i_1*d_i_2)/(2*d_i_1*gamma-d_i_2)
        diff = d_i-d_p

        if 0.5 < diff < 1:
            pred[i] = 1
        elif -1 < diff < -0.5:
            pred[i] = -1
        diff_log.append(diff)

    min_segment = 1
    segments = []
    for i in range(length):
        if pred[i] == -1:
            obs_start = i
            obs_end = 0
            end_range = i+11 if i+11 < length else length
            for j in range(i+1,end_range):
                if pred[j] == 1 and j-i > min_segment:
                    obs_end = j
                    break
            if obs_start != 0 and obs_end != 0:
                segments.append((obs_start,obs_end))

    for start,end in segments:
        new_pred[start:end] = -1
    return new_pred

"""
def fit_poly(pts):
    pred = []
    # non_zeros = pts[:,3] !=0
    # pts = pts[non_zeros]
    model = make_pipeline(PolynomialFeatures(1),RANSACRegressor())
    try:
        model.fit(np.c_[pts[:,0],pts[:,2]],pts[:,1][:,np.newaxis])
    except:
        return [1]*pts.shape[0]
    y_hat = model.predict(np.c_[pts[:,0],pts[:,2]])
    error = [np.abs(y_hat[i] - pts[i,1]) for i in range(len(y_hat))]
    mean_error = np.mean(error)
    for term in error:
        if term > 5*mean_error:
            pred.append(0)
        else:
            pred.append(1)
    return pred

def clustering(pts):
    non_zeros = pts[:,3] != 0
    pts = pts[non_zeros]
    model = IsolationForest(contamination=0.1).fit(pts[:,:3])
    pred = model.predict(pts[:,:3])
    return pred
"""

def visualise_pred(model,dataset):

    model.eval()
    for sample in dataset:
        image, points, points_label = sample['image'],sample['point_cloud'],sample['labels']
        seq_len, proj_points = sample['ring_lengths'],sample['proj_points']

        points = points[:,:,:4]
        # points = points.permute(0, 2, 1)

        # Two class
        points_label[points_label == 2] = 0

        # with torch.no_grad():
        #     pred = model.forward(points, seq_len)

        # pred = np.argmax(pred.cpu().numpy(),axis=1).squeeze()
        # print(np.unique(pred_out))

        image = image.cpu().numpy()
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        proj_points = proj_points.cpu().numpy()
        seq_len = seq_len.cpu().numpy()
        points = points.numpy()
        points_label = points_label.numpy()

        # Create context regions
        # region_prop = np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)
        # sigma = 5
        # span = 15
        bin_rings = np.zeros((points.shape[0],image.shape[1]))
        for i in range(points.shape[0]):
            len = int(seq_len[i])
            projection_pts = proj_points[i,:len]
            pred = get_breakpoints(points[i,:len])
            # pred = fit_poly(points[i,:len])
            for j in range(projection_pts.shape[0]):
                x_0, y_0 = int(projection_pts[j, 1]), int(projection_pts[j, 0])
                if pred[j] == -1:
                    pt_color = (0,0,255)
                    bin_rings[points.shape[0]-i-1,y_0] = 255
                else:
                    pt_color = (0,255,0)
                    bin_rings[points.shape[0]-i-1,y_0] = 0

                cv2.circle(image,(y_0,x_0),2,pt_color,thickness=1)
                # if pred[j] == 1:
                #     x_0, y_0 = int(projection_pts[j, 1]), int(projection_pts[j, 0])
                #     for x in range(x_0-span,x_0+span+1):
                #         for y in range(y_0-span,y_0+span+1):
                #             if 0<x<720 and 0<y<1280:
                #                 region_prop[x,y] += np.exp(-0.5*((x-x_0)**2 + (y-y_0)**2)/sigma**2)

        # if np.max(region_prop) != 0 :
        #     region_prop = region_prop/np.max(region_prop)
        # else:
        #     assert np.min(region_prop) == np.max(region_prop),"Error in region prop"
        # region_prop = np.clip(region_prop,0,1)
        # region_prop = 255*region_prop
        # region_prop = region_prop.astype(np.uint8)
        # region_prop = cv2.applyColorMap(region_prop,colormap=cv2.COLORMAP_JET)
        # cv2.imshow('feed',region_prop)
        # root_path = os.path.join(label_path.split('labels')[0],'context_ransac')
        # file_name = label_path.split('/')[-1].split('.')[0]
        # if not os.path.exists(root_path):
        #     os.makedirs(root_path)
        # np.save(os.path.join(root_path,file_name + '.npy'),region_prop)

        """
        range_label = np.zeros((8,256))
        proj_dict = dict()

        for ring_id in range(8):
            label_count = np.zeros((256, 2))
            label_count = np.concatenate((label_count, 100 * np.ones((256, 1))), axis=1)

            len = int(seq_len[ring_id])
            projection_pts = proj_points[ring_id,:len]
            pred_labels = pred_out[ring_id,:len]
            bin_ids = bin_indexes[ring_id,:len]

            for i in range(projection_pts.shape[0]):
                x,y = int(projection_pts[i,0]),int(projection_pts[i,1])
                label_count[int(bin_ids[i]),int(pred_labels[i])] += 1
                label_count[int(bin_ids[i]),2] = 0
                proj_dict[ring_id,bin_ids[i]] = [x,y]

            range_label[ring_id] = np.argmax(label_count,axis=1)

        range_label[range_label == 2] = -10

        range_label_copy = range_label.copy()
        range_label_copy = np.concatenate((np.ones((1,256)),range_label_copy,np.zeros((1,256))),axis=0)
        # range_label_copy = range_label[0:7]
        range_label_copy = range_label_copy.transpose(1,0)
        range_label_copy = np.concatenate((np.zeros((1,10)),range_label_copy,np.zeros((1,10))),axis=0)
        range_label_copy = range_label_copy.transpose(1,0)
        range_label_copy = range_label_copy[np.newaxis,np.newaxis,:,:]
        # range_label_copy = np.reshape(range_label_copy,(258,1,18))
        range_label_copy = torch.from_numpy(range_label_copy).float()

        weights = np.array([[4,3,2],[1,1,1],[5,3,4]])
        weights = np.reshape(weights,(1,1,3,3))
        weights = torch.from_numpy(weights).float()
        # conv_label = F.conv1d(range_label_copy,weights,stride=1)
        conv_label = F.conv2d(range_label_copy,weights)

        conv_label = conv_label.numpy()
        conv_label = conv_label.squeeze()
        # print(np.unique(conv_label))
        # conv_label = conv_label.transpose(1,0)
        # print(conv_label.shape)
        # print(np.unique(conv_label))
        color_pred = np.zeros((8, 256, 3))
        color_pred[range_label == 0] = [255,0,0]
        color_pred[range_label == 1] = [0,255,0]
        # color_pred[conv_label == ] = [0, 0, 255]
        # color_pred[pred_out == 2] = [255, 0, 0]

        for key in proj_dict:
            color = color_pred[int(key[0]), int(key[1])]
            pt_color = [int(x) for x in color]
            cv2.circle(image, (proj_dict[key][0], proj_dict[key][1]), 2, pt_color, thickness=1)
        """

        cv2.imshow("feed",image)
        cv2.imshow("pred",bin_rings)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break
        # cv2.waitKey(0)


if __name__ == '__main__':

    visualise_mode = True
    exp_num = "exp_8"
    num_epochs = 40
    save_interval = 1

    # model = Conv1d(num_filters=128,kernel_size=3,inp_dim=5,num_classes=2,batch_size=8*batch_size,dropout_rate=0.2)
    model = UNet(in_channels=4,n_classes=2,depth=3,wf=5)

    if not visualise_mode:
        batch_size = 8
        train_dataset = ProjSet("/scratch/ash/IIIT_Labels/train/", class_num=2, split="train")
        val_dataset = ProjSet("/scratch/ash/IIIT_Labels/val/", class_num=2, split="val")
        use_gpu = not visualise_mode
        print("Using GPU: {}".format(use_gpu))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        if use_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # poly_decay = lambda epoch: pow((1-epoch/num_epochs),0.9)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=poly_decay)

        log_dir = "/scratch/ash/road_prior/logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(os.path.join(log_dir,exp_num))
        writer.add_text(text_string=str(list(model.children())), tag='model_info')
        evaluator = Evaluator(num_classes=2)

        # checkpoint = torch.load('/scratch/ash/road_prior/checkpoints/exp_4/best_model.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        best_recall = 0.0

        for i in range(0, num_epochs):
            start = time.time()
            _, __ = run_epoch(model, optimizer, train_loader, evaluator, writer, epoch=i, mode="train", is_cuda=use_gpu)
            # print("Epoch took:", time.time() - start)
            # scheduler.step(epoch=i)

            if i % save_interval == 0:
                recall, confusion_mat = run_epoch(model, optimizer, val_loader, evaluator, writer, epoch=i, mode="val",
                                                  is_cuda=use_gpu)
                # writer.add_text("confusion matrix on val set", str(list(confusion_mat)), global_step=i)
                # print("Saving checkpoint for Epoch:{}".format(i))
                checkpoint = {'epoch': i,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                dir_path = os.path.join('/scratch/ash/road_prior/checkpoints',exp_num)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                file_path = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(i))
                torch.save(checkpoint, file_path)
                if recall > best_recall:
                    shutil.copyfile(file_path, os.path.join(dir_path, 'best_model.pth.tar'))
                    best_recall = recall

    else:
        test_dataset = ProjSet("/media/ash/OS/IIIT_Labels/test/", class_num=2, split="test")
        # test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=4)
        # checkpoint = torch.load('../checkpoints/road_prior/exp_6/checkpoint_49.pth.tar', map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'])
        visualise_pred(model,test_dataset)


# Sort rings by length in descending order (required for pack_pad_sequence function)
# sorted_rings = torch.argsort(seq_lengths, descending=True)
# points = points[sorted_rings]
# labels = labels[sorted_rings]
# seq_lengths = seq_lengths[sorted_rings]

# Convert label tensor to form: BxT where B=batch size and T = max length of sequence in batch
# max_len = torch.max(seq_lengths)
# labels = labels[:,:int(max_len.item())]