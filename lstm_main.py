import os
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from sklearn import preprocessing
from matplotlib import pyplot as plt
import warnings
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lstm_model import LSTM
from lstm_dataloader import ProjSet
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")


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


def scale_inp(inp):
    """Scale inp array between range 0-1"""

    assert inp.shape[2] == 4,"Incorrect inp feature dimension: Required 4-dim features"
    for axis in range(inp.shape[2]):
        inp[:,:,axis] = (inp[:,:,axis]-torch.min(inp[:,:,axis]))/(torch.max(inp[:,:,axis])-torch.min(inp[:,:,axis]))
    return inp


def run_epoch(model,optim,dataloader,writer,epoch,mode,is_cuda=False):

    accum_loss = 0.0
    batch_count = 0
    len_dataset = len(dataloader)

    for sample in dataloader:

        points, labels, seq_lengths = sample['point_cloud'], sample['labels'], sample['ring_lengths']
        points = points.view(points.shape[0] * points.shape[1], points.shape[2], -1)
        labels = labels.view(labels.shape[0] * labels.shape[1], -1)
        seq_lengths = seq_lengths.view(seq_lengths.shape[0] * seq_lengths.shape[1])

        # Scale/Normalise input to 0-1 range for each feature
        points = scale_inp(points)

        # Sort rings by length in descending order (required for pack_pad_sequence function)
        sorted_rings = torch.argsort(seq_lengths, descending=True)
        points = points[sorted_rings]
        labels = labels[sorted_rings]
        seq_lengths = seq_lengths[sorted_rings]

        class_weights = calculate_weights_batch(labels, num_classes=3)
        class_weights = torch.from_numpy(class_weights)

        if is_cuda:
            points, labels, seq_lengths = points.cuda(), labels.cuda(), seq_lengths.cuda()

        if mode == "train":
            optim.zero_grad()
            pred = model.forward(points, seq_lengths)
            loss = model.compute_loss(pred, labels, seq_lengths, weight=class_weights)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                pred = model.forward(points, seq_lengths)
                loss = model.compute_loss(pred, labels, seq_lengths, weight=class_weights)

        accum_loss += loss.item()
        batch_count += 1
        writer.add_scalar('{}/Loss/iter'.format(mode), loss.item(), epoch * len_dataset + batch_count)

    print("{}_loss:{}".format(mode,accum_loss / batch_count))
    writer.add_scalar('{}/Loss/epoch'.format(mode), accum_loss / batch_count, epoch)


def visualise_pred(model,dataset):

    for sample in dataset:
        image, points, points_label = sample['image'],sample['point_cloud'],sample['labels']
        seq_len, proj_points = sample['ring_lengths'],sample['proj_points']

        points = scale_inp(points)
        sorted_rings = torch.argsort(seq_len,descending=True)
        points = points[sorted_rings]
        seq_len = seq_len[sorted_rings]
        proj_points = proj_points[sorted_rings]

        with torch.no_grad():
            pred = model.forward(points, seq_len)

        seq_len = seq_len.cpu().numpy()
        pred_softmax = F.softmax(pred, dim=2).numpy()
        pred_out = np.argmax(pred_softmax,axis=2).squeeze()

        image = image.cpu().numpy()
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        proj_points = proj_points.cpu().numpy()

        for ring_id in range(16):
            len = int(seq_len[ring_id])
            projection_pts = proj_points[ring_id,0:len]
            pred_labels = pred_out[ring_id,0:len]

            for i in range(projection_pts.shape[0]):
                if pred_labels[i] == 1:
                    pt_color = (0,255,0)
                elif pred_labels[i] == 2:
                    pt_color = (0,0,255)
                else:
                    pt_color = (255,0,0)
                cv2.circle(image,(int(projection_pts[i,0]),int(projection_pts[i,1])),2,pt_color,thickness=1)
        cv2.imshow("feed",image)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break
        # cv2.waitKey(0)


if __name__ == '__main__':

    # train_dataset = ProjSet(dir_names=["seq_1","seq_2","seq_3","seq_4","seq_5","seq_6"],class_num=2,split="train)
    train_dataset = ProjSet(dir_names=["seq_1"],class_num=2,split="train")
    # val_dataset = ProjSet(dir_names=["file_3", "file_5"], class_num=2,split="val")
    test_dataset = ProjSet(dir_names=["vindhya_2"], class_num=2,split="test")

    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=4)
    # val_loader = DataLoader(val_dataset,batch_size=16,shuffle=True,drop_last=True,num_workers=4)

    num_epochs = 250
    model = LSTM(nb_layers=1,is_cuda=False)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    poly_decay = lambda epoch: pow((1-epoch/num_epochs),0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=poly_decay)

    log_dir = "/home/ash/Small-Obs-Project/Image_lidar_fusion/lstm_logs/"
    writer = SummaryWriter(os.path.join(log_dir,"exp_4"))
    # checkpoint = torch.load('/home/ash/Small-Obs-Project/Image_lidar_fusion/checkpoints/exp-3/checkpoint_245.pth.tar',
    #                         map_location='cpu')

    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(num_epochs):
        run_epoch(model,optimizer,train_loader,writer,epoch=i,mode="train",is_cuda=False)
        scheduler.step(epoch=i)

        if i % 5 == 0 and i != 0:
            print("Saving checkpoint for Epoch:{}".format(i))
            checkpoint = {'epoch': i,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            file_path = os.path.join('/home/ash/Small-Obs-Project/Image_lidar_fusion/checkpoints', 'exp_3')
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            file_path = os.path.join(file_path, 'checkpoint_{}.pth.tar'.format(i))
            torch.save(checkpoint, file_path)

    # visualise_pred(model,test_dataset)
    #TODO : linear layer model: add dropout, padding?, loss function? (LATER)
    #TODO : add validation while training
    #TODO : write accuracy metric and run validation experiment on trained checkpoints