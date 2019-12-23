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
from focal_loss import FocalLoss

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


def scale_inp(inp):
    """Scale inp array between range 0-1"""

    assert inp.shape[2] == 4,"Incorrect inp feature dimension: Required 4-dim features"
    for axis in range(inp.shape[2]):
        inp[:,:,axis] = (inp[:,:,axis]-torch.min(inp[:,:,axis]))/(torch.max(inp[:,:,axis])-torch.min(inp[:,:,axis]))
    return inp


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

        # Scale/Normalise input to 0-1 range for each feature
        points = scale_inp(points)

        # Sort rings by length in descending order (required for pack_pad_sequence function)
        # sorted_rings = torch.argsort(seq_lengths, descending=True)
        # points = points[sorted_rings]
        # labels = labels[sorted_rings]
        # seq_lengths = seq_lengths[sorted_rings]

        # Convert label tensor to form: BxT where B=batch size and T = max length of sequence in batch
        # max_len = torch.max(seq_lengths)
        # labels = labels[:,:int(max_len.item())]

        # Shuffle rings batch
        shuffle_indexes = torch.randperm(points.shape[0])
        points,labels,seq_lengths = points[shuffle_indexes],labels[shuffle_indexes],seq_lengths[shuffle_indexes]

        # Two class label: Obstacle and Road
        # small_obstacle_mask = labels == 2
        # small_obstacle_mask = small_obstacle_mask.cpu().numpy()
        labels[labels == 2] = 0
        # print(np.unique(labels.numpy()))

        class_weights = calculate_weights_batch(labels, num_classes=2)
        class_weights = torch.from_numpy(class_weights).float()
        # print("Class weights", class_weights)

        # get only x,y,z features
        # points = points[:,:,:3]
        points = points.permute(0,2,1)

        if is_cuda:
            points, labels, seq_lengths = points.cuda(), labels.cuda(), seq_lengths.cuda()
            class_weights = class_weights.cuda()

        # Focal loss
        loss_function = FocalLoss(gamma=0, alpha=class_weights)

        if mode == "train":
            optim.zero_grad()
            pred = model.forward(points,seq_lengths)
            # print("pred shape",pred.shape)
            # loss = model.compute_loss(pred, labels, weight=class_weights)
            loss = loss_function.forward(pred,labels)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                pred = model.forward(points, seq_lengths)
                # loss = model.compute_loss(pred, labels, weight=class_weights)
                loss = loss_function.forward(pred,labels)

        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
        evaluator.add_batch(pred_label,labels.cpu().numpy(),[])

        accum_loss += loss.item()
        batch_count += 1
        writer.add_scalar('{}/Loss/iter'.format(mode), loss.item(), epoch * len_dataset + batch_count)

    iou,iou_road = evaluator.get_metrics(class_num=1)
    writer.add_scalar('{}/Loss/epoch'.format(mode), accum_loss / batch_count, epoch)
    writer.add_scalar('{}/IOU/small_obstacle'.format(mode),iou, epoch)
    writer.add_scalar('{}/IOU/Road'.format(mode),iou_road,epoch)
    # writer.add_scalar('{}/Recall/road'.format(mode),recall_road, epoch)

    # print("{}_loss:{},epoch:{}".format(mode, accum_loss / batch_count, epoch))
    print("{}_IOU:{},epoch:{}".format(mode,iou,epoch))
    return iou,[]


def visualise_pred(model,dataset):
    model.eval()
    for sample,_ in dataset:
        image, points, points_label = sample['image'],sample['point_cloud'],sample['labels']
        seq_len, proj_points = sample['ring_lengths'],sample['proj_points']

        # Two class label : Off-road and Road
        # points_label[points_label == 2] = 0

        # points = scale_inp(points)
        # sorted_rings = torch.argsort(seq_len,descending=True)
        # points = points[sorted_rings]
        # seq_len = seq_len[sorted_rings]
        # proj_points = proj_points[sorted_rings]
        # points = points[:,:,:3]
        points = points.permute(0, 2, 1)
        points_label = points_label.numpy()

        with torch.no_grad():
            pred = model.forward(points, seq_len)

        seq_len = seq_len.cpu().numpy()
        pred_softmax = F.softmax(pred, dim=1).numpy()
        pred_out = np.argmax(pred_softmax,axis=1).squeeze()
        # print(np.unique(pred_out))
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
                # elif pred_labels[i] == 2:
                #     pt_color = (0,0,255)
                else:
                    pt_color = (0,0,255)
                cv2.circle(image,(int(projection_pts[i,0]),int(projection_pts[i,1])),2,pt_color,thickness=1)
        cv2.imshow("feed",image)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break
        # cv2.waitKey(0)


if __name__ == '__main__':

    visualise_mode = True
    batch_size = 1
    num_epochs = 50
    save_interval = 1

    model = Conv1d(num_filters=128,kernel_size=3,inp_dim=4,num_classes=3,batch_size=16*batch_size,dropout_rate=0.2)

    if not visualise_mode:
        train_dataset = ProjSet("/scratch/ash/IIIT_Labels/train/", class_num=2, split="train")
        val_dataset = ProjSet("/scratch/ash/IIIT_Labels/val/", class_num=2, split="val")
        use_gpu = not visualise_mode
        print("Using GPU for training....")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        if use_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # poly_decay = lambda epoch: pow((1-epoch/num_epochs),0.9)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=poly_decay)

        log_dir = "/scratch/ash/new_ring_wise/logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(os.path.join(log_dir, "exp_1"))
        writer.add_text(text_string=str(list(model.children())), tag='model_info')
        evaluator = Evaluator(num_classes=2)

        # checkpoint = torch.load('/scratch/ash/lstm_logs_2/checkpoints/exp_6/checkpoint_99.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

        best_recall = 0.0

        for i in range(0, num_epochs):
            start = time.time()
            _, __ = run_epoch(model, optimizer, train_loader, evaluator, writer, epoch=i, mode="train", is_cuda=use_gpu)
            print("Epoch took:", time.time() - start)
            # scheduler.step(epoch=i)

            if i % save_interval == 0 and i != 0:
                recall, confusion_mat = run_epoch(model, optimizer, val_loader, evaluator, writer, epoch=i, mode="val",
                                                  is_cuda=use_gpu)
                # writer.add_text("confusion matrix on val set", str(list(confusion_mat)), global_step=i)
                print("Saving checkpoint for Epoch:{}".format(i))
                checkpoint = {'epoch': i,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                dir_path = os.path.join('/scratch/ash/new_ring_wise/checkpoints', 'exp_1')
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                file_path = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(i))
                torch.save(checkpoint, file_path)
                if recall > best_recall:
                    shutil.copyfile(file_path, os.path.join(dir_path, 'best_model.pth.tar'))
                    best_recall = recall

    else:
        test_dataset = ProjSet("/home/ash/labelme/IIIT_Labels/val/", class_num=2, split="test")
        checkpoint = torch.load('checkpoints/exp_8/best_model.pth.tar', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        visualise_pred(model, test_dataset)
