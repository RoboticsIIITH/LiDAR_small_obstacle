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
from context_network import ContextNet
from squeeze_seg import SqueezeSeg
from tqdm import tqdm

warnings.filterwarnings("ignore")
# torch.set_num_threads(1)


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

    for sample in tqdm(dataloader):

        points, labels = sample['inp'], sample['labels']

        class_weights = calculate_weights_batch(labels, num_classes=2)
        class_weights = torch.from_numpy(class_weights).float()

        points = points.permute(0, 3, 1, 2)
        geometric_mask = points[:,4,:,:].cpu().numpy().squeeze()

        if is_cuda:
            points, labels = points.cuda(), labels.cuda()
            class_weights = class_weights.cuda()

        # Focal loss
        loss_function = FocalLoss(gamma=0, alpha=class_weights)

        if mode == "train":
            optim.zero_grad()
            pred = model.forward(points)
            loss = loss_function.forward(pred,labels)
            loss.backward()
            optim.step()
        else:
            with torch.no_grad():
                pred = model.forward(points)
                loss = loss_function.forward(pred,labels)

        pred_label = torch.argmax(pred, dim=1).detach().cpu().numpy()
        evaluator.add_batch(pred_label,labels.cpu().numpy(),geometric_mask)

        accum_loss += loss.item()
        batch_count += 1
        writer.add_scalar('{}/Loss/iter'.format(mode), loss.item(), epoch * len_dataset + batch_count)

    recall,iou,inp_recall,pred_inp_recall,inp_iou = evaluator.get_metrics(class_num=1)

    # Metric for recall of geometric contexts
    writer.add_scalar('{}/Loss/epoch'.format(mode), accum_loss / batch_count, epoch)
    writer.add_scalar('{}/Precision'.format(mode),iou, epoch)
    writer.add_scalar('{}/Net Recall'.format(mode),recall,epoch)
    writer.add_scalar('{}/Input Contexts Recall'.format(mode), inp_recall, epoch)
    writer.add_scalar('{}/Input Contexts Precision'.format(mode), inp_iou, epoch)
    writer.add_scalar('{}/Pred Contexts Recall'.format(mode), pred_inp_recall, epoch)


    print("Mode : {},Epoch : {}".format(mode,epoch))
    print("Recall: {}, Precision : {}, Inp Precision :{}".format(recall,iou,inp_iou))
    print("Input Recall: {}, Pred and Input Recall : {}".format(inp_recall,pred_inp_recall))
    return recall,[]


def visualise_pred(model,dataset):

    model.eval()
    for sample in dataset:
        image, points, points_label = sample['image'],sample['inp'],sample['labels']
        seq_len, proj_points = sample['ring_lengths'],sample['proj_points']

        # points = points.unsqueeze(0)
        # points = points.permute(0, 3, 1, 2)

        # with torch.no_grad():
        #     pred = model.forward(points)

        # pred = np.argmax(pred.cpu().numpy(),axis=1).squeeze()
        # geometric_mask = points[:, 4, :, :].cpu().numpy().squeeze()
        # context_mask = geometric_mask == 1
        # pred = (pred & context_mask).astype(int)

        # print(np.unique(pred_out))

        image = image.cpu().numpy()
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        proj_points = proj_points.cpu().numpy()
        seq_len = seq_len.cpu().numpy()
        points = points.numpy()
        points_label = points_label.numpy()
        # points_label[points_label == -100] = 0
        points_label = 255*points_label.astype(np.uint8)
        points_label = points_label[::-1,:]
        depth = points[::-1,:,3]
        depth = (depth-np.min(depth))/(np.max(depth)-np.min(depth))
        depth = 255*depth.astype(np.uint8)
        cv2.imshow("range",depth)
        # print(np.unique(points_label))
        # plt.imshow(points_label,cmap='gray')
        # plt.show()

        # for i in range(points.shape[2]):
        #     len = int(seq_len[i])
        #     projection_pts = proj_points[i,:len]
        #     pred = points[i,:,4]
        #     pred_out = pred[i]
        #     for j in range(projection_pts.shape[0]):
        #         x_0, y_0 = int(projection_pts[j, 1]), int(projection_pts[j, 0])
        #         if pred[i,y_0] == 1:
        #             pt_color = (0,0,255)
        #         else:
        #             pt_color = (0,255,0)
        #
        #         cv2.circle(image,(y_0,x_0),2,pt_color,thickness=1)

        cv2.imshow("feed",image)
        if cv2.waitKey(10) == ord('q'):
            print('Quitting....')
            break
        cv2.waitKey(0)


if __name__ == '__main__':

    visualise_mode = True
    exp_num = "exp_6"
    num_epochs = 20
    save_interval = 1

    # model = ContextNet(inp_size=5,out_size=2)
    model = SqueezeSeg()

    if not visualise_mode:
        batch_size = 4
        train_dataset = ProjSet("/media/ash/OS/IIIT_Labels/train/", class_num=2, split="train")
        val_dataset = ProjSet("/media/ash/OS/IIIT_Labels/val/", class_num=2, split="val")
        # use_gpu = not visualise_mode
        use_gpu = True
        print("Using GPU: {}".format(use_gpu))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

        if use_gpu:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # poly_decay = lambda epoch: pow((1-epoch/num_epochs),0.9)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=poly_decay)

        log_dir = "../context_network_logs/logs/"
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
                recall, confusion_mat = run_epoch(model, optimizer, val_loader, evaluator, writer, epoch=i, mode="val",is_cuda=use_gpu)
                # writer.add_text("confusion matrix on val set", str(list(confusion_mat)), global_step=i)
                # print("Saving checkpoint for Epoch:{}".format(i))
                checkpoint = {'epoch': i,
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                dir_path = os.path.join('../context_network_logs/checkpoints',exp_num)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                file_path = os.path.join(dir_path, 'checkpoint_{}.pth.tar'.format(i))
                torch.save(checkpoint, file_path)
                if recall > best_recall:
                    shutil.copyfile(file_path, os.path.join(dir_path, 'best_model.pth.tar'))
                    best_recall = recall

    else:
        test_dataset = ProjSet("/media/ash/OS/IIIT_Labels/val/", class_num=2, split="test")
        # test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=4)
        # checkpoint = torch.load('../context_network_logs/checkpoints/checkpoint_12.pth.tar', map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'])
        visualise_pred(model,test_dataset)