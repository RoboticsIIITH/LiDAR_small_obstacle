import numpy as np
import time

class Evaluator:
    def __init__(self,num_classes=3,inp_dim=600):

        self.num_classes = num_classes
        self.pred_labels = []
        self.target_labels = []
        self.obs_mask = []
        self.pad_dim = inp_dim

    def add_batch(self,pred,target,obs_mask):

        # assert pred.shape[1] == target.shape[1], "Prediction and Target dimensions don't match(Incorrect padding)"
        # pred,target = pred.copy(),target.copy()
        # if pred.shape[1] < self.pad_dim:
        #     pred = np.append(pred,-100*np.ones((pred.shape[0],self.pad_dim-pred.shape[1])),axis=1)
        #     target = np.append(target,-100*np.ones((target.shape[0],self.pad_dim-target.shape[1])),axis=1)

        # Pred, target shape: Batch size x 650
        self.pred_labels.append(pred)
        self.target_labels.append(target)
        # self.obs_mask.append(obs_mask)

    def _generate_confusion_matrix(self,class_num):

        self.pred_labels = np.array(self.pred_labels)
        self.target_labels = np.array(self.target_labels)
        self.obs_mask = np.array(self.obs_mask,dtype=int)

        # small_obstacle_pred = self.pred_labels[self.obs_mask]
        # small_obstacle_target = self.target_labels[self.obs_mask]
        # intersection = (small_obstacle_pred & small_obstacle_target).astype(int)
        # union = (small_obstacle_pred | small_obstacle_target).astype(int)
        # iou = float(sum(intersection) / sum(union))

        mask = (self.target_labels >= 0) & (self.target_labels < self.num_classes)
        # Ignore all padded elements (-100)
        self.target_labels = self.target_labels[mask]
        self.pred_labels = self.pred_labels[mask]

        predictions_class = self.pred_labels == class_num
        target_class = self.target_labels == class_num
        intersection = (predictions_class & target_class).astype(int)
        union = (predictions_class | target_class).astype(int)
        iou_road = float(sum(intersection) / sum(union))

        # label = self.num_classes * self.target_labels + self.pred_labels
        # label = label.astype('int')
        # count = np.bincount(label, minlength=self.num_classes ** 2)
        # conf_matrix = count.reshape(self.num_classes, self.num_classes)
        return iou_road,0

    def get_metrics(self,class_num):
        iou,iou_road = self._generate_confusion_matrix(class_num)
        # conf_matrix = conf_matrix.astype(np.float)

        # precision = conf_matrix[class_num, class_num] / conf_matrix[:, class_num].sum()
        # recall = conf_matrix[class_num, class_num] / conf_matrix[class_num, :].sum()
        # if np.isnan(precision):
        #     precision = 0.0
        # if np.isnan(recall):
        #     recall = 0.0

        # conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        return iou,iou_road

    def reset(self):
        self.pred_labels=[]
        self.target_labels=[]
        self.obs_mask = []
