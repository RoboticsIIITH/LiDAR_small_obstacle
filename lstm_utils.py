import numpy as np
import time

class Evaluator:
    def __init__(self,num_classes=3,inp_dim=600):

        self.num_classes = num_classes
        self.pred_labels = []
        self.target_labels = []
        self.pad_dim = inp_dim

    def add_batch(self,pred,target):

        assert pred.shape[1] == target.shape[1], "Prediction and Target dimensions don't match(Incorrect padding)"
        pred,target = pred.copy(),target.copy()
        if pred.shape[1] < self.pad_dim:
            pred = np.append(pred,-100*np.ones((pred.shape[0],self.pad_dim-pred.shape[1])),axis=1)
            target = np.append(target,-100*np.ones((target.shape[0],self.pad_dim-target.shape[1])),axis=1)

        # Pred, target shape: Batch size x 650
        self.pred_labels.append(pred)
        self.target_labels.append(target)

    def _generate_confusion_matrix(self):

        self.pred_labels = np.array(self.pred_labels)
        self.target_labels = np.array(self.target_labels)

        mask = (self.target_labels>=0) & (self.target_labels < self.num_classes)
        # Ignore all padded elements (-100)
        self.target_labels = self.target_labels[mask]
        self.pred_labels = self.pred_labels[mask]

        label = self.num_classes * self.target_labels + self.pred_labels
        label = label.astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        conf_matrix = count.reshape(self.num_classes, self.num_classes)
        return conf_matrix

    def get_metrics(self,class_num=2):
        conf_matrix = self._generate_confusion_matrix()
        conf_matrix = conf_matrix.astype(np.float)

        precision = conf_matrix[class_num,class_num]/conf_matrix[:,class_num].sum()
        recall = conf_matrix[class_num, class_num]/conf_matrix[class_num,:].sum()
        if np.isnan(precision):
            precision = 0.0
        if np.isnan(recall):
            recall = 0.0

        conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        return recall,precision,conf_matrix

    def reset(self):
        self.pred_labels=[]
        self.target_labels=[]
