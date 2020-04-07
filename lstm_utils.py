import numpy as np
import time

class Evaluator:
    def __init__(self,num_classes):

        self.num_classes = num_classes
        self.pred_labels = None
        self.target_labels = None
        self.obs_mask = None

    def add_batch(self,pred,target,obs):

        if self.pred_labels is not None:

            self.pred_labels = np.append(self.pred_labels,pred,axis=0)
            self.target_labels = np.append(self.target_labels,target,axis=0)
            self.obs_mask = np.append(self.obs_mask,obs,axis=0)
        else:
            self.pred_labels = pred
            self.target_labels = target
            self.obs_mask = obs

    def _generate_confusion_matrix(self,class_num):

        # Ignore all padded elements (-100)
        mask = (self.target_labels >= 0) & (self.target_labels < self.num_classes)
        self.target_labels = self.target_labels[mask]
        self.pred_labels = self.pred_labels[mask]
        self.obs_mask = self.obs_mask[mask]

        predictions_class = self.pred_labels == class_num
        target_class = self.target_labels == class_num

        intersection = (predictions_class & target_class).astype(int)
        # union = (predictions_class | target_class).astype(int)
        # iou = float(sum(intersection) / sum(union))
        recall = float(sum(intersection) / sum(target_class.astype(int)))
        precision = float(sum(intersection) / sum(predictions_class.astype(int)))

        # geometric_pred = self.obs_mask == 1
        # geometric_intersection = (geometric_pred & target_class)  # Geometric contexts that were right
        # geometric_recall = float(sum(geometric_intersection.astype(int)) / sum(target_class.astype(int)))
        # geometric_union = (geometric_pred | target_class).astype(int)
        # geometric_precision = float(sum(geometric_intersection.astype(int)) / sum(geometric_pred.astype(int)))
        # pred_geometric_recall = float(sum(predictions_class & geometric_intersection).astype(int) / sum(geometric_intersection.astype(int)))

        # label = self.num_classes * self.target_labels + self.pred_labels
        # label = label.astype('int')
        # count = np.bincount(label, minlength=self.num_classes ** 2)
        # conf_matrix = count.reshape(self.num_classes, self.num_classes)
        geometric_recall = 0
        geometric_precision = 0
        pred_geometric_recall = 0
        return recall,precision,geometric_recall,pred_geometric_recall,geometric_precision

    def get_metrics(self,class_num):
        recall,iou,input_recall,pred_inp_recall,inp_iou = self._generate_confusion_matrix(class_num)
        # conf_matrix = conf_matrix.astype(np.float)

        # precision = conf_matrix[class_num, class_num] / conf_matrix[:, class_num].sum()
        # recall = conf_matrix[class_num, class_num] / conf_matrix[class_num, :].sum()
        # if np.isnan(precision):
        #     precision = 0.0
        # if np.isnan(recall):
        #     recall = 0.0

        # conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        return recall,iou,input_recall,pred_inp_recall,inp_iou

    def reset(self):
        self.pred_labels=None
        self.target_labels=None
        self.obs_mask = None
