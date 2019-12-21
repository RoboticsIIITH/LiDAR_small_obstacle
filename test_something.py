import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix,classification_report

# dir_path = "/home/ash/labelme/IIIT_Labels/vindhya_2/region_prop_updated"
# files = os.listdir(dir_path)
# count = 0
# for file_name in files:
#     region = np.load(os.path.join(dir_path,file_name))
#     if len(np.unique(region)) == 1:
#         continue
#     count += 1
# print(count)
y_label = np.random.randint(0,3,(32*16,650))
y_pred = np.random.randint(0,3,(32*16,650))
y_label = y_label.flatten()
y_pred = y_pred.flatten()
start = time.time()
conf_matrix = confusion_matrix(y_label,y_pred,labels=[0,1,2])
conf_matrix.astype(np.float)/conf_matrix.sum(axis=1)[:,np.newaxis]
# report = classification_report(y_label,y_pred,labels=[0,1,2])
print("took:",time.time()-start)