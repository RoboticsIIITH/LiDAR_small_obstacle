import cv2
import os
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/groundTruth_denser/"
files =  sorted(os.listdir(path))
img = cv2.imread(os.path.join(path,files[0]),-1)
img = np.array(img,dtype=np.float)
img = img/256.
points = np.where(img!=0)
points_2 = np.array(points)
points_2 = points_2.transpose()
values = [img[x,y] for x,y in points_2]
non_points = np.where(img == 0)
non_points = np.array(non_points)
non_points = non_points.transpose()
dense = interpolate.griddata(points,values,method='cubic',xi=non_points)
#dense = interpolate.CloughTocher2DInterpolator(points,values)
#dense = interpolate.LinearNDInterpolator(points,values)
#values_dense = dense(non_points)
#print(np.unique(dense)[:100])

# TODO: why doesn't normal interpolation work?

img = np.array(img,dtype=np.uint8)
near_img = cv2.resize(img,dsize=(1280,720), interpolation = cv2.INTER_CUBIC)
plt.imshow(near_img,cmap='jet')
plt.show()