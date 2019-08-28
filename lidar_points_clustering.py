from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from PIL import Image
import scipy.cluster as cluster
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors

path = "/media/ash/OS/small_obstacle_bag/synced_data/file_1/"

label_files = sorted(os.listdir(os.path.join(path,"labels")))
velodyne_files = sorted(os.listdir(os.path.join(path,"depth")))

cloud_files = [os.path.join(path+"depth/",x) for x in label_files]
label_files = [os.path.join(path+"labels/",x) for x in label_files]

depth = Image.open(cloud_files[25])
depth = np.array(depth)
x=[]
y=[]
value = []
gt = Image.open(label_files[25])
gt = gt.convert('L')
gt = np.array(gt)
road_x,road_y = np.where(gt == 38)
max_road_x = np.max(road_x)
min_road_x = np.min(road_x)
max_road_y = np.max(road_y)
min_road_y = np.min(road_y)

for i in range(depth.shape[0]):
    for j in range(depth.shape[1]):
        if depth[i,j] == 0:
            continue
        if i<max_road_x and i>min_road_x and j<max_road_y and j>min_road_y and gt[i,j]!=0:
            x.append(-i)
            y.append(j)
            value.append(depth[i, j])


feature_vector = [[x[i],y[i],value[i]] for i in range(len(value))]
feature_vector = np.array(feature_vector)
norm_feature_vector = cluster.vq.whiten(feature_vector)
value = np.array(value)
value = value[:,np.newaxis]

"""k means clustering"""
cluster,distortion = cluster.vq.kmeans(norm_feature_vector,k_or_guess=150)

"""SVM clustering"""
# clf = OneClassSVM(gamma='auto',kernel='poly',degree=3).fit(value)
# pred = clf.predict(value)

"Isolation forest clustering"
clf = IsolationForest(contamination=0.1).fit(value)
pred = clf.predict(value)

# clf = LocalOutlierFactor().fit(feature_vector)
# pred = clf.fit_predict(feature_vector)

"""Nearest Neighbours"""
# nbrs = NearestNeighbors(n_neighbors=5,radius=100).fit(value)
# distances, indices = nbrs.kneighbors(value)
# print(np.argmin(distances,axis=1),distances)

colors=[]
for elem in pred:
    if elem == -1:
        colors.append('red')
    elif elem == 1:
        colors.append('blue')

fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xs=feature_vector[:,1],ys=feature_vector[:,2],zs=feature_vector[:,0],s=1,color=colors)
# ax.scatter(xs=cluster[:,1],ys=cluster[:,2],zs=cluster[:,0],s=10,color=colors)
plt.show()
