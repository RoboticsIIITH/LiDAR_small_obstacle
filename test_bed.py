import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_2/velodyne/"
lidar_files = [os.path.join(path,file) for file in sorted(os.listdir(path))]

data = np.load(lidar_files[100])
ring_info = data[:,4]
data = data[ring_info<6]
# data = data[data[:,1]>0]
fig = plt.figure()
ax = fig.add_subplot('111',projection='3d')
# data = data[data[:,1]>0]
data = data[data[:,2]<-1.2]
# data = data[data[:,0]<5]
# data = data[data[:,0]>-5]
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=10)
# A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
C, _, _, _ = linalg.lstsq(A, data[:, 2])
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
XX=X.flatten()
YY=Y.flatten()
# Z = C[0] * X + C[1] * Y + C[2]
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
plt.show()
print(data.shape)

