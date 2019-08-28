import os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import time
from matplotlib.animation import FuncAnimation

img_path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1d/image"
results = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/laplacian"

rgb_file = sorted(os.listdir(img_path))
pred_files = sorted(os.listdir(results))

fig1=plt.figure(figsize=(16,10))
ax1 = fig1.add_subplot(121)
#fig2=plt.figure(figsize=(8,5))
ax2 = fig1.add_subplot(122)

file_num = 0
img_num = 0

img = cv2.imread(os.path.join(img_path,rgb_file[file_num]),-1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
im = ax1.imshow(img[:,:,1],cmap='gray')
# img = img
# im = ax1.imshow(img,cmap='gray')

pred = np.load(os.path.join(results,pred_files[img_num]))
print(np.max(pred))
im2 = ax2.imshow(pred,cmap='gray')


def update(i):
    global file_num
    A = cv2.imread(os.path.join(img_path,rgb_file[file_num]),-1)
    A = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
    A= A[:,:,1]
    im.set_array(A)
    file_num += 1
    return im

def update_image(i):
    global img_num
    B = np.load(os.path.join(results,pred_files[img_num]))
    im2.set_array(B)
    img_num += 1
    return im2

ani = FuncAnimation(fig1, update, frames=range(len(rgb_file)), interval=30, blit=False)
ani2 = FuncAnimation(fig1, update_image, frames=range(len(pred_files)), interval=30, blit=False)
plt.show()
ax1.close()
ax2.close()