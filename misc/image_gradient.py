import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

img_path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/image"
results = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/laplacian"

if not os.path.exists(results):
    os.mkdir(results)

rgb_file = sorted(os.listdir(img_path))

for file in rgb_file:
    img = Image.open(os.path.join(img_path, file))
    # img = img.convert('L')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = img[:, :, 1]
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=9)
    # print(np.max(laplacian),np.min(laplacian),laplacian.shape)
    # cv2.imwrite(os.path.join(results,file),laplacian)
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=9)
    np.save(os.path.join(results,file.split('.')[0] + '.npy'),laplacian)
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1), plt.imshow(img,cmap='gray')
    # plt.title('Original')
    # plt.subplot(2, 2, 2), plt.imshow(laplacian,cmap='gray')
    # plt.title('Laplacian')
    # plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X')
    # plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y')
    # plt.show()
