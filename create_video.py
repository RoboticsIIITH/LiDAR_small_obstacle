import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
img_array = []
path = '/home/ash/labelme/IIIT_Labels/seq_5/SegmentationClassVisualization/'
files = sorted(os.listdir(path))[60:240]
for file in files:
    img = cv2.imread(os.path.join(path,file))
    img_array.append(img)
    size = (img.shape[1],img.shape[0])

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('banner_vid.mp4',fourcc,20.0,size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()