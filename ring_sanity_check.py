import os
import sys
import cv2
import numpy as np
from PIL import Image

path = '/Users/aditya/Documents/Code/aditya/iiit_research/iiit_data/train/seq_2/'
ring_path = os.path.join(path, "rings")

rings = sorted(os.listdir(ring_path))
ring_paths = []
for ring in rings:
    ring_paths.append(os.path.join(ring_path, ring))
img = Image.open(ring_paths[90]).convert('P')
# img = Image.open(ring_paths[1])
print('Mode: ', img.mode)
print(img.size)
img.show()
print(np.unique(np.asarray(img), return_counts=True))
# cv2.imshow('ring', img)
# if cv2.waitKey(0) == ord('q'):
#     sys.exit()

