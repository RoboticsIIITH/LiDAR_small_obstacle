import os
import numpy as np

dir_path = "/home/ash/labelme/IIIT_Labels/vindhya_2/region_prop_updated"
files = os.listdir(dir_path)
count = 0
for file_name in files:
    region = np.load(os.path.join(dir_path,file_name))
    if len(np.unique(region)) == 1:
        continue
    count += 1
print(count)