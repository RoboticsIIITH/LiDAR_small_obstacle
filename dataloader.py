import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torchvision import transforms


class SmallObs(data.Dataset):

    def __init__(self, dir_paths, class_num):

        self.root_paths = dir_paths
        self.img_paths = []
        self.lidar_paths = []
        self.label_paths = []
        for path in self.root_paths:
            self.img_paths.append(os.path.join(path,"image"))
            self.lidar_paths.append(os.path.join(path, "velodyne"))
            self.label_paths.append(os.path.join(path, "labels"))


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):

        input_path = self.file_paths[index]
        temp=input_path.split('labels')
        img_path = temp[0] + 'image' + temp[1]
        depth_path = temp[0] + 'depth' + temp[1]

        _img = np.array(Image.open(img_path))
        _depth = np.array(Image.open(depth_path),dtype=np.float)
        assert np.max(_depth) > 255. , "Found 8 bit depth, 16 bit depth is required"
        _depth = _depth/256.																# Converts 16 bit uint depth to 0-255 float
        _target = np.asarray(Image.open(input_path))

        sample={'image':_img,'label':_target,'depth':_depth}

        if self.split == 'train':
            return self.transform_tr(sample)

        elif self.split == 'val':
            return self.transform_val(sample)

        elif self.split == 'test':
            return self.transform_ts(sample)





"""
if __name__ == '__main__':
    #from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512

    cityscapes_train = SmallObs(args,split='train')
    trainloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)
    images,labels=next(iter(trainloader))
    print(images.shape,labels.shape,type(labels))
"""