import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, root, opt, transform = None):
        self.root        = root
        self.images      = sorted(os.listdir(root+opt))
        self.opt         = opt
        self.transform   = transform
        
    def __getitem__(self, index):
        image      = imageio.imread('/home/sumanth/imagenet/ILSVRC/images_224/'+self.opt+self.images[index])
        shading    = imageio.imread('/home/sumanth/imagenet/ILSVRC/shading_112/'+self.opt+self.images[index])
        if(len(image.shape)!=3):
            image_ = np.zeros((224, 224, 3), dtype=np.uint8)
            image_[:, :, 0] = image
            image_[:, :, 1] = image
            image_[:, :, 2] = image
            image = image_
        image      = self.transform(image)
        shading    = shading/255
        shading    = torch.from_numpy(shading).float()
        return image, shading
        
    def __len__(self):
        return len(self.images)