import sys, os
#import cv2
import glob
import copy
import random
import numpy as np
import pandas as pd
import arg_parser
#from PIL import Image
#import matplotlib.image as mpimage



import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Normalize
import torchvision.models as models


from train3 import train
from net import ResnetUnetHybrid, Bottleneck
from load_data import ImageNetDataset



def main(args):
    trainLossList = []
    validLossList = []
    use_gpu = torch.cuda.is_available()

    transform     = transforms.Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], inplace = True)
    ])

    trainDataset = ImageNetDataset('/home/sumanth/imagenet/ILSVRC/images_224/', 'train/', transform)
    validDataset = ImageNetDataset('/home/sumanth/imagenet/ILSVRC/images_224/', 'val/', transform)

    trainLoader  = DataLoader(trainDataset, batch_size = 32, shuffle = True,  num_workers = 32)
    validLoader  = DataLoader(validDataset, batch_size = 32, shuffle = False, num_workers = 32)

    net = ResnetUnetHybrid(Bottleneck, [3, 4, 6, 3])
    #net.fc = nn.Linear(2048, 1024, bias = True)


    if use_gpu:
        net = net.cuda(0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)

    model = {
        'net': net,
        'criterion': criterion,
        'optimizer': optimizer
    }

    print('Training Begins')
    trainLossList, validLossList = train(model, trainLoader, validLoader, use_gpu, num_epochs = 1000)
    trainLossList = np.array(trainLossList)
    validLossList = np.array(validLossList)
    DF = pd.DataFrame({'Train Losses':trainLossList, 'Valid Losses': validLossList})
    DF.to_csv('/home/sumanth/imagenet/ILSVRC/results/losses_df.csv')
    print('Training Ends')
    
if __name__ == '__main__':
    args = arg_parser.parse_arguments()
    main(args)




