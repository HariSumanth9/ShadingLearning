import warnings
warnings.filterwarnings("ignore")

import os
import tqdm
import numpy as np
import imageio
import torch

def train(model, trainLoader, validLoader, use_gpu, num_epochs):
    trainLossList = []
    validLossList = []
    bestValidLoss = 0
    net       = model['net']
    optimizer = model['optimizer']
    criterion = model['criterion']
    for epoch in range(num_epochs):
        trainRunningLoss = 0
        validRunningLoss = 0
        num              = 0
        iteration        = 0
        
        net.train(True)
        for data in tqdm.tqdm(trainLoader):
            images, targets = data
            if use_gpu:
                images  = images.cuda(0)
                targets = targets.cuda(0)
            outputs = net(images)
            optimizer.zero_grad()
            loss = criterion(outputs.view(-1), targets.view(-1))
            loss.backward()
            optimizer.step()
            if(num%2000==0):
                torch.save(net.state_dict(), '/home/sumanth/imagenet/saved_models/last_even_iter.pth')
            if(num%2000==1):
                torch.save(net.state_dict(), '/home/sumanth/imagenet/saved_models/last_odd_iter.pth')
            num += 1
            trainRunningLoss += loss.item()
        trainRunningLoss = trainRunningLoss/len(trainLoader)
        
        
        net.train(False)
        for data in tqdm.tqdm(validLoader):
            images, targets = data
            if use_gpu:
                images  = images.cuda(0)
                targets = targets.cuda(0)
            outputs = net(images)
            loss    = criterion(outputs.view(-1), targets.view(-1))
            validRunningLoss += loss.item()
            if(iteration == 0):
                for i in range(len(outputs)):
                    image  = images[i]
                    target = targets[i]
                    output = outputs[i][0]
                    image  = image.detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    output = output.detach().cpu().numpy()
                    image[0] = image[0]*0.229 + 0.485
                    image[1] = image[1]*0.224 + 0.456
                    image[2] = image[2]*0.225 + 0.406
                    image  = image.transpose(1, 2, 0)
                    image  = image*255
                    target = target*255
                    output = output*255
                    image  = image.astype(np.uint8)
                    target = target.astype(np.uint8)
                    output = output.astype(np.uint8)
                    f_name = '/home/sumanth/imagenet/ILSVRC/results/images/'  + str(epoch) + "_" + str(i) + '.jpg'
                    imageio.imwrite(f_name, image)
                    f_name = '/home/sumanth/imagenet/ILSVRC/results/targets/' + str(epoch) + "_" + str(i) + '.jpg'
                    imageio.imwrite(f_name, target)
                    f_name = '/home/sumanth/imagenet/ILSVRC/results/outputs/' + str(epoch) + "_" + str(i) + '.jpg'
                    imageio.imwrite(f_name, output)
                iteration += 1
        torch.save(net.state_dict(), '/home/sumanth/imagenet/saved_models/last_epoch.pth')
        validRunningLoss = validRunningLoss/len(validLoader)
        if epoch == 0:
            bestValidLoss = validRunningLoss
            torch.save(net.state_dict(), '/home/sumanth/imagenet/saved_models/best_epoch.pth')
        else:
            if(validRunningLoss < bestValidLoss):
                bestValidLoss = validRunningLoss
                torch.save(net.state_dict(), '/home/sumanth/imagenet/saved_models/best_epoch.pth')
        
        trainLossList.append(trainRunningLoss)
        validLossList.append(validRunningLoss)
        print('[|Epoch: {:.0f}/{:.0f}| |Train Loss: {:.5f}| |Valid Loss: {:.5f}|]'.format(epoch+1, num_epochs, trainRunningLoss, validRunningLoss))
    return trainLossList, validLossList


