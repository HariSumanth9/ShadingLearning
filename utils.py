import torch
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def dice_coefficient(pred, target):
    smooth = 1e-15
    num = pred.size()[0]
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()
    intersection = (m1*m2).sum(1)
    union = (m1 + m2).sum(1) + smooth - intersection
    score = intersection/union
    return score.mean()


def get_weights(labels_batch):
    weights = np.array([])
    labels_batch_numpy = labels_batch.numpy()
    n = labels_batch_numpy.shape[0]
    labels_batch_numpy = labels_batch_numpy.astype('uint8')
    for i in range(n):
        label = labels_batch_numpy[i][0]
        trnsf = distance_transform_edt(label)
        trnsf = ((np.abs((trnsf.max() - trnsf))/trnsf.max())*(label)+1)
        trnsf = trnsf.flatten()
        weights = np.concatenate((weights, trnsf))
    weights = torch.from_numpy(weights)
    return weights