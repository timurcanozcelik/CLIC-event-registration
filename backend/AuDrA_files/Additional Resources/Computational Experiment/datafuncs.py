from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
# from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import SubsetRandomSampler
from torch import nn, optim
from torch.autograd import Variable
import torchvision

class ImageFolderWithRatings(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ratings):
        super().__init__(root, transform)
        self.ratings = ratings
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithRatings, self).__getitem__(index)
        path = self.imgs[index][0]
        fname = path.split("/")[-1]
        fname = fname.split(".")[0]
        # print(path)
        # print(fname)
        rating = self.ratings[np.where(self.ratings[:, 0] == fname)][0][1]
        # rating = rating.astype(float32)
        tuple_with_rating= (original_tuple + (rating,))
        return tuple_with_rating

def get_subset(indices, start, end):
    return indices[start : start + end]
