from __future__ import print_function
from datafuncs import ImageFolderWithRatings, ImageFolderWithFilenamesAndRatings, get_subset
from invert import Invert
import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torch import nn, optim
import torchvision
from torchvision import transforms

class AuDrADataModule_FILENAME(pl.LightningDataModule):
    def __init__(self, args, data_dir = 'primary_images/', ratings_path = 'primary_jrt.csv'):
        super().__init__()
        self.data_dir = data_dir
        self.ratings_path = ratings_path
        self.args = args
        self.transform = transforms.Compose([
            Invert(),
            transforms.Resize(self.args.in_shape[-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [self.args.img_means,self.args.img_means,self.args.img_means],
                std = [self.args.img_stds,self.args.img_stds,self.args.img_stds]
            )
        ])
        self.scaler = MinMaxScaler()

    def setup(self, stage = None):
        #  load and normalize ratings
        ratings = pd.read_csv("primary_jrt.csv", header = None).to_numpy()
        self.scaler.fit(ratings [:,1].reshape((-1,1)))
        ratings[:,1] = self.scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))

        #  load images
        self.data = ImageFolderWithFilenamesAndRatings(
            root = self.data_dir,
            transform = self.transform,
            ratings = ratings
        )

        #  get indices
        train_count = int(len(self.data) * self.args.train_pct)
        val_count = int(len(self.data) * self.args.val_pct)
        indices = torch.randperm(len(self.data))

        data_indices = {"train" : get_subset(indices, 0, train_count),
                             "val" : get_subset(indices, train_count, val_count),
                             "test" : get_subset(indices, train_count + val_count, len(indices))
        }

        self.training_set = Subset(self.data, indices = data_indices["train"])
        self.validation_set = Subset(self.data, indices = data_indices["val"])
        self.test_set = Subset(self.data, indices = data_indices["test"])

    def train_dataloader(self):
        return DataLoader(self.training_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size = self.args.batch_size, drop_last = True, num_workers = 20)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.args.batch_size,  num_workers = 20)

def RaterGeneralizationOneDataloader (args, data_dir = 'rater_generalization_one_images/', ratings_path = 'rg1_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv("rg1_jrt.csv", header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))


    data = ImageFolderWithFilenamesAndRatings(
        root = data_dir,
        transform = transform,
        ratings = ratings
    )
    # print(len(data))
    # print(type(data))
    # print(data[0])
    rg1_loader = DataLoader(data, batch_size = 1)

    return rg1_loader


def FarGeneralizationDataloader (args, data_dir = 'far_generalization_images/', ratings_path = 'fg_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv("fg_jrt.csv", header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))

    data = ImageFolderWithFilenamesAndRatings(
        root = data_dir,
        transform = transform,
        ratings = ratings
    )
    # print(len(data))
    # print(type(data))
    # print(data[0])
    fg_loader = DataLoader(data, batch_size = 1)

    return fg_loader

def RaterGeneralizationTwoDataloader (args, data_dir = 'rater_generalization_two_images/', ratings_path = 'rg2_jrt.csv'):
    transform = transforms.Compose([
        Invert(),
        transforms.Resize(args.in_shape[-1]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [args.img_means,args.img_means,args.img_means],
            std = [args.img_stds,args.img_stds,args.img_stds]
        )
    ])

    scaler = MinMaxScaler()
    ratings = pd.read_csv("rg2_jrt.csv", header = None).to_numpy()
    scaler.fit(ratings [:,1].reshape((-1,1)))
    ratings[:,1] = scaler.transform(ratings[:,1].reshape((-1,1))).reshape((-1,))


    data = ImageFolderWithFilenamesAndRatings(
        root = data_dir,
        transform = transform,
        ratings = ratings
    )
    rg2_loader = DataLoader(data, batch_size = 1)

    return rg2_loader 
