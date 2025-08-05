"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from __future__ import print_function
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models


class AuDrA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.args = args
        self.validation_vals = []

        # INITIALIZE MODEL
        try:
            self.model = models.__dict__[self.args.architecture](pretrained=self.args.pretrained)
        except:
            raise Exception('invalid architecture specified')

        # REMOVE LAST FULLY CONNECTED LAYER
        last_layer = list(self.model.children())[-1]
        if isinstance(last_layer, nn.Sequential):
            count = 0
            for layer in last_layer:
                if isinstance(layer, nn.Linear):
                    # fetch the first of the many Linear layers
                    count += 1
                    in_features = layer.in_features
                if count == 1:
                    break
        elif isinstance(last_layer, nn.Linear):
              in_features = last_layer.in_features

        # DEFINE NEW REGRESSION HEAD
        classifier = nn.Sequential(OrderedDict([
            ('bc1', nn.BatchNorm1d(in_features)),
            ('relu1', nn.ReLU()),
            ('fc1', nn.Linear(in_features, self.args.num_outputs, bias=True)),
        ]))

        # REPLACE THE CLASSIFIER WITH THE NEW REGRESSION HEAD
        if self.model.__dict__['_modules'].get('fc', None):
            self.model.fc = classifier
        else:
            self.model.classifier = classifier


    def forward(self, x):
        return self.model(x)

    def loss(self, pred_rating, true_rating):
        loss = self.args.loss_func(pred_rating, true_rating.view(-1,1))
        mae = torch.mean(torch.abs(pred_rating - true_rating))
        return loss, mae

    def training_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        return pred_loss

    def validation_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        self.validation_vals.append(pred)
        val_pred_loss, mae = self.loss(pred, ratings)

        self.log("val_pred_loss", val_pred_loss, on_step = True, on_epoch = True, sync_dist = True)
        self.log("hp_metric", val_pred_loss, on_step = False, on_epoch = True, sync_dist = True)


    def test_step(self, batch, batch_idx):
        images, labels, ratings = batch
        ratings = ratings.float()

        pred = self(images)
        pred_loss, mae = self.loss(pred, ratings)

        self.log("pred_loss_test", pred_loss)


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.args.learning_rate)
