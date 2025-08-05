from argparse import ArgumentParser
from AuDrADataModule_JRT_CHECK_IMGS_AND_FNAMES import AuDrADataModule_FILENAME, RaterGeneralizationOneDataloader, RaterGeneralizationTwoDataloader, FarGeneralizationDataloader
from AuDrA_pl_plus_imgsums import AuDrA
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch import nn, optim
import torchvision.models as models

#  Set seed
torch.manual_seed(42)

#  Hyperparams & Config
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--architecture', default = 'resnet18')
parser.add_argument('--pretrained', default = True)
parser.add_argument('--in_shape', default = [3,224,224], type = int)
# parser.add_argument('--in_shape', default = [3,400,400], type = int)
parser.add_argument('--img_means', default = 0.1612 , type = float)
parser.add_argument('--img_stds', default = 0.4075, type = float)
parser.add_argument('--num_outputs', default = 1, type = int)
parser.add_argument('--learning_rate', default = 0.00034664640432471026)
parser.add_argument('--batch_size', default = 16)
# parser.add_argument('--batch_size', default = 1)
parser.add_argument('--train_pct', default = 0.7, type = float)
parser.add_argument('--val_pct', default = 0.1, type = float)
parser.add_argument('--loss_func', default = nn.MSELoss(), type = object)
parser.add_argument('--num_workers', default = 20)
args = parser.parse_args()


#  Init DataModule & Model
early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=0.00, patience=3, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='val_correlation',
    mode='max',
    prefix='',
)

dm = AuDrADataModule_FILENAME(args = args)
dm.setup()
model = AuDrA(args)
tb_logger = pl.loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer.from_argparse_args(args,
                                        gpus = 1,
                                        num_nodes = 1,
                                        deterministic = True,
                                        # distributed_backend = 'dp',
                                        distributed_backend = None,
                                        max_epochs = 136,
                                        # max_epochs = 1,
                                        precision = 16,
                                        logger = tb_logger,
                                        checkpoint_callback = checkpoint_callback,
                                        # early_stop_callback = early_stop_callback,
)
trainer.fit(model, dm)
traindf = pd.DataFrame.from_dict(model.train_storage_dict)
traindf["phase"] = "train"
# traindf.to_csv('training_storage_dataframe.csv', index = False)
traindf.to_csv('training_storage_dataframe_fnames.csv', index = False)

valdf = pd.DataFrame.from_dict(model.val_storage_dict)
valdf["phase"] = "validation"
# traindf.to_csv('training_storage_dataframe.csv', index = False)
valdf.to_csv('validation_storage_dataframe_fnames.csv', index = False)
trainer.test()

trainer.test()
testdf = pd.DataFrame.from_dict(model.test_storage_dict)
testdf["phase"] = "test"
# traindf.to_csv('training_storage_dataframe.csv', index = False)
testdf.to_csv('test_storage_dataframe_fnames.csv', index = False)