"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
from argparse import ArgumentParser
from AuDrADataModule import AuDrADataModule
from AuDrA_Pl_1_2_1_hparamversion import AuDrA
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchvision.models as models
import optuna
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback
import os

#  Set Vars
seed = torch.manual_seed(42)
DIR = 'optuna_dir'
MODEL_DIR = os.path.join(DIR, "result")
STUDY_NAME = "AuDrA_search"
STORAGE_NAME = "sqlite:///{}.db".format(STUDY_NAME)

#  Hyperparams & Config
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--architecture', default = 'resnet34')
parser.add_argument('--pretrained', default = True)
parser.add_argument('--in_shape', default = [3,224,224], type = int)
parser.add_argument('--img_means', default = 0.1612 , type = float)
parser.add_argument('--img_stds', default = 0.4075, type = float)
parser.add_argument('--num_outputs', default = 1, type = int)
parser.add_argument('--learning_rate', default = 1e-4)
parser.add_argument('--batch_size', default = 128)
parser.add_argument('--train_pct', default = 0.7, type = float)
parser.add_argument('--val_pct', default = 0.1, type = float)
parser.add_argument('--loss_func', default = nn.MSELoss(), type = object)
parser.add_argument('--num_workers', default = 20)
args = parser.parse_args()

# Hparam updater func (update hparams in place)
def update_args_(args, params):
  dargs = vars(args)
  dargs.update(params)

#  Define search objective
def objective(trial):

    hparams = {
        'architecture': trial.suggest_categorical('architecture', choices = ['resnet18', 'resnet34']),
        'pretrained': trial.suggest_categorical('pretrained', choices = [True, False]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', choices = [16, 32]),
        }
    update_args_(args, hparams) # update hparams in place

    dm = AuDrADataModule(args = args) # leave this outside the train func?
    model = AuDrA(args)

    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus = 2,
                                            num_nodes = 1,
                                            distributed_backend = 'dp',
                                            max_epochs = 150,
                                            precision = 16,
                                            deterministic = True,
                                            checkpoint_callback = False,
                                            logger = True,
                                            progress_bar_refresh_rate = 100,
                                            callbacks = [PyTorchLightningPruningCallback(trial, monitor = "val_pred_loss")],
                                            )
    trainer.fit(model, dm)

    return trainer.callback_metrics["val_pred_loss"].item()


def run_study():
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(study_name = STUDY_NAME, storage = STORAGE_NAME, direction = "minimize", pruner = pruner, load_if_exists = True)
    study.optimize(objective, n_trials = 500, show_progress_bar=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

run_study()
