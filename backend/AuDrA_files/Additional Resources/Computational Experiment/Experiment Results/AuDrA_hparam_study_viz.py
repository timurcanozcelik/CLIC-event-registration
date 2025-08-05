"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022, December 2). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity. https://doi.org/10.31234/osf.io/t63dm

"""
import optuna
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback
import os
import pandas as pd
import sqlite3

#  Set Vars
seed = torch.manual_seed(42)
DIR = 'optuna_dir'
MODEL_DIR = os.path.join(DIR, "result")
STUDY_NAME = "AuDrA_search"
STORAGE_NAME = "sqlite:///{}.db".format(STUDY_NAME)

# Load study & grab best vals
study = optuna.load_study(study_name=STUDY_NAME,storage = STORAGE_NAME)
cnx = sqlite3.connect('AuDrA_search.db')
studydf = study.trials_dataframe()
sorteddf = studydf.sort_values(by = ['value'], )
bestfit = study.best_params
bestval = study.best_value
print(bestval)
besttrial = study.best_trial
print(besttrial)
bt_iv = besttrial.intermediate_values
best_iv_epoch = min(bt_iv, key = bt_iv.get)

# Show best hparams
print("  Params: ")
for key, value in besttrial.params.items():
    print("    {}: {}".format(key, value))

# Visualize study
intermediate_val_fig = optuna.visualization.plot_intermediate_values(study)
intermediate_val_fig.show()

contour_fig = optuna.visualization.plot_contour(study, params=['learning_rate','architecture'])
contour_fig.show()

contour_fig_all = optuna.visualization.plot_contour(study)
contour_fig_all.show()

opthist_fig = optuna.visualization.plot_optimization_history(study)
opthist_fig.show()

hparam_importances = optuna.visualization.plot_param_importances(study)
hparam_importances.show()
