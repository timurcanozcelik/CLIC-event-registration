This folder contains the scripts necessary to run the computational experiment (hyperparameter search) in the AuDrA paper. NOTE: run this at your own risk; this script requires two modern NVIDIA GPUs to run and can take well over a month to sample 505 hyperparameter settings, depending on your hardware setup. Also note that your results may differ slightly from our own due to stochasticity in the selection script.

To run the experiment:
1. download the computational experiment folder
2. clone the 'audra_experiment_environment.yml' conda environment
3. add the 'primary_images' folder (from AuDrA Drawings folder at https://osf.io/h4adm/) to the Computational Experiment folder
4. activate the 'audra_experiment_environment.yml' conda environment
5. run 'python AuDrA_Pl_1_2_1_hparamversion.py' in your terminal


If to access the hyperparameter settings explored in our experiment, the SQLite database that stored the results is available in the Experiment Results directory, along with a script that facilitates viewing the database and visualizing results ('AuDrA_hparam_study_viz.py')
