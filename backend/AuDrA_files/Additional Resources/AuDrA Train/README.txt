This directory allows one to train AuDrA with the best-fitting model settings. The script outputs the data files that are used in the Analysis folder in the OSF repository for this project. Upon completing the training, the script also prints the AuDrA-human and Elaboration-human correlations in the console.

To run:
1. clone the training environment ('audra_train_environment.yml')
2. get the AuDrA Drawings from: https://osf.io/h4adm/, add each of the folders to the directory with AuDrA_train_bestfit.py (e.g., primary_images folder)
3. open 'AuDrA_train_bestfit.py'
4. activate the 'audra_train_environment' conda environment
5. run 'python AuDrA_train_bestfit.py' in the terminal
