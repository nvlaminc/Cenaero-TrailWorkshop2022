# training_baseline.py

Script for the training and validation of the baseline.

- Generation of the validation and training dart series.
- Hyperparameters definition + model creation
- Model Training and metrics computation.

# training_cluster.py 

Same than training_baseline.py but with the loading of the corresponding model pretrained on the assigned centroids for the considered clustering method.

# id_kept.npy

For faster training, id_kept during the investigation to not repeat the training on every participants for comparison.

# dl_trainer.py

Script to generalize the training on every measurement.