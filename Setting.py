import os
import Utils as ut


""" Path """
project_path = "./"
model_path = ut.create_dir(project_path + "Model/")
result_path = ut.create_dir(project_path+"Result/")
tensorboard_path = ut.create_dir(project_path+"Tensorboard/")

raw_data_path = "/home/ejjeon/DATA/data/[GIST] Raw_MI_data/"
preprocessed_data_path = "/home/ejjeon/DATA/data/[GIST] preprocessed data/npy/"

""" Hyperparameter """
lr = 1e-3
n_class = 2
drop = 0.5
total_epoch = 100
w_decay = 0.05
alpha = 0.5
beta = 0.3
gamma = 0.5
bs = 40
