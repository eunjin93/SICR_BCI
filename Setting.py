import Utils as ut


""" Path """
path = "/DataCommon/ejjeon/"
project_path = path + "Projects_BCI/SCIR/"


KU_time_path = path + "data/KU_Motor_Imagery/TIME/5-fold/"
GIST_time_path = path + "data/GIST_Motor_Imagery/TIME/5-fold/"

model_path = ut.create_dir(project_path + "Save_model/")
result_path = ut.create_dir(project_path + "Results/")
selected_path = ut.create_dir(project_path + "Selected/")


""" Hyperparamters """

bs = 40
lr = 1e-3
num_cl = 2
drop = 0.5
total_epoch = 100
w_decay = 0.1
scenario = 1
gpu = 0
fold = 1
data = "GIST"
network = "EEGNet"
alpha = 0.5
beta = 0.3
gamma = 0.5