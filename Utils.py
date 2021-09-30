import os
import numpy as np
import torch
import glob
import torch.utils.data as data
from torch.autograd import Function
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import Setting as st


def load_time(sbj):
    tr = np.load(st.preprocessed_data_path + "Preprocessed_s%02d_train.npy" % (sbj))
    vl = np.load(st.preprocessed_data_path + "Preprocessed_s%02d_valid.npy" % (sbj))
    ts = np.load(st.preprocessed_data_path + "Preprocessed_s%02d_test.npy" % (sbj))

    return tr, vl, ts

def read_file_name(query, pool_path):
    answer = glob.glob(pool_path + query)
    return answer

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def delete_dir(path):
    if os.path.isfile(path):
        os.remove(path)

    return

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


def estimate_JSD_MI(joint, marginal, mean=False):
    joint = (torch.log(torch.tensor(2.0)) - F.softplus(-joint))
    marginal = (F.softplus(-marginal)+marginal - torch.log(torch.tensor(2.0)))

    out = joint - marginal
    if mean:
        out = out.mean()
    return out

def evaluation(encoder1, encoder2, classifier, decomposer, loader, criterion):
    loss_all = 0
    pred_all = np.empty(shape=(0), dtype=np.float32)
    real_all = np.empty(shape=(0), dtype=np.float32)

    iter = 1
    for x, y in loader:
        localf = encoder1(x.cuda())

        rele, irre = decomposer(localf)

        globalf = encoder2(rele)
        logit = classifier(globalf)
        _, pred = torch.max(F.softmax(logit, -1).data, -1)

        loss = criterion(logit, y.cuda())
        loss_all = loss_all + loss.cpu()
        pred_all = np.concatenate((pred_all, pred.cpu()), axis=-1)

        real_all = np.concatenate((real_all, y), axis=-1)
        iter = iter + 1

        del x, y, loss, logit, pred

    acc = accuracy_score(real_all, pred_all)
    return loss_all, acc

def Gaussian_normalization(data, mean, std, num_ch , train=True):
    # data: [channel, freq or time , trials]

    if train == True:
        mean_ch = np.empty(shape=(num_ch))
        std_ch = np.empty(shape=(num_ch))

        for cha in range(num_ch):
            mean_ch[cha] = data[cha,:,:].mean()
            std_ch[cha] = data[cha,:,:].std()

        mean = mean_ch
        std = std_ch

    for ch in range(num_ch):
        data[ch,:,:] = (data[ch,:,:]-mean[ch])/(std[ch]+0.00000001)

    return data, mean, std
