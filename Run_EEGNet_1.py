import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

import os
import csv

import Setting as st
import Utils as ut
import Module as md
import RADAM as radam


case = "%s_%s_Sce%d_fold%d" %(st.data, st.network, st.scenario, st.fold)

os.environ["CUDA_VISIBLE_DEVICES"] = str(st.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 30, "> %s (%s) Ours Sceanrio %d: fold %d" %(st.data, st.network, st.scenario, st.fold))

if st.data == "KU":
    sources = np.arange(1, 55)
    num_ch = 62
    num_time = int(3000 / 2)
    fs = int(1000 / 2)
    nfeatg = 736
    nfeatl, nfeatl2 = 16, 374

elif st.data == "GIST":
    all = np.arange(1, 53)
    sources = np.setdiff1d(all, [29,34]) # Remove bad subjects
    num_ch = 64
    num_time = 1024
    fs = 512
    nfeatg = 512
    nfeatl, nfeatl2 = 16, 255
    
else:
    raise NameError("Check the dataset.")

""" Build Network """
print("=" * 30, "> Build network")
local_enc = md.LocalEncoder_EEGNet(fs=fs, num_ch=num_ch, num_time=num_time).to(device)
global_enc = md.GlobalEncoder_EEGNet(num_ch=num_ch, num_time=num_time, nfeatl=nfeatl).to(device)
local_disc = md.Local_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, nfeatl2=nfeatl2, num_ch=num_ch).to(device)
global_disc = md.Global_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, num_ch=num_ch).to(device)
decomposer = md.Decomposer(nfeatl).to(device)
mine = md.MINE(nfeatr=int(nfeatl * nfeatl2), nfeati=int(nfeatl* nfeatl2)).to(device)
classifier = md.Classifier(nfeatg).to(device)

""" Optimizer """
parameters = list(local_enc.parameters()) + list(global_enc.parameters()) + list(local_disc.parameters()) + list(global_disc.parameters()) + list(classifier.parameters()) + list(mine.parameters()) + list(decomposer.parameters())
opt = radam.RAdam(parameters, lr=st.lr, weight_decay=st.w_decay)
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.99)

""" Load data """
print("=" * 30, "> Load data (%s TIME, sce %d, fold %d)" %(st.data, st.scenario, st.fold))

alltr = np.empty(shape=(num_ch, num_time, 0), dtype=np.float32)
allvl = np.empty(shape=(num_ch, num_time, 0), dtype=np.float32)
lbltr = np.empty(shape=(0), dtype=np.int32)
lblvl = np.empty(shape=(0), dtype=np.int32)

allts = dict()
lblts = dict()
s = 0

for ia in sources:
    if st.data == "KU":
        trdat, vldat, tsdat = ut.load_TIME_KU(ia, st.fold)
        # downsampling
        trdat = ut.downsampling(trdat, int(num_time), axis=1)
        vldat = ut.downsampling(vldat, int(num_time), axis=1)
        tsdat = ut.downsampling(tsdat, int(num_time), axis=1)

    elif st.data == "GIST":
        trdat, vldat, tsdat = ut.load_TIME_GIST(ia, st.fold)

    else:
        raise NameError("Check the name of datset.")

    trlbl = np.concatenate((np.zeros(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32)), axis=-1)
    vllbl = np.concatenate((np.zeros(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32)), axis=-1)
    tslbl = np.concatenate((np.zeros(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32)), axis=-1)

    alltr = np.concatenate((alltr, trdat), axis=-1)
    allvl = np.concatenate((allvl, vldat), axis=-1)

    allts["Sub{0}".format(ia)] = tsdat

    lbltr = np.concatenate((lbltr, trlbl), axis=-1)
    lblvl = np.concatenate((lblvl, vllbl), axis=-1)

    lblts["Sub{0}".format(ia)] = torch.from_numpy(tslbl).long()

""" Gaussian normalization """
train, mean, std = ut.Gaussian_normalization(data=alltr, mean=0, std=1, train=True, num_ch=num_ch)
valid, _, _ = ut.Gaussian_normalization(data=allvl, mean=mean, std=std, train=False, num_ch=num_ch)

test = dict()
for ia in sources:
    temp, _, _ = ut.Gaussian_normalization(allts["Sub{0}".format(ia)], mean, std, train=False, num_ch=num_ch)
    temp = np.transpose(temp, [-1,0,1])
    test["Sub{0}".format(ia)] = torch.from_numpy(np.expand_dims(temp, axis=1)).float()

num_tr = train.shape[-1]
num_vl = valid.shape[-1]

""" Reshape data """
train = np.transpose(train, [-1,0,1])
train = np.expand_dims(train, axis=1)
valid = np.transpose(valid, [-1,0,1])
valid = np.expand_dims(valid, axis=1)

print("Data shape:", train.shape, valid.shape)

""" Convert numpy to torch tensor """
train = torch.from_numpy(train).float()
trlbl = torch.from_numpy(lbltr).long()
valid = torch.from_numpy(valid).float()
vllbl = torch.from_numpy(lblvl).long()

""" Dataset wrapping tensors """
tr_tensor = TensorDataset(train, trlbl)
vl_tensor = TensorDataset(valid, vllbl)

""" Dataloader """
ts_loader = dict()
for ia in sources:
    ts_loader["Sub{0}".format(ia)] = DataLoader(TensorDataset(test["Sub{0}".format(ia)], lblts["Sub{0}".format(ia)]), batch_size=st.bs, shuffle=False, pin_memory=True, drop_last=False)
tr_loader = DataLoader(tr_tensor, batch_size=st.bs, shuffle=True, pin_memory=True, drop_last=False)
vl_loader = DataLoader(vl_tensor, batch_size=st.bs, shuffle=True, pin_memory=True, drop_last=False)

""" Training """
print("=" * 30, "> Train")
iter = 0
cls_criterion = nn.CrossEntropyLoss().cuda()
writer = SummaryWriter("./logs_Ours/TIME/%s" %(case))
# Train
for ep in range(st.total_epoch):
    global_enc.train(), local_enc.train(), local_disc.train(), global_disc.train(), mine.train(), decomposer.train(), classifier.train()

    for bidx, batch in enumerate(zip(tr_loader)):

        # Split data and label per batch
        batchx, batchy = batch[0]

        # Reset gradient
        opt.zero_grad()

        # Feed input to enocders and then obtain local feature (relevant, irrelevant) and global feature
        localf = local_enc(batchx.cuda())
        rele, irre = decomposer(localf)
        globalf = global_enc(rele)

        # Feed the relevant feature to classifier
        logits = classifier(globalf)
        loss_class = cls_criterion(logits, batchy.cuda())

        # To ensure good decomposition, estimate MI between relevant feature and irrelevant feature
        rele = torch.reshape(rele, (rele.shape[0], -1))
        irre = torch.reshape(irre, (irre.shape[0], -1))
        ishuffle = torch.index_select(irre, 0, torch.randperm(irre.shape[0]).to(device))
        djoint = mine(rele, irre)
        dmarginal = mine(rele, ishuffle)
        loss_decomposition = - ut.estimate_JSD_MI(djoint, dmarginal, True)

        # Estimate global MI
        gshuffle = torch.index_select(globalf, 0, torch.randperm(globalf.shape[0]).to(device))
        gjoint = global_disc(rele, globalf)
        gmarginal = global_disc(rele, gshuffle)
        loss_global_mi = ut.estimate_JSD_MI(gjoint, gmarginal, True)

        # Estimate local MI
        ljoint = local_disc(rele, globalf)
        lmarginal = local_disc(rele, gshuffle)
        temp = ut.estimate_JSD_MI(ljoint, lmarginal, False)
        loss_local_mi = temp.mean()

        loss_dim = - (loss_global_mi + loss_local_mi)
        
        # All objective function
        loss_all = st.alpha * loss_class + st.beta * loss_decomposition + st.gamma * loss_dim

        loss_all.backward()
        opt.step()
        opt.zero_grad()

        # Tensorboard
        writer.add_scalars("Train", {"L_all": loss_all.item(), "L_class": loss_class.item(), "L_MINE": loss_decomposition.item(), "L_Global_MI": loss_global_mi.item(),
                                     "L_Local_MI": loss_local_mi.item(), "L_DIM": loss_dim.item()}, iter)
        iter = iter + 1

    scheduler.step() # learning rate decay
    print("%s: %d epoch" %(case, ep))

    # Evaluation
    with torch.no_grad():
        global_enc.eval(), global_disc.eval(), local_enc.eval(), local_disc.eval(), mine.eval(), classifier.eval(), decomposer.eval()

        # Validation
        loss_val, acc_val = ut.evaluation(local_enc, global_enc, classifier, decomposer, vl_loader, cls_criterion)
        writer.add_scalars("Valid", {"L_cls": loss_val.item(), "Acc": acc_val}, ep)
        print("ACC_validation: %.4f" % (acc_val))

        # Test
        tst_sum = []
        for sbj in sources:
            loss_test, acc_test = ut.evaluation(local_enc, global_enc, classifier, decomposer, ts_loader["Sub{0}".format(sbj)], cls_criterion)
            writer.add_scalars("Test_Sub{0}".format(sbj), {"L_cls": loss_test.item(), "ACC": acc_test}, ep)
            tst_sum.append(acc_test)

        print("ACC_test_mean: %.4f" % (np.mean(np.array(tst_sum))))
