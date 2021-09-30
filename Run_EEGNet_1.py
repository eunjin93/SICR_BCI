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


exp_name = "GIST_EEGNet"

gpu = 0


os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all = np.arange(1, 53)
sources = np.setdiff1d(all, [29, 34])  # Remove bad subjects
num_ch = 64
num_time = 1024
fs = 512
nfeatg = 512
nfeatl, nfeatl2 = 16, 255

""" Build Network """
print("=" * 30, "> Build network")
local_enc = md.LocalEncoder_EEGNet(fs=fs, num_ch=num_ch, num_time=num_time).to(device)
global_enc = md.GlobalEncoder_EEGNet(num_ch=num_ch, num_time=num_time, nfeatl=nfeatl).to(device)
local_disc = md.Local_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, nfeatl2=nfeatl2, num_ch=num_ch).to(device)
global_disc = md.Global_disc_EEGNet(nfeatl=nfeatl, nfeatg=nfeatg, num_ch=num_ch).to(device)
decomposer = md.Decomposer(nfeatl).to(device)
mine = md.MINE(nfeatr=int(nfeatl * nfeatl2), nfeati=int(nfeatl * nfeatl2)).to(device)
classifier = md.Classifier(nfeatg).to(device)

""" Optimizer """
parameters = list(local_enc.parameters()) + list(global_enc.parameters()) + list(local_disc.parameters()) + list(
    global_disc.parameters()) + list(classifier.parameters()) + list(mine.parameters()) + list(decomposer.parameters())
opt = radam.RAdam(parameters, lr=st.lr, weight_decay=st.w_decay)
scheduler = lr_scheduler.ExponentialLR(opt, gamma=0.99)

""" Load data """
all_trdat = np.empty(shape=(num_ch, num_time, 0), dtype=np.float32)
all_vldat = np.empty(shape=(num_ch, num_time, 0), dtype=np.float32)
all_trlbl = np.empty(shape=(0), dtype=np.int32)
all_vllbl = np.empty(shape=(0), dtype=np.int32)

all_tsdat = dict()
all_tslbl = dict()
s = 0

for ia in sources:
    print("="*30, "Load Sub. %d's data" % ia)
    trdat, vldat, tsdat = ut.load_time(ia) #[channel, time, trial]

    trlbl = np.concatenate((np.zeros(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32), # Left
                            np.ones(shape=(int(trdat.shape[-1] / 2)), dtype=np.int32)), axis=-1) # Right
    vllbl = np.concatenate((np.zeros(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(vldat.shape[-1] / 2)), dtype=np.int32)), axis=-1)
    tslbl = np.concatenate((np.zeros(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32),
                            np.ones(shape=(int(tsdat.shape[-1] / 2)), dtype=np.int32)), axis=-1)

    all_trdat = np.concatenate((all_trdat, trdat), axis=-1)
    all_vldat = np.concatenate((all_vldat, vldat), axis=-1)

    all_tsdat["Sub{0}".format(ia)] = tsdat

    all_trlbl = np.concatenate((all_trlbl, trlbl), axis=-1)
    all_vllbl = np.concatenate((all_vllbl, vllbl), axis=-1)

    all_tslbl["Sub{0}".format(ia)] = torch.from_numpy(tslbl).long()

""" Gaussian normalization """
train, mean, std = ut.Gaussian_normalization(data=all_trdat, mean=0, std=1, train=True, num_ch=num_ch)
valid, _, _ = ut.Gaussian_normalization(data=all_vldat, mean=mean, std=std, train=False, num_ch=num_ch)

test = dict()
for ia in sources:
    temp, _, _ = ut.Gaussian_normalization(all_tsdat["Sub{0}".format(ia)], mean, std, train=False, num_ch=num_ch)
    temp = np.transpose(temp, [-1, 0, 1])
    test["Sub{0}".format(ia)] = torch.from_numpy(np.expand_dims(temp, axis=1)).float()

num_tr = train.shape[-1]
num_vl = valid.shape[-1]

""" Reshape data """
train = np.transpose(train, [-1, 0, 1])
train = np.expand_dims(train, axis=1)
valid = np.transpose(valid, [-1, 0, 1])
valid = np.expand_dims(valid, axis=1)

print("Data shape:", train.shape, valid.shape)

""" Convert numpy to torch tensor """
train = torch.from_numpy(train).float()
trlbl = torch.from_numpy(all_trlbl).long()
valid = torch.from_numpy(valid).float()
vllbl = torch.from_numpy(all_vllbl).long()

""" Dataset wrapping tensors """
tr_tensor = TensorDataset(train, trlbl)
vl_tensor = TensorDataset(valid, vllbl)

""" Dataloader """
ts_loader = dict()
for ia in sources:
    ts_loader["Sub{0}".format(ia)] = DataLoader(TensorDataset(test["Sub{0}".format(ia)], all_tslbl["Sub{0}".format(ia)]), batch_size=st.bs, shuffle=False, pin_memory=True, drop_last=False)

tr_loader = DataLoader(tr_tensor, batch_size=st.bs, shuffle=True, pin_memory=True, drop_last=False)
vl_loader = DataLoader(vl_tensor, batch_size=st.bs, shuffle=True, pin_memory=True, drop_last=False)

""" Training """
print("=" * 30, "> Train")
iter = 0
cls_criterion = nn.CrossEntropyLoss().cuda()
writer = SummaryWriter(st.tensorboard_path + exp_name)

# Train
for ep in range(st.total_epoch):
    global_enc.train(), local_enc.train(), local_disc.train(), global_disc.train(), mine.train(), decomposer.train(), classifier.train()

    for bidx, batch in enumerate(zip(tr_loader)):
        # Split data and label per batch
        batchx, batchy = batch[0] #[batch, 1, channel, time]

        # Reset gradient
        opt.zero_grad()

        # Feed input to enocders and then obtain local feature (relevant, irrelevant) and global feature
        localf = local_enc(batchx.cuda()) #[batch, d1, 1, t1]
        rele, irre = decomposer(localf) #[batch, d1, 1, t1], #[batch, depth, 1, t1]
        globalf = global_enc(rele) #[batch, d2]

        # Feed the relevant feature to classifier
        logits = classifier(globalf) #[batch, 2]
        loss_class = cls_criterion(logits, batchy.cuda())

        # To ensure good decomposition, estimate MI between relevant feature and irrelevant feature
        rele_ = torch.reshape(rele, (rele.shape[0], -1)) #[batch, d1*t1]
        irre_ = torch.reshape(irre, (irre.shape[0], -1)) #[batch, d1*t1]
        ishuffle = torch.index_select(irre_, 0, torch.randperm(irre_.shape[0]).to(device))
        djoint = mine(rele_, irre_) #[batch, 1]
        dmarginal = mine(rele_, ishuffle) #[batch, 1]
        loss_decomposition = - ut.estimate_JSD_MI(djoint, dmarginal, True)

        # Estimate global MI
        gshuffle = torch.index_select(globalf, 0, torch.randperm(globalf.shape[0]).to(device)) #[batch, d2]
        gjoint = global_disc(rele, globalf) #[batch, 1]
        gmarginal = global_disc(rele, gshuffle) #[batch, 1]
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
        writer.add_scalars("Train",
                           {"L_all": loss_all.item(), "L_class": loss_class.item(), "L_MINE": loss_decomposition.item(),
                            "L_Global_MI": loss_global_mi.item(),
                            "L_Local_MI": loss_local_mi.item(), "L_DIM": loss_dim.item()}, iter)
        iter = iter + 1

    scheduler.step()  # learning rate decay
    print("%s: %d epoch" % (exp_name, ep))

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
            loss_test, acc_test = ut.evaluation(local_enc, global_enc, classifier, decomposer,
                                                ts_loader["Sub{0}".format(sbj)], cls_criterion)
            writer.add_scalars("Test_Sub{0}".format(sbj), {"L_cls": loss_test.item(), "ACC": acc_test}, ep)
            tst_sum.append(acc_test)

        print("ACC_test_mean: %.4f" % (np.mean(np.array(tst_sum))))
