import torch
import torch.nn as nn
import torch.nn.functional as F

import Setting as st
import Utils as ut

act_func = "elu"

class BasicConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='elu', norm_layer='bn', bias=False):
        super(BasicConv_Block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if norm_layer == 'bn':
            self.bn = nn.BatchNorm2d(out_planes)
        elif norm_layer == 'in':
            self.bn = nn.InstanceNorm2d(out_planes)
        elif norm_layer is None:
            self.bn = None
        else:
            assert False
        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'elu':
            self.act_func = nn.ELU()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

F1 = 8
D = 2
F2 = 16
drop_prob = 0.5

""" EEGNet """
class LocalEncoder_EEGNet(nn.Module):
    def __init__(self, fs, num_ch, num_time):
        super(LocalEncoder_EEGNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, int(fs / 2)), stride=1, bias=False,
                            padding=(0, (int(fs / 2) // 2) - 1))  # []
        self.b1 = nn.BatchNorm2d(F1)
        self.c2 = Conv2dWithConstraint(in_channels=F1, out_channels=F1 * D, kernel_size=(num_ch, 1), stride=1,
                                       bias=False, groups=F1, padding=(0, 0), max_norm=1)
        self.b2 = nn.BatchNorm2d(F1 * D)
        self.p2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.d2 = nn.Dropout(drop_prob)

    def forward(self, input):
        h1 = self.b1(self.c1(input))
        h2 = self.d2(self.p2(F.elu(self.b2(self.c2(h1)))))
        return h2


class GlobalEncoder_EEGNet(nn.Module):
    def __init__(self, num_ch, num_time, nfeatl):
        super(GlobalEncoder_EEGNet, self).__init__()
        self.c3 = nn.Conv2d(in_channels=nfeatl, out_channels=F1 * D, kernel_size=(1, 16), stride=1, bias=False,
                            groups=(nfeatl), padding=(0, 16 // 2))
        self.b3 = nn.BatchNorm2d(F1 * D)
        self.p3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.d3 = nn.Dropout(drop_prob)

    def forward(self, x):
        h3 = self.d3(self.p3(F.elu(self.b3(self.c3(x)))))
        h3_ = torch.flatten(h3, start_dim=1)
        return h3_


class Global_disc_EEGNet(nn.Module):
    def __init__(self, nfeatl, nfeatg, num_ch):
        super(Global_disc_EEGNet, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels=nfeatl, out_channels=F1 * D, kernel_size=(1, 16), stride=1, bias=False,
                      groups=(nfeatl), padding=(0, 16 // 2)),
            nn.BatchNorm2d(F1 * D),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(drop_prob)
        )
        self.dense1 = nn.Linear(int(nfeatg*2), 1)
        self.drop1 = nn.Dropout()

    def forward(self, localf, globalf):
        localff = self.local_conv(localf)
        localff = localff.view(localf.shape[0], -1)

        concat = torch.cat((localff, globalf), dim=-1)
        out = self.drop1(self.dense1(concat))

        return out

class Local_disc_EEGNet(nn.Module):
    def __init__(self, nfeatl, nfeatg, nfeatl2, num_ch):
        super(Local_disc_EEGNet, self).__init__()
        self.num_ch = num_ch
        self.nfeatl = nfeatl
        self.nfeatl2 = nfeatl2
        self.nfeatg = nfeatg

        self.drop1 = nn.Dropout()
        self.conv = nn.Conv2d(int(self.nfeatg+self.nfeatl), 1, kernel_size=1)

    def forward(self, localf, globalf):
        # Concat-and-convolve architecture
        globalff = globalf.unsqueeze(2).unsqueeze(3)
        globalff = globalff.repeat(1,1,1,self.nfeatl2)
        concat = torch.cat((localf, globalff), dim=1)
        out = self.drop1(self.conv(concat))
        out = out.view(out.shape[0],-1)

        return out

class Classifier(nn.Module):
    def __init__(self, nfeatr):
        super(Classifier, self).__init__()
        self.dense1 = nn.Linear(nfeatr, st.num_cl)

    def forward(self, latent):
        out = self.dense1(latent)

        return out


class MINE(nn.Module):
    def __init__(self, nfeatr, nfeati):
        super(MINE, self).__init__()
        self.fc1_x = nn.Linear(nfeatr, int(nfeatr/16))
        self.bn1_x = nn.BatchNorm1d(int(nfeatr/16))
        self.fc1_y = nn.Linear(nfeati, int(nfeati/16))
        self.bn1_y = nn.BatchNorm1d(int(nfeati/16))

        self.fc2 = nn.Linear(int(nfeati/16) + int(nfeatr/16),int(nfeati/16) + int(nfeatr/16))
        self.bn2 = nn.BatchNorm1d(int(nfeati/16) + int(nfeatr/16))

        self.fc3 = nn.Linear(int(nfeati/16) + int(nfeatr/16), 1)

    def forward(self, x, y, lambd=1):
        # GRL
        x = ut.GradReverse.grad_reverse(x, lambd)
        y = ut.GradReverse.grad_reverse(y, lambd)

        x = F.dropout(self.bn1_x(self.fc1_x(x)))
        y = F.dropout(self.bn1_y(self.fc1_y(y)))

        h = F.elu(torch.cat((x,y), dim=-1))
        h = F.elu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h


class Decomposer(nn.Module):
    def __init__(self, nfeat):
        super(Decomposer, self).__init__()
        self.nfeat = nfeat
        self.embed_layer = nn.Sequential(nn.Conv2d(nfeat, nfeat*2, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(nfeat*2), nn.ELU(), nn.Dropout())

    def forward(self, x):
        embedded = self.embed_layer(x)
        rele, irre = torch.split(embedded, [int(self.nfeat), int(self.nfeat)], dim=1)

        return rele, irre