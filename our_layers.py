import torch
import torch.nn as nn
import torch.nn.functional as F


class BackhookPreConv(nn.Module):
    def __init__(self):
        super(BackhookPreConv, self).__init__()

    def forward(self, x):
        return x.clone()


class BackhookPreReLU(nn.Module):
    def __init__(self):
        super(BackhookPreReLU, self).__init__()

    def forward(self, x):
        return x.clone()


class SubBRC(nn.Module):
    def __init__(self,featIn,featOut, kerSize=3, stride=1, padding=0, detached=False, has_nonlinear=True, has_bn=True,  affine=False, bias=True):
        super(SubBRC, self).__init__()
        if has_bn:
            self.bn = nn.BatchNorm2d(featIn, affine=affine)
        self.conv = nn.Conv2d(featIn, featOut, kerSize, stride=stride, padding=padding, bias=bias)
        self.detached = detached
        self.has_bn =  has_bn
        self.has_nonlinear = has_nonlinear
        self.backhook_conv = BackhookPreConv()
        self.backhook_relu = BackhookPreReLU()

    def forward(self, x):
        if self.detached == True:
            x = torch.autograd.Variable(x.data,requires_grad=False)
        if self.has_bn:
            x = self.bn(x)
        if self.has_nonlinear:
            x = F.relu( self.backhook_relu(x) )
        x = self.conv(  self.backhook_conv(x) )
        return x
