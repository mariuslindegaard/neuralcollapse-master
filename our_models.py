import torch
import torch.nn as nn
import torch.nn.functional as F

class NetSimpleConv(nn.Module):
    def __init__(self, input_channels, hidden_size, numClass, init_scale=1, init_type='const_norm', affine=True, has_nonlinear=1, has_bn=1, bias=True):
        super(NetSimpleConv, self).__init__()

        self.hidden_size = hidden_size
        featIn = input_channels
        featOut = hidden_size
        self.conv1_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0, has_bn=0, affine=affine, bias=bias)

        featIn = featOut
        featOut = featOut*2
        self.conv2_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        featIn = featOut
        featOut = featOut*2
        self.conv3_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        featIn = featOut
        featOut = featOut*2
        self.conv4_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        featIn = featOut
        featOut = numClass
        self.conv5_sub = SubBRC(featIn,featOut, kerSize=2, stride=1, padding=0, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        self.conv5_sub.conv.lastLayer = True

        #featIn = featOut; featOut = numClass
        #self.fc1 = nn.Linear(featIn , featOut)
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim

        init_convnet(self,init_scale,init_type, bn_affine=affine)

    def forward(self, x, detached=None, detach_last=None):
        if  detach_last is None:
            detach_last = detached

        # import pdb; pdb.set_trace()

        x = self.conv1_sub(x)

        x =  self.conv2_sub(x)

        x = self.conv3_sub(x)

        x = self.conv4_sub(x)

        x = self.conv5_sub(x)
        #import pdb; pdb.set_trace()
        x = x.squeeze(3).squeeze(2)
        self.final_out = x # save the output for computing the loss

        return x

class NetSimpleConv4(nn.Module):
    def __init__(self, input_channels, hidden_size, numClass, init_scale=1, init_type='const_norm', affine=True, has_nonlinear=1, has_bn=1, bias=True):
        super(NetSimpleConv4, self).__init__()

        self.hidden_size = hidden_size
        featIn = input_channels; featOut = hidden_size
        self.conv1_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=0, has_bn=0, affine=affine, bias=bias)

        featIn = featOut
        featOut = featOut*2
        self.conv2_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        featIn = featOut
        featOut = featOut*2
        self.conv3_sub = SubBRC(featIn,featOut, kerSize=3, stride=2, padding=1, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        featIn = featOut
        featOut = numClass
        self.conv5_sub = SubBRC(featIn,featOut, kerSize=4, stride=1, padding=0, detached=False, has_nonlinear=has_nonlinear, has_bn=has_bn, affine=affine, bias=bias)

        self.conv5_sub.conv.lastLayer = True

        self.last_layer = self.conv5_sub.conv

        #featIn = featOut; featOut = numClass
        #self.fc1 = nn.Linear(featIn , featOut)
        self.numClass = numClass
        self.nnsoftmax_layer = nn.Softmax(1) # 2nd dim

        init_convnet(self,init_scale,init_type, bn_affine=False)

    def forward(self, x, detached=None, detach_last=None):
        if  detach_last is None:
            detach_last = detached

        # import pdb; pdb.set_trace()

        x = self.conv1_sub(x)

        x =  self.conv2_sub(x)

        x = self.conv3_sub(x)

        x = self.conv5_sub(x)
        #import pdb; pdb.set_trace()
        x = x.squeeze(3).squeeze(2)
        self.final_out = x # save the output for computing the loss

        return x

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
            self.bn   = nn.BatchNorm2d(featIn, affine=affine)
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

def init_convnet(model,init_scale,init_type='const_norm',bn_affine=True,bnpost=''):

    init_0_scale, init_1_scale =  parse_init_scale(init_scale)

    print('init_0_scale:' + str(init_0_scale))

    print('init_1_scale:' + str(init_1_scale))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #nn.init.normal_(m.weight, std=init_0_scale)

            if hasattr( m,'lastLayer') and m.lastLayer:
                init_scale__ = init_1_scale
            else:
                init_scale__ = init_0_scale

            # const norm
            if init_type == 'const_norm':
                nn.init.normal_(m.weight, std=0.05)
                m.weight.data = init_scale__ * m.weight.data / torch.norm(m.weight.data,2)
            else:
                nn.init.normal_(m.weight, std=init_scale__)


        elif isinstance(m, nn.BatchNorm2d):
            if bn_affine is False:
                m.weight = None
                m.bias   = None
                m.affine = False
            else:
                m.affine = True
                m.weight = nn.Parameter(m.running_var.clone().detach(), requires_grad=True)
                m.weight.data.fill_(1)
                m.bias   = nn.Parameter(m.running_var.clone().detach(), requires_grad=True)
                m.bias.data.zero_()

            if bnpost.startswith('div'):
                div_by = int(bnpost.lstrip('div'))
                m.affine = True
                m.weight = nn.Parameter(m.running_var.clone().detach(), requires_grad=False)
                m.weight.data.fill_(1/div_by)
                m.bias   = nn.Parameter(m.running_var.clone().detach(), requires_grad=False)
                m.bias.data.zero_()


        elif isinstance(m, nn.Linear):

            if hasattr( m,'lastLayer') and m.lastLayer:
                init_scale__ = init_1_scale
            else:
                init_scale__ = init_0_scale

            if init_type == 'const_norm':
                nn.init.normal_(m.weight, std=0.05)
                m.weight.data = init_scale__ * m.weight.data / torch.norm(m.weight.data,2)
            else:
                nn.init.normal_(m.weight, std=init_scale__)
            m.bias.data.zero_()

def parse_init_scale(init_scale):
    if isinstance(init_scale, str):
        # init_scale 4,1.5
        init_scale = init_scale.split(',')
        for i in range(0,len(init_scale)):
            init_scale[i] = float(init_scale[i])
        if len(init_scale) == 1:
            init_0_scale = init_scale[0]
            init_1_scale = init_0_scale
        else:
            init_0_scale = init_scale[0]
            init_1_scale = init_scale[1]

    else:
        init_0_scale = init_scale
        init_1_scale = init_scale

    return  (init_0_scale, init_1_scale)
