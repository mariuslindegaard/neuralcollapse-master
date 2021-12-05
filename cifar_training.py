import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

from our_models import NetSimpleConv, NetSimpleConv4

from tqdm import tqdm

import os

parser = argparse.ArgumentParser(description='Simple ConvNet training on MNIST to achieve neural collapse')
parser.add_argument('-cr', '--criterion', default='mse', type=str, help='loss function used, cross entropy vs MSE (default)')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=0.067, type=float, help='initial learning rate (default: 0.1)')
parser.add_argument('-lr-decay', '--learning-rate-decay', default=0.1, type=float, help='learning rate decay (default: 0.1)')
parser.add_argument('-mom', '--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--init-scale', default=1.0, type=float, help='initialization scale for network weights (default:1.0)')
parser.add_argument('--epochs', default=350, type=int, help='total epochs (default: 350)')
parser.add_argument('--lr-decay-steps', default=3, type=int, help='number of learning rate decay steps (default: 3)')
parser.add_argument('--no-bias', action='store_true')
parser.add_argument('--use-fc', action='store_true')

# dataset parameters
im_size             = 32
C                   = 10
input_ch            = 3


# analysis parameters
epoch_list          = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                       12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                       32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                       85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                       225, 245, 268, 293, 320, 350, 400, 450, 500, 550, 600,
                       650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100]

def train(model, criterion, optimizer, scheduler, trainloader, epochs, epoch_list, save_dir, one_hot=False, use_cuda=False):

    use_cuda = use_cuda and torch.cuda.is_available()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    ebar = tqdm(total=epochs, position=0, leave=True)
    for e in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        pbar = tqdm(total=len(trainloader), position=0, leave=True)
        for batch_idx, (inputs, labels) in enumerate(trainloader, start=1):

            # labels -= 1
            # if inputs.shape[0] != batch_size:
            #     continue

            if one_hot:
                labels = F.one_hot(labels, num_classes=C).float()
            if use_cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if one_hot:
                accuracy = torch.mean((torch.argmax(outputs,dim=1)==torch.argmax(labels, dim=1)).float()).item()
            else:
                accuracy = torch.mean((torch.argmax(outputs,dim=1)==labels).float()).item()

            running_loss += loss.item()
            running_accuracy += accuracy

            pbar.update(1)
            pbar.set_description(
                'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
                'Batch Loss: {:.6f} \t'
                'Batch Accuracy: {:.6f}'.format(
                    e+1,
                    batch_idx,
                    len(trainloader),
                    100. * batch_idx / len(trainloader),
                    loss.item(),
                    accuracy))
        pbar.close()
        lr_scheduler.step()
        ebar.update(1)
        ebar.set_description(
            'Train\t\tEpoch: {}/{} \t'
            'average Epoch Loss: {:.6f} \t'
            'average Epoch Accuracy: {:.6f}'.format(
                e+1,
                epochs,
                running_loss/len(trainloader),
                running_accuracy/len(trainloader)))
        if e+1 in epoch_list:
            torch.save(model.state_dict(), os.path.join(save_dir,'%d.pt'%(e+1)))
    ebar.close()

if __name__ == "__main__":

    global args
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    tx = transforms.Compose([transforms.ToTensor(), normalize])
    data = datasets.CIFAR10(root='cifar10', train=True, transform=tx, download=True)
    trainloader = DataLoader(data, batch_size=args.batch_size)

    # binary_classes = [1,2]
    # idxs = [i for i in range(len(data)) if data.targets[i] in binary_classes]
    # data_binary = Subset(data, idxs)
    # trainloader = DataLoader(data_binary, batch_size=batch_size)

    # model = models.vgg16_bn(pretrained=False,num_classes=C)
    # first_layer_ch = model.features[0].weight.shape[0]
    # model.features[0] = nn.Conv2d(input_ch, first_layer_ch, 3, 1, 1)

    if args.epochs > epoch_list[-1]:
        epoch_list.extend(list(np.arange(epoch_list[-1],args.epochs,8))[1:])
        epoch_list.append(args.epochs)

    model = NetSimpleConv(input_ch, 32, C, init_scale=args.init_scale, bias= not args.no_bias)

    save_dir = 'cifar_regular_expt_lr%.3f_wd%.4f'%(args.learning_rate, args.weight_decay)
    if args.no_bias:
        save_dir += '_no_bias'
    if args.use_fc:
        save_dir += '_2fc'

    save_dir = os.path.join(save_dir, 'mse' if args.criterion=='mse' else 'cross_entropy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.MSELoss() if args.criterion=='mse' else nn.CrossEntropyLoss()
    if args.weight_decay>0:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum)

    epochs_lr_decay = [i*args.epochs//args.lr_decay_steps for i in range(1,args.lr_decay_steps)]

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=epochs_lr_decay,
                                                  gamma=args.learning_rate_decay)

    train(model, criterion, optimizer, lr_scheduler, trainloader, args.epochs, epoch_list, save_dir, one_hot=args.criterion=='mse', use_cuda=True)
