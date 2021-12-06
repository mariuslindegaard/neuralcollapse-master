import numpy as np
import scipy as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

from our_models import NetSimpleConv,NetSimpleConv4, NetSimpleConv2FC

import matplotlib.pyplot as plt
import pickle
import argparse
import os
import collections

parser = argparse.ArgumentParser(description='Neural Collapse measurement script on CIFAR10')
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
                       650, 700, 750, 800, 850, 900, 950, 1000]

attrs = ['accuracy', 'loss', 'reg_loss', 'Sw_invSb', 'norm_M_CoV', 'norm_W_CoV', 'cos_M', 'cos_W',
         'W_M_dist', 'NCC_mismatch']

class Measurements(collections.UserDict):
    metrics = ('accuracy', 'loss', 'reg_loss', 'Sw_invSb', 'norm_M_CoV', 'norm_W_CoV', 'cos_M', 'cos_W',
             'W_M_dist', 'NCC_mismatch')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for metric in self.metrics:
            self[metric] = list()
        # self.accuracy     = []
        # self.loss         = []
        # self.reg_loss     = []

        # # NC1
        # self.Sw_invSb     = []

        # # NC2
        # self.norm_M_CoV   = []
        # self.norm_W_CoV   = []
        # self.cos_M        = []
        # self.cos_W        = []

        # # NC3
        # self.W_M_dist     = []

        # # NC4
        # self.NCC_mismatch = []

    def compute_metrics(self, model, criterion, dataloader, weight_decay, one_hot=False, use_cuda=True):
        model.eval()

        N = [0 for _ in range(C)]
        mean = [0 for _ in range(C)]
        Sw = 0

        loss = 0
        net_correct = 0
        NCC_match_net = 0

        class features:
            pass

        def feature_hook(self, input, output):
            features.value = input[0].clone()

        model.conv5_sub.conv.register_forward_hook(feature_hook)
        classifier = model.conv5_sub.conv

        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # labels -= 1  # Remove since the results are already 0-indexed
            # import pdb; pdb.set_trace()
            if one_hot:
                oh_labels = F.one_hot(labels, num_classes=C).float()
            if use_cuda:
                inputs = Variable(inputs.cuda())
                if one_hot:
                    oh_labels = Variable(oh_labels.cuda())
                else:
                    labels = Variable(labels.cuda())
            outputs = model(inputs)

            batchloss = criterion(outputs, oh_labels) if one_hot else criterion(outputs, labels)
            loss += batchloss.item()
            h = features.value.data.view(inputs.shape[0], -1)  # B CHW

            for c in range(C):
                idxs = (labels == c).nonzero(as_tuple=True)[0]
                h_c = h[idxs, :]  # B CHW

                # update class means
                mean[c] += torch.sum(h_c, dim=0)  #  CHW
                N[c] += h_c.shape[0]

        for c in range(C):
            mean[c] /= N[c]
        M = torch.stack(mean).T
        loss /= sum(N)

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # labels -= 1  # Remove since the results are already 0-indexed
            if one_hot:
                oh_labels = F.one_hot(labels, num_classes=C).float()
            if use_cuda:
                inputs = Variable(inputs.cuda())
                if one_hot:
                    oh_labels = Variable(oh_labels.cuda())
                else:
                    labels = Variable(labels.cuda())
            outputs = model(inputs)

            h = features.value.data.view(inputs.shape[0], -1)  # B CHW
            for c in range(C):
                # features belonging to class c
                idxs = (labels == c).nonzero(as_tuple=True)[0]
                h_c = h[idxs, :]  # B CHW

                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # B CHW
                cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                                   z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim=0)

                # during calculation of within-class covariance, calculate:
                # 1) network's accuracy
                net_pred = torch.argmax(outputs[idxs, :], dim=1).cpu()
                net_correct += sum(net_pred == labels.cpu()[idxs]).item()

                # 2) agreement between prediction and nearest class center
                NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim=1) \
                                          for i in range(h_c.shape[0])])
                NCC_pred = torch.argmin(NCC_scores, dim=1).cpu()
                NCC_match_net += sum(NCC_pred == net_pred).item()

        Sw /= sum(N)
        measurements['loss'].append(loss)
        measurements['accuracy'].append(net_correct / sum(N))
        measurements['NCC_mismatch'].append(1 - NCC_match_net / sum(N))

        reg_loss = loss
        for param in model.parameters():
            reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
        measurements['reg_loss'].append(reg_loss)

        # global mean
        muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1

        # between-class covariance
        M_ = M - muG
        Sb = torch.matmul(M_, M_.T) / C

        # tr{Sw Sb^-1}
        Sw = Sw.cpu().numpy()
        Sb = Sb.cpu().numpy()
        eigvec, eigval, _ = svds(Sb, k=C - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
        measurements['Sw_invSb'].append(np.trace(Sw @ inv_Sb))

        # avg norm
        W = classifier.weight.view(C, -1)
        M_norms = torch.norm(M_, dim=0)
        W_norms = torch.norm(W.T, dim=0)

        measurements['norm_M_CoV'].append((torch.std(M_norms) / torch.mean(M_norms)).item())
        measurements['norm_W_CoV'].append((torch.std(W_norms) / torch.mean(W_norms)).item())

        # ||W^T - M_||
        normalized_M = M_ / torch.norm(M_, 'fro')
        normalized_W = W.T / torch.norm(W.T, 'fro')
        measurements['W_M_dist'].append((torch.norm(normalized_W - normalized_M) ** 2).item())

        # mutual coherence
        def coherence(V):
            G = V.T @ V
            if use_cuda:
                G += torch.ones((C, C)).cuda() / (C - 1)
            else:
                G += torch.ones((C, C)) / (C - 1)
            G -= torch.diag(torch.diag(G))
            return torch.norm(G, 1).item() / (C * (C - 1))

        measurements['cos_M'].append(coherence(M_ / M_norms))
        measurements['cos_W'].append(coherence(W.T / W_norms))


def compute_metrics(measurements, model, criterion, dataloader, weight_decay, one_hot=False, use_cuda=True):
    model.eval()

    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    class features:
        pass

    def feature_hook(self, input, output):
        features.value = input[0].clone()

    model.conv6_sub.conv.register_forward_hook(feature_hook)  # TODO(mariu): Update both of these to reflect a layer choice, not just "conv6" or "conv5"
    classifier = model.conv6_sub.conv

    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # labels -= 1  # Remove since the results are already 0-indexed
        # import pdb; pdb.set_trace()
        if one_hot:
            oh_labels = F.one_hot(labels, num_classes=C).float()
        if use_cuda:
            inputs = Variable(inputs.cuda())
            if one_hot:
                oh_labels = Variable(oh_labels.cuda())
            else:
                labels = Variable(labels.cuda())
        outputs = model(inputs)

        batchloss = criterion(outputs, oh_labels) if one_hot else criterion(outputs, labels)
        loss += batchloss.item()
        h = features.value.data.view(inputs.shape[0],-1) # B CHW

        for c in range(C):
            idxs = (labels == c).nonzero(as_tuple=True)[0]
            h_c = h[idxs,:] # B CHW

            # update class means
            mean[c] += torch.sum(h_c, dim=0) # CHW
            N[c] += h_c.shape[0]

    for c in range(C):
        mean[c] /= N[c]
    M = torch.stack(mean).T
    loss /= sum(N)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # labels -= 1  # Remove since the results are already 0-indexed
        if one_hot:
            oh_labels = F.one_hot(labels, num_classes=C).float()
        if use_cuda:
            inputs = Variable(inputs.cuda())
            if one_hot:
                oh_labels = Variable(oh_labels.cuda())
            else:
                labels = Variable(labels.cuda())
        outputs = model(inputs)

        h = features.value.data.view(inputs.shape[0],-1) # B CHW
        for c in range(C):
            # features belonging to class c
            idxs = (labels == c).nonzero(as_tuple=True)[0]
            h_c = h[idxs,:] # B CHW

            # update within-class cov
            z = h_c - mean[c].unsqueeze(0) # B CHW
            cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                               z.unsqueeze(1))  # B 1 CHW
            Sw += torch.sum(cov, dim=0)

            # during calculation of within-class covariance, calculate:
            # 1) network's accuracy
            net_pred = torch.argmax(outputs[idxs,:], dim=1).cpu()
            net_correct += sum(net_pred==labels.cpu()[idxs]).item()

            # 2) agreement between prediction and nearest class center
            NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                      for i in range(h_c.shape[0])])
            NCC_pred = torch.argmin(NCC_scores, dim=1).cpu()
            NCC_match_net += sum(NCC_pred==net_pred).item()


    Sw /= sum(N)
    measurements.loss.append(loss)
    measurements.accuracy.append(net_correct/sum(N))
    measurements.NCC_mismatch.append(1-NCC_match_net/sum(N))

    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    measurements.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T
    measurements.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # avg norm
    W  = classifier.weight.view(C,-1)
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    measurements.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    measurements.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    measurements.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

    # mutual coherence
    def coherence(V):
        G = V.T @ V
        if use_cuda:
            G += torch.ones((C,C)).cuda() / (C-1)
        else:
            G += torch.ones((C,C)) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    measurements.cos_M.append(coherence(M_/M_norms))
    measurements.cos_W.append(coherence(W.T/W_norms))

if __name__ == "__main__":

    global args
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    tx = transforms.Compose([transforms.ToTensor(), normalize])
    data = datasets.CIFAR10(root='cifar10', train=True, transform=tx)
    dataloader = DataLoader(data, batch_size=args.batch_size)

    # binary_classes = [1,2]
    # idxs = [i for i in range(len(data)) if data.targets[i] in binary_classes]
    # data_binary = Subset(data, idxs)
    # dataloader = DataLoader(data_binary, batch_size=batch_size)

    if args.epochs > epoch_list[-1]:
        epoch_list.extend(list(np.arange(epoch_list[-1],args.epochs,8))[1:])
        epoch_list.append(args.epochs)

    ## Specify save directory
    save_dir = 'cifar_regular_expt_lr%.3f_wd%.4f'%(args.learning_rate, args.weight_decay)
    if args.no_bias:
        save_dir += '_no_bias'
    if args.use_fc:
        save_dir += '_2fc'
    save_dir = os.path.join(save_dir, 'mse' if args.criterion=='mse' else 'cross_entropy')

    measurements = Measurements()
    for e in epoch_list:  # TODO(marius): Make stop at epochs, not go through the whole list. Or rather: Do all files in the directory
        print('Loading %s : %d.pt'%(save_dir,e))
        model = NetSimpleConv2FC(input_ch, 32, C, init_scale=0.01, bias= not args.no_bias)  # TODO(marius): Make model dynamic from config
        model.load_state_dict(torch.load(os.path.join(save_dir,'%d.pt'%(e)), map_location=torch.device('cpu')))

        criterion = nn.MSELoss(reduction='sum') if args.criterion=='mse' else nn.CrossEntropyLoss(reduction='sum')
        measurements.compute_metrics(model, criterion, dataloader, args.weight_decay, one_hot= args.criterion=='mse', use_cuda=True)

    with open(os.path.join(save_dir,'%s.pkl'%(args.criterion,)), 'wb') as f:
        pickle.dump(measurements,f)

    if not os.path.exists(os.path.join(save_dir, 'plots')):
        os.makedirs(os.path.join(save_dir, 'plots'))

    # attrs = ['accuracy', 'loss', 'reg_loss', 'Sw_invSb', 'norm_M_CoV', 'norm_W_CoV', 'cos_M', 'cos_W', 'W_M_dist', 'NCC_mismatch']
    # for a in attrs:
    #     plt.plot(epoch_list, getattr(measurements, a), 'bx-')
    #     plt.title(a)
    #     plt.savefig(os.path.join(os.path.join(save_dir, 'plots'), '%s.pdf'%(a,)))
    #     plt.close()

    ## Make plots for each measurement
    for name, value in measurements:
        plt.plot(epoch_list, value, 'rx-')
        plt.title(name)
        plt.savefig(os.path.join(os.path.join(save_dir, 'plots'), f'{name}.pdf'))
        plt.close()
