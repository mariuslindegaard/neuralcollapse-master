import numpy as np
from scipy.sparse.linalg import svds
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pickle
import argparse
import os
import collections
from tqdm import tqdm
import warnings

import utils
import data_loader
import our_models

parser = argparse.ArgumentParser(description='Neural Collapse measurement script on CIFAR10')
parser.add_argument('-cfg', '--config', type=str, default="config/default.yaml",
                    help='Config file path. YAML-format expected, see "./config/default.yaml" for format.')


class Measurements(collections.UserDict):
    metrics = ('accuracy', 'loss', 'reg_loss', 'Sw_invSb', 'norm_M_CoV', 'norm_W_CoV', 'cos_M', 'cos_W',
             'W_M_dist', 'NCC_mismatch') # , 'SQI_eps', 'SQI_eps_var', 'SQI_symm_dist', 'SQI_e1', 'SQI_e2')

    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        for metric in self.metrics:
            self[metric] = list()
        self.config_params = None
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

    def _compute_eps_metrics(self, batch_idx: int, outputs: torch.Tensor, labels: torch.Tensor, num_classes: int):
        """Compute metrics related to epsilon in labels

        Note that both outputs and labels are one-hot encoded

        See 'assumption 1' in [Neural Collapse in Deep Homogenous Classifiers and the Role of Weight Decay,
        Rangamani and Banburski-Fahey, 2021 (preprint)]
        By their definitions, epsilon=eps1 and (epsilon/(C-1))=eps2
        """
        t = ('SQI_eps', 'SQI_eps_var', 'SQI_symm_dist', 'SQI_eps1', 'SQI_eps2')

        print(outputs.shape, labels.shape)




    def compute_metrics(self, model, criterion, dataloader, weight_decay, num_classes, config_params, use_cuda=True):
        self.config_params = config_params

        model.eval()

        N = [0 for _ in range(num_classes)]
        mean = [0 for _ in range(num_classes)]
        Sw = torch.tensor([0])

        loss = 0
        net_correct = 0
        NCC_match_net = 0

        class features:
            pass

        def feature_hook(self, input, output):
            features.value = input[0].clone()

        model.nc_measurements_layer.register_forward_hook(feature_hook)
        classifier = model.nc_measurements_layer

        use_cuda = use_cuda and torch.cuda.is_available()
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        print("First loop:")
        for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # labels -= 1  # Remove since the results are already 0-indexed
            # import pdb; pdb.set_trace()
            if use_cuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            outputs = model(inputs)

            batchloss = criterion(outputs, labels)
            loss += batchloss.item()
            h = features.value.data.view(inputs.shape[0], -1)  # B CHW

            for c in range(num_classes):
                idxs = (torch.argmax(labels, dim=1) == c).nonzero(as_tuple=True)[0]
                if not len(idxs):  # Continue if no images classified to 'c'
                    continue
                h_c = h[idxs, :]  # B CHW

                # update class means
                mean[c] += torch.sum(h_c, dim=0)  #  CHW
                N[c] += h_c.shape[0]

        for c in range(num_classes):
            mean[c] /= N[c]
        M = torch.stack(mean).T
        loss /= sum(N)

        print("Second loop:")
        for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            if use_cuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            outputs = model(inputs)

            h = features.value.data.view(inputs.shape[0], -1)  # B CHW
            for c in range(num_classes):
                # features belonging to class c
                idxs = (torch.argmax(labels, dim=1) == c).nonzero(as_tuple=True)[0]
                if not len(idxs):  # Continue if no images classified to 'c'
                    continue
                h_c = h[idxs, :]  # B CHW

                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # B CHW
                cov = torch.matmul(z.unsqueeze(-1),  # B CHW 1
                                   z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim=0)

                # during calculation of within-class covariance, calculate:
                # 1) network's accuracy
                net_pred = torch.argmax(outputs[idxs, :], dim=1).cpu()
                true_class = torch.argmax(labels[idxs, :], dim=1).cpu()
                net_correct += sum(net_pred == true_class).item()

                # 2) agreement between prediction and nearest class center
                NCC_scores = torch.stack([torch.norm(h_c[i, :] - M.T, dim=1) \
                                          for i in range(h_c.shape[0])])
                NCC_pred = torch.argmin(NCC_scores, dim=1).cpu()
                NCC_match_net += sum(NCC_pred == net_pred).item()

        Sw /= sum(N)
        self['loss'].append(loss)
        self['accuracy'].append(net_correct / sum(N))
        self['NCC_mismatch'].append(1 - NCC_match_net / sum(N))

        reg_loss = loss
        for param in model.parameters():
            reg_loss += 0.5 * weight_decay * torch.sum(param ** 2).item()
        self['reg_loss'].append(reg_loss)

        # global mean
        muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1

        # between-class covariance
        M_ = M - muG
        Sb = torch.matmul(M_, M_.T) / num_classes

        # tr{Sw Sb^-1}
        Sw = Sw.cpu().numpy()
        Sb = Sb.cpu().numpy()
        eigvec, eigval, _ = svds(Sb, k=num_classes - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
        self['Sw_invSb'].append(np.trace(Sw @ inv_Sb))  # Gets divide by 0 for the first epochs, it is fine...

        # avg norm
        W = classifier.weight.view(num_classes, -1)
        M_norms = torch.norm(M_, dim=0)
        W_norms = torch.norm(W.T, dim=0)

        self['norm_M_CoV'].append((torch.std(M_norms) / torch.mean(M_norms)).item())
        self['norm_W_CoV'].append((torch.std(W_norms) / torch.mean(W_norms)).item())

        # ||W^T - M_||
        normalized_M = M_ / torch.norm(M_, 'fro')
        normalized_W = W.T / torch.norm(W.T, 'fro')
        self['W_M_dist'].append((torch.norm(normalized_W - normalized_M) ** 2).item())

        # mutual coherence
        def coherence(V, C):
            G = V.T @ V
            if use_cuda:
                G += torch.ones((C, C)).cuda() / (C - 1)
            else:
                G += torch.ones((C, C)) / (C - 1)
            G -= torch.diag(torch.diag(G))
            return torch.norm(G, 1).item() / (C * (C - 1))

        self['cos_M'].append(coherence(M_ / M_norms, num_classes))
        self['cos_W'].append(coherence(W.T / W_norms, num_classes))

def main(args):

    # Parse config file
    config_params, \
        (model_cfg, data_cfg, optimizer_cfg, logging_cfg, measurements_cfg), \
        (save_dir, save_dir_data, save_dir_measurements) = utils.init_config(args.config)

    # Get dataset from config
    trainloader, image_ch, image_size, num_classes = data_loader.load_dataset(data_cfg)

    # Get model from config and dataset params
    model_ref = getattr(our_models, model_cfg['model-name'])
    model = model_ref(image_ch, image_size, num_classes,
                      init_scale=model_cfg['init-scale'], bias=not model_cfg['no-bias'])

    # TODO(marius): Why does model have init_scale=0.01? Does it even matter, seeing as we overload the weights anyway?
    # model = NetSimpleConv(input_ch, 32, C, init_scale=0.01,
    #                       bias=not args.no_bias)
    # Get optimizer from config
    criterion, optimizer, lr_scheduler = utils.get_optimizer(model, optimizer_cfg)

    # Train model
    # train(model, criterion, optimizer, lr_scheduler, trainloader, optimizer_cfg['epochs'], logging_cfg['epoch-list'],
    #       save_dir_data, config_params, one_hot=optimizer_cfg['criterion'] == 'mse', use_cuda=True)
    if logging_cfg['epoch-list'][-1] != optimizer_cfg['epochs']:
        warnings.warn(f"Epoch-list does not end at number of epochs, {logging_cfg['epoch-list'][-1]} is not {optimizer_cfg['epochs']}")
    measurements = Measurements()
    for e in logging_cfg['epoch-list']:  # TODO(marius): Make dependent on number of epochs, or throw an error when epochs<epoch_list[-1]
        print('Loading %s : %d.pt'%(save_dir_data,e))
        model.load_state_dict(torch.load(os.path.join(save_dir_data, f'{e}.pt'), map_location=torch.device('cpu')))

        measurements.compute_metrics(model, criterion, trainloader, optimizer_cfg['weight-decay'], num_classes,
                                     config_params, use_cuda=True)

    with open(os.path.join(save_dir_measurements, 'measurements.pkl'), 'wb') as f:
        pickle.dump(measurements, f)

    if not os.path.exists(os.path.join(save_dir_measurements, 'plots')):
        os.makedirs(os.path.join(save_dir_measurements, 'plots'))

    ## Make plots for each measurement
    for name, value in measurements.items():
        plt.plot(logging_cfg['epoch-list'], value, 'rx-')
        plt.title(name)
        plt.grid()
        plt.savefig(os.path.join(save_dir_measurements, 'plots', f'{name}.pdf'))
        plt.close()


def test():
    print("----"*20, "\nTEST OF do_measurements.py\n", "----"*20)
    args.config = "../config/cifar_short_2fc.yaml"
    main(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    # test()

