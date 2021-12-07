import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

import our_models
import data_loader

from tqdm import tqdm

import os
import shutil
import warnings

parser = argparse.ArgumentParser(description='Simple ConvNet training on MNIST to achieve neural collapse')
parser.add_argument('-cfg', '--config', default='config/default.yaml', type=str,
                    help='Config file path. YAML-format expected, see "./config/default.yaml" for format.')

def train(model, criterion, optimizer, scheduler, trainloader, epochs, epoch_list, save_dir, config_params, one_hot=False, use_cuda=False):
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

            if use_cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # One-hot labels is always assumed
            accuracy = torch.mean((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).float()).item()

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
        scheduler.step()
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


def get_optimizer(model, optimizer_cfg):
    criterion = nn.MSELoss() if optimizer_cfg['criterion'] == 'mse' else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=optimizer_cfg['lr'],
                          momentum=optimizer_cfg['momentum'],
                          weight_decay=optimizer_cfg['weight-decay'])

    epochs_lr_decay = [i * optimizer_cfg['epochs'] // optimizer_cfg['lr-decay-steps'] for i in
                       range(1, optimizer_cfg['lr-decay-steps'])]

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=epochs_lr_decay,
                                                  gamma=optimizer_cfg['lr-decay'])

    return criterion, optimizer, lr_scheduler

def init_config(config_path):
    """Return parsed config and create relevant directories"""
    with open(config_path, "r") as config_file:
        config_params = yaml.safe_load(config_file)
    model_cfg     = config_params['Model']  # noqa:E221
    data_cfg      = config_params['Data']  # noqa:E221
    optimizer_cfg = config_params['Optimizer']  # noqa:E221
    logging_cfg   = config_params['Logging']  # noqa:E221

    save_dir = logging_cfg['save-dir']
    save_dir_data = os.path.join(save_dir, 'data')
    if not os.path.exists(save_dir_data):
        os.makedirs(save_dir_data)

    shutil.copy(config_path, os.path.join(save_dir, "config.yaml"), follow_symlinks=True)

    return config_params, (model_cfg, data_cfg, optimizer_cfg, logging_cfg), (save_dir, save_dir_data)


def main():

    # Parse config file
    config_params,\
        (model_cfg, data_cfg, optimizer_cfg, logging_cfg),\
        (save_dir, save_dir_data) = init_config(args.config)

    # Get dataset from config
    trainloader, image_ch, image_size, num_classes = data_loader.load_dataset(data_cfg)

    # Warn if epochs might not be logged properly
    if optimizer_cfg['epochs'] != logging_cfg['epoch-list'][-1]:
        warnings.warn(f"Last logging epoch is {logging_cfg['epoch-list'][-1]} but the model "
                      f"will train for {optimizer_cfg['epochs']} epochs.")

    # Get model from config and dataset params
    model_ref = getattr(our_models, model_cfg['model-name'])
    model = model_ref(image_ch, image_size, num_classes,
                             init_scale=model_cfg['init-scale'], bias = not model_cfg['no-bias'])

    # Get optimizer from config
    criterion, optimizer, lr_scheduler = get_optimizer(model, optimizer_cfg)

    # Train model
    train(model, criterion, optimizer, lr_scheduler, trainloader, optimizer_cfg['epochs'], logging_cfg['epoch-list'],
          save_dir_data, config_params, one_hot=optimizer_cfg['criterion'] == 'mse', use_cuda=True)


if __name__ == "__main__":
    import yaml
    args = parser.parse_args()
    main()
