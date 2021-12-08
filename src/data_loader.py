import torch as th
import torch.nn.functional
from torchvision import datasets, transforms  # , models
from torch.utils.data import DataLoader  # , Subset

from typing import Tuple


class _DatasetWrapper:
    loader: DataLoader
    input_shape: Tuple[int, int, int]
    num_classes: int


    def __init__(self, data_cfg: dict, *args, **kwargs):
        """Init the dataset with given id"""
        id_mapping = {
            'cifar10': _DatasetWrapper.cifar10,
            'mnist': _DatasetWrapper.mnist
        }

        if not data_cfg['dataset-id'].lower() in id_mapping.keys():
            raise NotImplementedError(f"Dataset with id '{id}' is not implemented. "
                                      f"Id must be one of \n{id_mapping.keys()}")

        # Prepeare datset
        id_mapping[data_cfg['dataset-id']](self, data_cfg, *args, **kwargs)


    def cifar10(self, data_cfg: dict, download=True):
        """Cifar10 dataset"""

        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        tx = transforms.Compose([transforms.ToTensor(), normalize])

        self.input_shape = (32, 32, 3)
        self.num_classes = 10

        def target_transform(target):
            # tmp = th.Tensor(target).to(torch.int64)
            # tmp = th.nn.functional.one_hot(
            #     tmp, num_classes=len(th.unique(tmp))
            # ).float()
            ret = th.zeros(self.num_classes)
            ret[target] = 1
            return ret.float()

        data = datasets.CIFAR10(root='cifar10', train=True, download=download,
                                transform=tx, target_transform=target_transform)

        self.loader = DataLoader(data, batch_size=data_cfg['batch-size'])
        # print(f"In-shape: {data.train_data.shape[1:]},\n"
        #       f"Out-shape: {data.train_labels.shape[1]}")
        # print(f"In-shape: {data.data.shape[1:]},\n"
        #       f"Out-shape: {data.target.shape[1]}")
        self.input_shape = (32, 32, 3)
        self.num_classes = 10

    def mnist(self, data_cfg: dict, download=True):
        """Mnist dataset"""

        im_size = 28
        padded_im_size = 32

        tx = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.1307], std=[0.3081])])

        def target_transform(target):
            ret = th.zeros(self.num_classes)
            ret[target] = 1
            return ret.float()

        data = datasets.MNIST(root='mnist', train=True, download=download,
                              transform=tx, target_transform=target_transform)

        self.loader = DataLoader(data, batch_size=data_cfg['batch-size'])
        # print(f"In-shape: {data.train_data.shape},\n"
        #       f"Out-shape: {data.train_labels.shape}")
        # print(f"In-shape: {data.data.shape},\n"
        #       f"Out-shape: {data.target.shape}")  # For newer torchvision verions
        self.input_shape = (32, 32, 1)
        self.num_classes = 10




def load_dataset(data_cfg: dict, *args, **kwargs):
    """Load the dataset"""

    wrapper = _DatasetWrapper(data_cfg, *args, **kwargs)
    (im_size, _, im_channels) = wrapper.input_shape

    return wrapper.loader, im_channels, im_size, wrapper.num_classes

def test():
    return load_dataset({'dataset-id': 'mnist', 'batch-size': 128})


if __name__ == "__main__":
    test()