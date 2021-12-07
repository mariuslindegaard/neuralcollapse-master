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
            'cifar10': _DatasetWrapper.cifar10
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
        data = datasets.CIFAR10(root='cifar10', train=True, transform=tx, download=download)
        data.train_labels = th.Tensor(data.train_labels).to(torch.int64)

        data.train_labels = th.nn.functional.one_hot(
            data.train_labels, num_classes=len(th.unique(data.train_labels))
        ).float()

        self.loader = DataLoader(data, batch_size=data_cfg['batch-size'])
        self.input_shape = data.train_data.shape[1:]  # (32, 32, 3)
        self.num_classes = data.train_labels.shape[1]  # 10



def load_dataset(data_cfg: dict, *args, **kwargs):
    """Load the dataset"""

    wrapper = _DatasetWrapper(data_cfg, *args, **kwargs)
    (im_size, _, im_channels) = wrapper.input_shape

    return wrapper.loader, im_channels, im_size, wrapper.num_classes

def test():
    return load_dataset({'dataset-id': 'cifar10', 'batch-size': 128})


if __name__ == "__main__":
    test()