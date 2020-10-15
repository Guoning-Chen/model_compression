import torch
import torchvision
import torchvision.transforms as transforms

import config


class bcolors:
    END  = '\033[0m'  # white (normal)
    R  = '\033[31m' # red
    G  = '\033[32m' # green
    O  = '\033[33m' # orange
    B  = '\033[34m' # blue
    P  = '\033[35m' # purple
    BOLD = '\033[1m'


def load_cifar(data_loc=config.DATA, batch_size=128, n_holes=1, length=16):
    """加载 cifar10数据集
    Args:
        data_loc (str): cifar10数据集的位置。
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_set = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)

    test_set = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader