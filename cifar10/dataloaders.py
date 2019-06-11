import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import os

import torchvision.transforms.functional as F


normalizing_mean=[0.485, 0.456, 0.406]
normalizing_std=[0.229, 0.224, 0.225]

class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def _prefetch(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self._prefetch()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self._prefetch()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break

class RandomDataset(data.Dataset):
    def __init__(self, num_random=10000, shape=(3,224,224)):
        self.size = num_random
        self.shape = shape
    def __len__(self):
        return self.size
    def __repr__(self):
        return self.__class__.__name__
    def __getitem__(self, index):
        img = torch.rand(*self.shape)
        target = 0 # Dummy target value
        return F.normalize(img, normalizing_mean, normalizing_std), target


class CIFAR10DataLoaders():
    normalize = transforms.Normalize(mean=normalizing_mean,
                                     std=normalizing_std)

    @staticmethod
    def train_loader(workers = 4, batch_size = 32, root="./data"):
        cifar_10_train = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            CIFAR10DataLoaders.normalize,
        ]), download=True)

        train_loader = torch.utils.data.DataLoader(
            cifar_10_train,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        return DataPrefetcher(train_loader)

    @staticmethod
    def train_loader_with_random(workers= 4, batch_size= 32, num_random= 30000, root= "./data"):
        cifar_10_train = datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            CIFAR10DataLoaders.normalize,
        ]), download=True)
        random_train = RandomDataset(num_random=num_random, shape=(3,32,32))
        dataset = data.ConcatDataset([cifar_10_train, random_train])
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        return train_loader

    @staticmethod
    def val_loader(workers= 4, batch_size= 256, root = "./data"):
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                CIFAR10DataLoaders.normalize,
            ])),
            batch_size= batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
        return DataPrefetcher(val_loader)


class CIFAR100DataLoaders():
    normalize = transforms.Normalize(mean=normalizing_mean,
                                     std=normalizing_std)

    @staticmethod
    def train_loader(workers = 4, batch_size = 32, root="./data"):
        cifar_100_train = datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            CIFAR100DataLoaders.normalize,
        ]), download=True)

        train_loader = torch.utils.data.DataLoader(
            cifar_100_train,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        return DataPrefetcher(train_loader)

    @staticmethod
    def train_loader_with_random(workers= 4, batch_size= 32, num_random= 30000, root= "./data"):
        cifar_100_train = datasets.CIFAR100(root=root, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            CIFAR100DataLoaders.normalize,
        ]), download=True)
        random_train = RandomDataset(num_random=num_random, shape=(3,32,32))
        dataset = data.ConcatDataset([cifar_100_train, random_train])
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        return train_loader

    @staticmethod
    def val_loader(workers= 4, batch_size= 256, root = "./data"):
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                CIFAR100DataLoaders.normalize,
            ])),
            batch_size= batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
        return DataPrefetcher(val_loader)
