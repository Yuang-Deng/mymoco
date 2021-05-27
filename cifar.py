import logging
import math
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import datasets
from torchvision import transforms

import moco.loader

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root, num_labeled, valid_num):
    simpleTransform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    cidar10transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean,
                                std=cifar10_std)
        ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs, valid_idxs = my_x_u_split(
        args, base_dataset.targets, num_labeled=num_labeled, valid_num=valid_num)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=simpleTransform)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=moco.loader.ThreeCropsTransform(cidar10transform, simpleTransform)
    )

    valid_dataset = CIFAR10SSL(
        root, valid_idxs, train=True, transform=simpleTransform)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=simpleTransform, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def x_u_split(args, labels, num_labeled=4000, num_classes=10, eval_step=1024):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled

    np.random.shuffle(labeled_idx)
    unlabeled_idx = np.delete(unlabeled_idx, labeled_idx)
    return labeled_idx, unlabeled_idx

def my_x_u_split(args, labels, num_labeled=4000, valid_num = 5000, num_classes=10, eval_step=1024):
    label_per_class = num_labeled // num_classes
    labels = np.array(labels)
    temp_idx = np.zeros(len(labels))
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == num_labeled
    temp_idx[labeled_idx] = 1

    p_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i and unlabeled_idx == labeled_idx)[0]
        idx = np.random.choice(idx, 10, False)
        p_idx.extend((idx))

    valid_idx = []
    label_per_class = valid_num // num_classes
    for i in range(num_classes):
        idx = np.where(temp_idx[labels == i] == 0)[0]
        idx = np.random.choice(idx, label_per_class, False)
        valid_idx.extend(idx)
    valid_idx = np.array(valid_idx)

    np.random.shuffle(labeled_idx)
    unlabeled_idx = np.delete(unlabeled_idx, valid_idx)
    return labeled_idx, unlabeled_idx, valid_idx

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

