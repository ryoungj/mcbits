#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The original MNIST dataloader """
import os
import torch
from torchvision import datasets, transforms
from torch.distributions import Bernoulli


class MNIST:
    def __init__(self, args):
        super(MNIST, self).__init__()

        data_root = os.path.join(args.dataset, "mnist")
        use_cuda = args.gpu is not None and torch.cuda.is_available()

        # Data loading code
        kwargs = {
            "num_workers": args.workers, "pin_memory": True
        } if use_cuda else {}

        if args.binarize == True:  # mnist to bits (dynamic binarization)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 lambda x: Bernoulli(x).sample()]
            )
        else:  # mnist to integer
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 lambda x: x * 255]
            )

        self.train_dataset = datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.MNIST(
            data_root,
            train=False,
            transform=transform,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
