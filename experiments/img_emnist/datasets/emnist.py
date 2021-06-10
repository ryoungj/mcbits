#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" EMNIST dataloader. Contain `mnist` and `letters` splits """
import os
import torch
from torchvision import datasets, transforms
from torch.distributions import Bernoulli


class EMNIST:
    def __init__(self, args):
        super(EMNIST, self).__init__()

        data_root = os.path.join(args.datadir, "emnist")
        use_cuda = args.gpu is not None and torch.cuda.is_available()

        # Data loading code
        kwargs = {
            "num_workers": args.workers, "pin_memory": True
        } if use_cuda else {}

        if args.binarize == True:  # emnist to bits (dynamic binarization)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 lambda x: x.transpose(1, 2),
                 lambda x: Bernoulli(x).sample()]
            )
        else:  # emnist to integer
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 lambda x: x.transpose(1, 2),
                 lambda x: x * 255]
            )

        self.train_dataset = datasets.EMNIST(
            data_root,
            split=args.split,
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.EMNIST(
            data_root,
            split=args.split,
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
