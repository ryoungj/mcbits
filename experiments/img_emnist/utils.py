import torch
import torch.utils.data
from torch import optim
import numpy as np
import random

import datasets
import models


def set_seed(seed, deterministic=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn = False


def get_model(args):
    print("=> Creating model '{}'".format(args.model))
    model = models.__dict__[args.model](latent_dim=args.latent_dim)
    return model


def init_model(args, model, data):
    print("=> Initializing model '{}'".format(args.model))
    if args.model == "BinaryVAE":
        final_layer = model.dec_fc3
    elif args.model == "LossyBinaryVAE":
        final_layer = model.dec_fc3
    else:
        # Only initialize BinaryVAE model now
        return model

    # initialize the bias of the final layer with the dataset mean
    train_data = data.train_dataset.data
    train_set_mean = (train_data.float() / 255.).mean(dim=0).view(-1)
    train_bias = -torch.log(1. / torch.clamp(train_set_mean, 0.001, 0.999) - 1.)
    final_layer.bias.data.copy_(train_bias)
    return model


def set_gpu(args, model):
    if args.gpu is not None:
        print("=> Use GPU: {}".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("=> Use CPU")
    return model


def get_optimizer(args, model):
    if hasattr(args, "eps"):
        assert args.optimizer == "Adam"
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    else:
        optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(args, optimizer):
    if args.scheduler == "constant":
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 1.0)  # constant lr
    elif args.scheduler == "multi_step":
        milestones = [1]
        for i in range(0, 7):
            milestones.append(milestones[-1] * 3 + 1)
        lrs = [round(10. ** (-i / 7.), 2) for i in range(0, 8)]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lrs[
            min(np.searchsorted(milestones, epoch, 'right'), 7)])
    else:
        print(args.scheduler, type(args.scheduler))
        raise NotImplementedError

    return scheduler


def get_dataset(args):
    print_str = f"=> Getting '{args.dataset}' dataset"
    if args.dataset == "EMNIST":
        print_str += f" ('{args.split}' split)"
    print(print_str)
    dataset = getattr(datasets, args.dataset)(args)

    return dataset


def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().cpu().numpy()


def ndarray_to_tensor(arr, device='cpu'):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a, device=device) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr.to(device)
    else:
        return torch.from_numpy(arr).to(device)


def torch_fun_to_numpy_fun(fun, device='cpu'):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args, device=device)
        with torch.no_grad():
            return tensor_to_ndarray(fun(*torch_args, **kwargs))

    return numpy_fun
