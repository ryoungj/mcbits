#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Argument parser for the training training and compressing.

The arguments come from three parts with priority: command line > config file > default values.

Training and compressing share a subset of arguments (mostly regarding to dataset used), while maintaining their own
arguments, respectively. In compressing time, the training arguments are loaded from the saved training config file
and kept intact.
"""

import argparse
import sys
import os
import yaml
import torch
from os.path import join
from .coders import CODER_LIST

USABLE_TYPES = set([float, int])


def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


def produce_override_string(args, override_args):
    lines = []
    for v in override_args:
        v_arg = getattr(args, v)
        if type(v_arg) in USABLE_TYPES:
            lines.append(v + ": " + str(v_arg))
        else:
            lines.append(v + ": " + "{}".format(str(v_arg)))

    return "\n===== Overrided =====\n" + "\n".join(lines) + "\n"


class AlwaysExecuteAction(argparse.Action):
    pass


class DefaultValue:

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Default({})".format(self.value)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ArgumentParser(argparse.ArgumentParser):

    def parse_known_args(self, args=None, namespace=None):
        # default Namespace built from parser defaults
        if namespace is None:
            namespace = argparse.Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not argparse.SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not argparse.SUPPRESS:
                        if isinstance(action, AlwaysExecuteAction):
                            default = DefaultValue(action.default)
                            setattr(namespace, action.dest, default)

        namespace, args = super(ArgumentParser, self).parse_known_args(args, namespace)

        for action in self._actions:
            if isinstance(action, AlwaysExecuteAction):
                if action.dest is not argparse.SUPPRESS:
                    value = getattr(namespace, action.dest, None)
                    if isinstance(value, DefaultValue):
                        action(self, namespace, value.value, None)

        return namespace, args


def get_config(args, sysarg_override=True):
    # get commands from command line
    if sysarg_override:
        override_args = argv_to_vars(sys.argv)
    else:
        override_args = []

    # load yaml file
    print("=> Reading YAML config from {}".format(args.config))
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    args.__dict__.update(loaded_yaml)
    if len(override_args) > 0:
        print(produce_override_string(args, override_args))


def get_parser():
    """
    The general argument parser which is needed by both training and compressing.
    In other words, these arguments have `train_*` and `compress_*` versions, parsed respectively.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--config",
        help="training/compressing config file",
        default=None,
    )
    parser.add_argument(
        "--logdir",
        default='./checkpoints',
        type=str,
        help="base directory for logging and saving models",
    )
    parser.add_argument(
        "--datadir",
        help="dataset base directory",
        default="./data",
    )
    parser.add_argument(
        "--dataset",
        help="dataset for training/compressing",
        type=str,
        default="EMNIST",
    )
    parser.add_argument(
        "--split",
        help="split of dataset (for EMNIST)",
        type=str,
        default="mnist",
        choices=["mnist", "letters"],
    )
    parser.add_argument(
        "--binarize",
        type=str2bool,
        default=True,
        help="dynamically binarize the data if true, otherwise return bernoulli probabilities",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--workers",
        default=20,
        type=int,
        help="number of data loading workers (default: 20)",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for initializing training/compressing",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="which GPU to use for training, set to -1 for CPU",
    )
    parser.add_argument(
        "--num_particles",
        help="number of particles for training/compressing",
        type=int,
        default=1,
    )

    return parser


def get_train_parser():
    """The parser for training specific arguments."""

    parser = get_parser()

    parser.add_argument(
        "--model",
        default="BinaryVAE",
        help="model architecture",
    )
    parser.add_argument(
        "--latent_dim",
        default=50,
        type=int,
        help="latent dimensions",
    )
    parser.add_argument(
        "--bound",
        help="the variational bound for training VAEs, `ELBO` or `IWAE`",
        type=str,
        default="ELBO",
        choices=["ELBO", "IWAE"],
    )
    parser.add_argument(
        "--optimizer",
        help="optimizer to use",
        type=str,
        default="Adam",
    )
    parser.add_argument(
        "--scheduler",
        help="learning rate scheduler to use",
        type=str,
        default="constant",
        choices=["constant", "multi_step"],
    )
    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument(
        "--eps",
        default=1e-4,
        type=float,
        help="epsilon for Adam",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        help="number of total epochs to run",
    )

    return parser


def get_compress_parser():
    """The parser for compressing specific arguments."""

    parser = get_parser()

    parser.add_argument(
        "--train_config",
        help="the training config file to load",
        type=str,
        default='./configs/train_config.yml',
    )
    parser.add_argument(
        "--num_compress",
        help="maximum number of data samples to compress",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--coder",
        help="the coder to use for compression",
        type=str,
        default="BitsBackCoder",
        choices=CODER_LIST,
    )
    parser.add_argument(
        "--lprec",
        help="the lower bound precision of the unsigned integer in rANS message",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bprec",
        help="the precision of each unsigned integer in rANS stack",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--log_num_bucket",
        help="the log of the number of discretization buckets for the latents",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--prop_mprec",
        help="the discretization precision for the proposal/posterior distribution",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--cond_mprec",
        help="the discretization precision for the conditional distribution",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--prior_mprec",
        help="the discretization precision for the prior distribution",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch_compute",
        type=str2bool,
        default=False,
        help="whether batch the compute of conditional likelihood over particles",
    )
    parser.add_argument(
        "--decode_check",
        type=str2bool,
        default=False,
        help="whether run decoding and check the correctness of decoded results",
    )


    # Args for lossy compression
    parser.add_argument(
        "--improve_latent_params",
        type=str2bool,
        default=False,
        help="should we also iteratively improve the latent params for lossy compression",
    )

    # Args for SMC compression
    # TODO: make them compatible with previous ones
    parser.add_argument("--data", help="name of dataset", type=str,
                        default="musedata")
    parser.add_argument("--dataset_path", help="dataset path", type=str,
                        default="/scratch/ssd002/home/ryoungj/code/mcbits_orig/data/pianorolls/musedata.pkl")
    # parser.add_argument("--logdir", help="model checkpoint path", type=str,
    #                     default="")

    return parser


def get_train_args(train_parser, config=None):
    if config is None:  # training time
        train_args = train_parser.parse_args()
        # get arguments from `arg.config` (from command line) and then override other arguments from command line
        if len(sys.argv) > 1:
            get_config(train_args)

        train_args.expdir = join(train_args.logdir, train_args.exp_name)
        train_args.ckpt_path = join(train_args.expdir, 'model.pt')
        train_args.config_path = join(train_args.expdir, 'train_config.yml')

        if (train_args.gpu == -1) or (not torch.cuda.is_available()):
            train_args.gpu = None
    else:  # compressing time, loading from training config file
        train_args = train_parser.parse_args([])  # do not parse, just use values from the saved training config
        train_args.config = config
        get_config(train_args, sysarg_override=False)

    return train_args


def get_compress_args(compress_parser):
    compress_args = compress_parser.parse_args()

    if compress_args.config:
        get_config(compress_args)

    if (compress_args.gpu == -1) or (not torch.cuda.is_available()):
        compress_args.gpu = None

    return compress_args


def dump_args(args, path):
    with open(path, 'w') as fp:
        yaml.dump(vars(args), fp)
