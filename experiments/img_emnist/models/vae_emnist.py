#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
THE VAE models for (E)MNIST datasets.

This code is adapted from https://github.com/bits-back/bits-back.
"""
import sys
import numpy as np

import torch
import torch.utils.data
from torch import nn, lgamma
from torch.nn import functional as F
from torch.distributions import Normal, Beta, Binomial, Bernoulli
from torchvision.utils import save_image

from . import obs_model


class VAE(nn.Module):
    """ VAE model wrapper. """

    def __init__(self):
        super(VAE, self).__init__()

    def loss(self, x, bound="ELBO", num_particles=1, latent_params=None):
        assert bound in ["ELBO", "IWAE"]

        elbo, iwae, _ = self.compute_bounds(x, num_particles=num_particles, latent_params=latent_params)
        if bound == "ELBO":
            loss = - torch.mean(elbo)
        else:
            loss = - torch.mean(iwae)

        return loss * np.log2(np.e) / 784.

    def compute_bounds(self, x, num_particles=1, prior_as_prop=False, latent_params=None,
                       coupled_sampling=False, sampling_shifts=None):

        if latent_params == None:
            if prior_as_prop:
                bs = x.shape[0]
                latent_params = torch.zeros(bs, self.latent_dim).to(x.device), torch.ones(bs, self.latent_dim).to(
                    x.device)
            else:
                latent_params = self.encode(x)

        latent_params_repeat = (latent_params[0].repeat(num_particles, 1), latent_params[1].repeat(num_particles, 1))

        if coupled_sampling:
            u_base = torch.rand_like(latent_params[0])
            us = []
            for i in range(num_particles):
                u = u_base + torch.from_numpy(sampling_shifts[i]).to(u_base).unsqueeze(0)  # (bs, dim)
                u = torch.clamp(u % 1.0, 1e-6)
                us.append(u)
            us = torch.cat(us, dim=0)
            z = Normal(*latent_params_repeat).icdf(us)
        else:
            z = self.latent_dist_sample(latent_params_repeat)

        obs_params = self.decode(z)
        log_cond = self.obs_model.log_probs(obs_params, x.repeat((num_particles, 1, 1, 1)))
        log_p = self.log_prior_prob(z)
        log_q = self.log_post_prob(latent_params_repeat, z)

        elbo = log_cond + log_p - log_q  # (bs * N,)
        elbo = elbo.view((num_particles, -1))  # (N, bs)

        iwae = torch.logsumexp(elbo, dim=0) - np.log(num_particles)
        elbo = elbo.mean(dim=0)

        joint = log_cond + log_p
        joint = joint.view((num_particles, -1)).mean(dim=0)

        return elbo, iwae, joint

    def sample(self, num=64):
        z = torch.randn(num, self.latent_dim).to(next(self.parameters()).device)
        obs_params = self.decode(z)
        sample = self.obs_model.sample(obs_params)

        return sample

    def reconstruct(self, x):
        x = x.view(-1, 784).float()
        latent_params = self.encode(x)
        z = self.latent_dist_sample(latent_params)

        obs_params = self.decode(z)
        sample = self.obs_model.sample(obs_params)

        return sample

    def log_prior_prob(self, z):
        return torch.sum(Normal(0, 1).log_prob(z), dim=-1)

    def log_post_prob(self, params, z):
        z_mu, z_std = params
        return torch.sum(Normal(z_mu, z_std).log_prob(z), dim=-1)

    def latent_dist_sample(self, params):
        mu, std = params
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class BinaryVAE(VAE):
    """
    This is the BinaryVAE model in the IWAE paper https://arxiv.org/pdf/1509.00519.pdf
    which is used with (dynamically) binarized (E)MNIST.

    Bernoulli observation model, Gaussian latent space.
    """

    def __init__(self, hidden_dim=200, latent_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.obs_model = obs_model.BernoulliOM()

        self.enc_fc1 = nn.Linear(784, self.hidden_dim)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.enc_fc31 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.enc_fc32 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.dec_fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.dec_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec_fc3 = nn.Linear(self.hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = x.view(-1, 28 * 28)
        h1 = torch.tanh(self.enc_fc1(x))
        h2 = torch.tanh(self.enc_fc2(h1))
        return self.enc_fc31(h2), torch.exp(self.enc_fc32(h2))

    def decode(self, z):
        """Take a z and output a probability (ie Bernoulli param) on each dim"""
        h1 = torch.tanh(self.dec_fc1(z))
        h2 = torch.tanh(self.dec_fc2(h1))
        return self.sigmoid(self.dec_fc3(h2))


class BinaryVAELossy(VAE):
    """
      This is the two-layer BinaryVAE model in the IWAE paper which is used with (dynamically) binarized MNIST.
      This model is designed for the toy lossy compression.
    """

    def __init__(self, hidden_dim=200, latent_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hyperlatent_dim = latent_dim // 2
        self.hyperhidden_dim = hidden_dim // 2

        self.obs_model = obs_model.BernoulliOM()
        self.latent_obs_model = obs_model.DiscreteGassianOM()

        self.enc_fc1 = nn.Linear(784, self.hidden_dim)
        self.enc_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.enc_fc3 = nn.Linear(self.hidden_dim, self.latent_dim)
        # self.enc_fc31 = nn.Linear(self.hidden_dim, self.latent_dim)
        # self.enc_fc32 = nn.Linear(self.hidden_dim, self.latent_dim)

        self.enc_hyper_fc1 = nn.Linear(self.latent_dim, self.hyperhidden_dim)
        self.enc_hyper_fc2 = nn.Linear(self.hyperhidden_dim, self.hyperhidden_dim)
        self.enc_hyper_fc31 = nn.Linear(self.hyperhidden_dim, self.hyperlatent_dim)
        self.enc_hyper_fc32 = nn.Linear(self.hyperhidden_dim, self.hyperlatent_dim)

        self.dec_fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.dec_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec_fc3 = nn.Linear(self.hidden_dim, 784)
        self.sigmoid = nn.Sigmoid()

        self.dec_hyper_fc1 = nn.Linear(self.hyperlatent_dim, self.hyperhidden_dim)
        self.dec_hyper_fc2 = nn.Linear(self.hyperhidden_dim, self.hyperhidden_dim)
        self.dec_hyper_fc31 = nn.Linear(self.hyperhidden_dim, self.latent_dim)
        self.dec_hyper_fc32 = nn.Linear(self.hyperhidden_dim, self.latent_dim)

    def encode_latent(self, x):
        x = x.view(-1, 28 * 28)
        out1 = torch.tanh(self.enc_fc1(x))
        out2 = torch.tanh(self.enc_fc2(out1))
        return self.enc_fc3(out2)
        # return self.enc_fc31(out2), torch.exp(self.enc_fc32(out2))

    def encode_hyperlatent(self, y):
        out1 = torch.tanh(self.enc_hyper_fc1(y))
        out2 = torch.tanh(self.enc_hyper_fc2(out1))
        return self.enc_hyper_fc31(out2), torch.exp(self.enc_hyper_fc32(out2))

    def decode_latent(self, y):
        out1 = torch.tanh(self.dec_fc1(y))
        out2 = torch.tanh(self.dec_fc2(out1))
        return self.sigmoid(self.dec_fc3(out2))

    def decode_hyperlatent(self, z):
        out1 = torch.tanh(self.dec_hyper_fc1(z))
        out2 = torch.tanh(self.dec_hyper_fc2(out1))
        return self.dec_hyper_fc31(out2), torch.exp(self.dec_hyper_fc32(out2))

    def latent_dist_sample(self, params):
        # currently use box-shaped uniform distribution
        mu = params
        if self.training:
            uniform_noise = torch.rand(mu.shape) - 0.5
            return mu + uniform_noise.to(mu.device)
        else:
            return torch.round(mu)

    def hyperlatent_dist_sample(self, params):
        mu, std = params
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def reconstruct(self, x):
        x = x.view(-1, 784).float()
        latent_params = self.encode_latent(x)
        y = self.latent_dist_sample(latent_params)
        obs_params = self.decode_latent(y)
        # sample = self.obs_model.sample(obs_params)
        sample = obs_params.detach()

        return sample

    def loss(self, x, lambd=1.0, bound="ELBO", num_particles=1, latent_params=None, hyperlatent_params=None,
             reduce_op=torch.mean,
             return_list=False):
        assert bound in ["ELBO", "IWAE"]

        if latent_params == None:
            latent_params = self.encode_latent(x)
        y = self.latent_dist_sample(latent_params)
        obs_params = self.decode_latent(y)
        loss_distort = - reduce_op(self.obs_model.log_probs(obs_params, x))

        elbo, iwae, _ = self.compute_bounds(y, num_particles=num_particles, hyperlatent_params=hyperlatent_params)
        if bound == "ELBO":
            loss_rate = - reduce_op(elbo)
        else:
            loss_rate = - reduce_op(iwae)

        loss_distort = loss_distort * np.log2(np.e) / 784.
        loss_rate = loss_rate * np.log2(np.e) / 784.
        loss = lambd * loss_distort + loss_rate

        if return_list:
            list = {"obs_params": obs_params}

            return loss, loss_rate, loss_distort, list
        else:
            return loss, loss_rate, loss_distort

    def compute_bounds(self, y, num_particles=1, prior_as_prop=False, hyperlatent_params=None):

        if hyperlatent_params == None:
            if prior_as_prop:
                bs = y.shape[0]
                hyperlatent_params = torch.zeros(bs, self.hyperlatent_dim).to(y.device), torch.ones(bs,
                                                                                                    self.hyperlatent_dim).to(
                    y.device)
            else:
                hyperlatent_params = self.encode_hyperlatent(y)

        hyperlatent_params_repeat = (
            hyperlatent_params[0].repeat(num_particles, 1), hyperlatent_params[1].repeat(num_particles, 1))
        z = self.hyperlatent_dist_sample(hyperlatent_params_repeat)

        obs_params = self.decode_hyperlatent(z)
        log_cond = self.latent_obs_model.log_probs(obs_params, y.repeat((num_particles, 1)))
        log_p = self.log_prior_prob(z)
        log_q = self.log_post_prob(hyperlatent_params_repeat, z)

        # print(log_cond.mean().item(), log_q.mean().item(), log_p.mean().item(),)

        elbo = log_cond + log_p - log_q  # (bs * N,)
        elbo = elbo.view((num_particles, -1))  # (N, bs)

        iwae = torch.logsumexp(elbo, dim=0) - np.log(num_particles)
        elbo = elbo.mean(dim=0)

        joint = log_cond + log_p
        joint = joint.view((num_particles, -1)).mean(dim=0)

        return elbo, iwae, joint


class BetaBinomialVAE(VAE):
    """BetaBinomial observation model, Gaussian latent space.

      This model is used for regular MNIST.
    """

    def __init__(self, hidden_dim=200, latent_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.obs_model = obs_model.BetaBinomialOM()

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.register_buffer('n', torch.ones(100, 784) * 255.)

        self.fc1 = nn.Linear(784, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn21 = nn.BatchNorm1d(self.latent_dim)
        self.bn22 = nn.BatchNorm1d(self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 784 * 2)

    def encode(self, x):
        """Return mu, sigma on latent"""
        x = x.view(-1, 28 * 28)
        h = x / 255.  # otherwise we will have numerical issues
        h = F.relu(self.bn1(self.fc1(h)))
        return self.bn21(self.fc21(h)), torch.exp(self.bn22(self.fc22(h)))

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.fc4(h)
        log_alpha, log_beta = torch.split(h, 784, dim=1)
        return torch.exp(log_alpha), torch.exp(log_beta), self.n
