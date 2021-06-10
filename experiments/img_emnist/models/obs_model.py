""" Observation models """
import numpy as np

from torch.distributions import Normal, Beta, Binomial, Bernoulli
import torch
from torch import lgamma


class BernoulliOM():
    def __init__(self):
        super(BernoulliOM, self).__init__()

    def log_probs(self, obs_params, x):
        probs = obs_params
        x = x.reshape(probs.shape)
        return torch.sum(Bernoulli(probs).log_prob(x), dim=1)

    def sample(self, obs_params):
        probs = obs_params
        return Bernoulli(probs).sample()

    def counts(self, probs):
        probs = np.stack((1. - probs, probs), axis=-1)
        probs = np.reshape(probs, (-1, np.shape(probs)[-1]))
        return probs


class BetaBinomialOM():
    def __init__(self):
        super(BetaBinomialOM, self).__init__()

    def beta_binomial_log_pdf(k, n, alpha, beta):
        numer = lgamma(n + 1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(
            alpha + beta)
        denom = lgamma(k + 1) + lgamma(n - k + 1) + lgamma(
            n + alpha + beta) + lgamma(
            alpha) + lgamma(beta)
        return numer - denom

    def log_probs(self, obs_params, x):
        alpha, beta, n = obs_params
        pdf = self.beta_binomial_log_pdf(x, n, alpha, beta)
        return torch.sum(pdf, dim=1)

    def sample(self, obs_params):
        x_alpha, x_beta, n = obs_params
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(255, p)
        x_sample = binomial.sample()
        x_sample = x_sample.float() / 255.
        return x_sample

    def pdf(self, obs_params):
        pass


class DiscreteGassianOM():
    """
    This is a discrete Gaussian observation model whose probability distribution is computed as Gaussian convolved with
    U(-0.5, 0.5)
    """

    def __init__(self):
        super().__init__()

    def log_probs(self, obs_params, x):
        means, stds = obs_params
        dist = Normal(means, stds)  # the continuous Gaussian
        probs = torch.clamp(dist.cdf(x + 0.5) - dist.cdf(x - 0.5), min=1e-10)
        # print(probs.max().item(), probs.min().item(), probs.mean().item())
        return torch.sum(torch.log(probs), dim=1)
