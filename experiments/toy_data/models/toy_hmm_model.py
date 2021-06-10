import numpy as np
from numpy.random import randint, choice
from scipy.special import logsumexp


def matrix_choice(p, axis=1):
    assert p.ndim == 2
    r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
    return (p.cumsum(axis=axis) > r).argmax(axis=axis)


class HMMToyModel:
    """Discrete HMM model of the form.

    This has prior p(z_1), transition operator T(z_i+1 | z_i), emission p(x_i | z_i)
    z_1 ~ mu(z_1)
    z_i+1 ~ f(z_i+1 | z_i)
    x_i ~ g(x_i | z_i)

    The joint is known.
    """

    def __init__(self, latent_alphabet_size, alphabet_size, length):
        self.lals = latent_alphabet_size
        self.als = alphabet_size
        self.length = length
        self.dtype = np.int
        self.ndim = 1

        # initalize random prior mu(z), transition operator f(z_i+1 | z_i), emission g(x_i | z_i)
        self.prior_counts = randint(1, 20, size=(latent_alphabet_size,))
        self.transition_counts = randint(1, 20, size=(latent_alphabet_size, latent_alphabet_size))
        self.emission_counts = randint(1, 20, size=(latent_alphabet_size, alphabet_size))

        self.prior_probs = self.prior_counts / self.prior_counts.sum()
        self.transition_probs = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        self.emission_probs = self.emission_counts / self.emission_counts.sum(axis=1, keepdims=True)

    def sample_message(self, message_length):
        """Sample x from the HMM generative process"""

        # In HMM, a single symbol corresponds to a sequence (x_1, x_2, ..., x_length)
        message = []
        for _ in range(message_length):
            # sample z_1 from mu(z)
            zi = choice(self.lals, p=self.prior_probs.flatten())
            x = [choice(self.als, p=self.emission_probs[zi].flatten())]

            for _ in range(self.length - 1):
                # sample z_i from f(z_i | z_i-1)
                zi = choice(self.lals, p=self.transition_probs[zi].flatten())
                # sample x_i from g(x_i | z_i)
                x.append(choice(self.als, p=self.emission_probs[zi].flatten()))
            message.append(x)
        return message

    def log_px_naive(self, x):
        """Compute log p(x) = log p(x_1, x_2, ..., x_length) by enumerating the latent space.

        This naive approach is only for debugging.
        """
        latent_space = np.zeros([self.lals for _ in range(self.length)])
        for idx, _ in np.ndenumerate(latent_space):
            log_latent_probs = np.log2(self.prior_probs[idx[0]])
            log_obs_probs = np.log2(self.emission_probs[idx[0], x[0]])

            for i in range(1, self.length):
                log_latent_probs += np.log2(self.transition_probs[idx[i - 1], idx[i]])
                log_obs_probs += np.log2(self.emission_probs[idx[i], x[i]])

            latent_space[idx] = np.exp2(log_latent_probs + log_obs_probs)

        log_prob = np.log2(np.sum(latent_space))
        return log_prob

    def log_px_exact(self, x):
        """
        Compute log p(x) = log p(x_1, x_2, ..., x_length) using the exact forward algorithm.
        """
        pred_dist = self.prior_probs  # predictive dist: p(z_i | x_1:i-1), initialize to the prior dist
        log_prob = 0.  # log p(x), base 2

        for i in range(self.length):
            ppf = self.emission_probs[:, x[i]]  # positive potential function, defined as emission_probs(:, x_i)
            inc_nconst = np.dot(pred_dist, ppf)  # incremental normalizing constant
            log_prob += np.log2(inc_nconst)
            filter_dist = pred_dist * ppf / inc_nconst  # filtering distribution: p(z_i | x_1:i)
            pred_dist = np.dot(self.transition_probs.T, filter_dist)  # predictive dist: p(z_i+1 | x_1:i)

        return log_prob

    def log_px_smc(self, x, N=10, resampling=True, adaptive=False, prop_counts=None, resample_crit=None):
        """Compute log p(x) = log p(x_1, x_2, ..., x_length) using sequential monte carlo method.

        By default, an i.i.d. uniform proposal and multinomial resampling is used for SMC.

        Args:
            N: the number of particles
            resampling: apply resampling
            adaptive: use adaptive resampling
            prop_counts: proposal distribution
        """

        # FUNCTIONS
        if prop_counts is None:
            def unifrom_prop(x, z=None):
                return np.ones((self.lals,)) / self.lals

            prop_counts = unifrom_prop

        if resample_crit is None:
            def ess_crit(w):
                return (1.0 / np.sum(w ** 2)) < len(w) / 2

            resample_crit = ess_crit

        def resample(w, y):
            # Multinomial resampling
            idxs = choice(N, p=w, size=(N,))
            y = y[idxs]
            return np.zeros((N,)), y

        # SETUP
        log_weights = np.zeros((N,))  # importance weights for each particle
        log_prob = 0.  # log p(x)
        z = np.zeros((N, 0), dtype=np.int)  # simulated particles
        num_resample = 0

        for t in range(self.length):
            # PROPOSE EXTENSION
            q = prop_counts(x[:t + 1], z[:, :t] if t > 0 else None)
            if q.ndim == 1:
                q = np.tile(q, (N, 1))
            z_t = matrix_choice(q)
            z = np.append(z, np.expand_dims(z_t, axis=1), axis=1)

            # COMPUTE INCREMENTAL WEIGHTS
            log_q = np.log(q[np.arange(N), z_t])
            log_e = np.log(self.emission_probs[:, x[t]][z_t])
            if t == 0:
                log_p = np.log(self.prior_probs[z_t])
            else:
                log_p = np.log(self.transition_probs[tuple(z[:, -2:].T)])
            log_alpha = log_e + log_p - log_q

            # UPDATE WEIGHTS
            log_weights += log_alpha

            resampled = False
            if resampling:
                weights = np.exp(log_weights - logsumexp(log_weights))
                if adaptive and (not resample_crit(weights)):
                    # if adaptive resampling is used and the resampling criteria is not satisfied (i.e., ESS >= N/2)
                    continue
                else:
                    resampled = True
                    num_resample += 1
                    log_prob += logsumexp(log_weights) - np.log(N)
                    log_weights, z = resample(weights, z)

        if not resampled:
            log_prob += logsumexp(log_weights) - np.log(N)

        return log_prob / np.log(2), num_resample / self.length

    def log_prob(self, message, method="exact", base=2, **kwargs):
        """Compute the log probability of the message"""
        if method == "exact":
            log_prob = sum([self.log_px_exact(x) for x in message])
        elif method == "smc":
            log_prob = 0.0
            prop_resample = 0.0
            for x in message:
                y, z = self.log_px_smc(x, **kwargs)
                log_prob += y
                prop_resample += z
            # print("Mean proporation of resample: ", prop_resample / len(message))
        else:
            raise NotImplementedError

        return log_prob * np.log2(base)
