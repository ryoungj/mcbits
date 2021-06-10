import numpy as np
from numpy.random import randint, choice


class ToyModel:
    """Discrete latent variable model of the form p(x,z) = p(x|z)p(z)

    The joint is known.
    """

    def __init__(self, latent_alphabet_size, alphabet_size):
        self.als = alphabet_size

        # initalize random joint p(x,z)
        joint_counts = randint(1, 20, size=(latent_alphabet_size, alphabet_size))
        self.joint_probs = joint_counts / joint_counts.sum()

        # compute prior p(z) = sumx p(x,z)
        self.z_probs = self.joint_probs.sum(axis=1)

        # compute marginal p(x) = sumz p(x,z)
        self.x_probs = self.joint_probs.sum(axis=0)

        # compute conditional p(x|z) =  p(x,z) / p(z)
        self.cond_probs = self.joint_probs / self.z_probs[:, None]

        # compute posterior p(z|x) =  p(x,z) / p(x)
        self.posterior_probs = self.joint_probs / self.x_probs[None, :]

    def sample_message(self, message_length):
        """Sample x from p(x)"""
        return [choice(self.als, p=self.x_probs) for _ in range(
            message_length)]

    def log_prob(self, message):
        """Compute the log probability of the message"""
        log_prob = sum([np.log2(self.x_probs[x]) for x in message])

        return log_prob

    @property
    def entropy(self):
        """Compute the the entropy"""
        return np.sum(- self.x_probs * np.log2(self.x_probs))

    def naive_code(self, q_probs):
        """Computes the negative joint, i.e., E_p(X)[E_q(Z|X)[-log p(X,Y)]]"""
        code_length = 0.
        for x, x_prob in enumerate(self.x_probs.flatten()):  # E_p(X)[ ... ]
            elbo = 0.
            for z in range(len(q_probs[:, x])):  # E_q(Z|X)[ ... ]
                elbo -= q_probs[z, x] * np.log2(self.joint_probs[z, x])  # -log p(X,Y)
            code_length += elbo * x_prob
        return code_length

    def elbo_code(self, q_probs):
        """Compute the negative elbo bound"""
        code_length = 0.
        for x, x_prob in enumerate(self.x_probs.flatten()):  # E_p(X)[ ... ]
            elbo = 0.
            for z in range(len(q_probs[:, x])):  # E_q(Z|X)[ ... ]
                elbo -= q_probs[z, x] * (np.log2(self.joint_probs[z, x]) - np.log2(
                    q_probs[z, x]))
            code_length += elbo * x_prob
        return code_length

    def is_code(self, q_probs, num_particles=10, sample_size=100):
        """Compute the negative iwae bound"""
        code_length = 0.
        for x, x_prob in enumerate(self.x_probs.flatten()):  # E_p(X)[ ... ]
            sum_is = 0.
            for _ in range(sample_size):
                is_bound = 0.
                for _ in range(num_particles):
                    z = np.random.choice(len(q_probs[:, x]), p=q_probs[:, x])
                    is_bound += self.joint_probs[z, x] / q_probs[z, x]
                sum_is += np.log2(np.sum(is_bound) / num_particles)
            code_length += (-sum_is / sample_size) * x_prob
        return code_length

    def coupled_is_code(self, q_probs, sampling_shifts, num_particles=10, sample_size=100):
        """Compute the negative coupled iwae bound"""
        code_length = 0.
        for x, x_prob in enumerate(self.x_probs.flatten()):  # E_p(X)[ ... ]
            sum_is = 0.
            q_cum = np.cumsum(q_probs[:, x])
            for _ in range(sample_size):
                qmc_bound = 0.

                u = np.random.rand()
                for i in range(0, num_particles):
                    u_shift = (u + sampling_shifts[i]) % 1.0
                    z = np.searchsorted(q_cum, u_shift, side='right')
                    qmc_bound += self.joint_probs[z, x] / q_probs[z, x]
                sum_is += np.log2(np.sum(qmc_bound) / num_particles)
            code_length += (-sum_is / sample_size) * x_prob
        return code_length


    def ais_code(self, q_probs, betas, sample_size=100):
        """Compute the negative ais bound"""
        def ais_transition(z, weights):
            prop_z = np.random.randint(len(weights) - 1)
            prop_z = prop_z if prop_z < z else prop_z + 1
            a = min(1.0, float(weights[prop_z] / weights[z]))
            accept = (np.random.rand() <= a)
            return prop_z if accept else z

        assert betas[0] == 1
        assert betas[-1] == 0

        code_length = 0.
        for x, x_prob in enumerate(self.x_probs.flatten()):  # E_p(X)[ ... ]
            def get_f(x, b):
                return np.power(q_probs[:, x], 1 - b) * np.power(self.joint_probs[:, x], b)

            sum_ais = 0.

            for _ in range(sample_size):
                n = len(betas) - 1
                i = n - 1
                z = np.random.choice(len(q_probs[:, x]), p=q_probs[:, x])
                fi = get_f(x, betas[i])
                ais = np.log2(fi[z]) - np.log2(q_probs[z, x])

                while i > 0:
                    z = ais_transition(z, fi)
                    fim1 = get_f(x, betas[i - 1])
                    ais += np.log2(fim1[z]) - np.log2(fi[z])
                    i -= 1
                    fi = fim1

                sum_ais += ais

            code_length += - (sum_ais / sample_size) * x_prob
        return code_length


