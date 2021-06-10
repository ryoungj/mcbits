# -*- coding: utf-8 -*-
"""
Coders implemented:
  * SimpleCoder
  * StochasticCoder
  * BitBackCoder
  * ISBitsBackCoder
  * AISBitsBackCoder
  * CISBitsBackCoder
  * SMCBitsBackCoder
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from . import rans, util
from .rans import RANSStack
from scipy.special import logsumexp
import time
import random

CODER_LIST = [
    "SimpleCoder",
    "StochasticCoder",
    "BitsBackCoder",
    "AISBitsBackCoder",
    "ISBitsBackCoder",
    "CISBitsBackCoder",
    "SMCBitsBackCoder",
]

PRINT_FREQ = 10


class Coder(object):
    """The base coder class"""

    def __init__(self, stack=None, multidim=True, **kwargs):
        """Initialize the coder.

            Args:
                stack: the pre-initialized rANS stack
                multidim: whether the symbol/latent are vector- or scalar-valued
        """
        self.stack = RANSStack(**kwargs) if stack is None else stack  # initialize the rANS stack
        self.num_encoded = 0  # number of encoded symbols

        self.multidim = multidim
        if self.multidim:
            self.pop = self.pop_fun_multi
            self.append = self.append_fun_multi
            self.sample = self.sample_fun_multi
        else:
            self.pop = self.pop_fun
            self.append = self.append_fun
            self.sample = self.sample_fun

    @property
    def bit_length(self):
        """Return the **total** bit length of the rANS message"""
        return self.stack.bit_length

    @property
    def net_bit_length(self):
        """Return the **net** bit length of the rANS message"""
        return self.stack.net_bit_length

    def reset(self):
        """Reset the coder"""
        self.num_encoded = 0
        self.stack.reset()

    def pop_fun(self, stack, count_stat_func):
        """Pop a scalar-valued symbol from the stack

        Args:
            stack: the rANS stack object
            count_stat_func: the statistical function of the distribution for popping the symbol

        Returns:
            sym: the popped scalar-valued symbol
            stack: the rANS stack
        """
        stack.update_count_stat_func(count_stat_func, mprec=count_stat_func.precision)
        sym = stack.pop()
        return sym, stack

    def append_fun(self, sym, stack, count_stat_func):
        """Append a scalar-valued symbol to the stack

        Args:
            sym: the scalar-valued symbol to push
            stack: the rANS stack object
            count_stat_func: the statistical function of the distribution for pushing the symbol

        Returns:
            stack: the rANS stack
        """
        stack.update_count_stat_func(count_stat_func, mprec=count_stat_func.precision)
        stack.append(int(sym))
        return stack

    def sample_fun(self, count_stat_func):
        """Sample a scalar-valued symbol

        Args:
            count_stat_func: the statistical function of the distribution for sampling the symbol

        Returns:
            sym: the sampled scalar-valued symbol
        """
        if not self.stack.use_statfunc:
            count = count_stat_func.count
            dist = np.array(count) / float(sum(count))
            sym = np.random.choice(len(dist), p=dist)
        else:
            cdf, ppf = count_stat_func.stat_func
            max_cumcounts = cdf(-1)
            # sample an integert in the range [0, 1 << mprec), and get the latent
            sym = ppf(np.random.randint(max_cumcounts))

        return sym

    def get_count_stat_func_multi(self, count_stat_func):
        """Get multi-dim statistical functions as a list"""

        if isinstance(count_stat_func, list) and isinstance(count_stat_func[0], util.CountStatFunc):
            count_stat_funcs = count_stat_func
        else:
            assert isinstance(count_stat_func, util.CountStatFuncMulti)
            count_stat_funcs = count_stat_func.count_stat_funcs

        return count_stat_funcs

    def pop_fun_multi(self, stack, count_stat_func):
        """Pop a vector-valued symbol from the stack

        Args:
            stack: the rANS stack object
            count_stat_func: the (multi-dim) statistical function of the distribution for popping the symbol

        Returns:
            sym: the popped vector-valued symbol
            stack: the rANS stack
        """
        sym = []
        count_stat_funcs = self.get_count_stat_func_multi(count_stat_func)
        for f in count_stat_funcs:
            s, stack = self.pop_fun(stack, f)
            sym.append(s)
        sym = np.stack(sym)
        return sym, stack

    def append_fun_multi(self, sym, stack, count_stat_func):
        """Append a vector-valued symbol to the stack

        Args:
            sym: the vector-valued symbol to push
            stack: the rANS stack object
            count_stat_func: the (multi-dim) statistical function of the distribution for pushing the symbol

        Returns:
            stack: the rANS stack
        """
        sym = sym.flatten()
        count_stat_funcs = self.get_count_stat_func_multi(count_stat_func)
        # GO IN REVERSED ORDER!
        for i, s in reversed(list(enumerate(sym))):
            f = count_stat_funcs[i]
            stack = self.append_fun(s, stack, f)
        return stack

    def sample_fun_multi(self, count_stat_func):
        """Sample a vector-valued symbol

        Args:
            count_stat_func: the (multi-dim) statistical function of the distribution for sampling the symbol

        Returns:
            sym: the sampled vector-valued symbol
        """
        sym = []
        # TODO: parallelize this computation
        count_stat_funcs = self.get_count_stat_func_multi(count_stat_func)
        for f in count_stat_funcs:
            sym.append(self.sample_fun(f))
        sym = np.stack(sym)
        return sym

    def encode_sym(self, sym):
        """Encoding a single symbol (either scalar- or vector-valued)

        This method needs to be implemented by all coders.
        """
        raise NotImplementedError

    def decode_sym(self):
        """Decoding a single symbol (either scalar- or vector-valued)

        This method needs to be implemented by all coders.
        """
        raise NotImplementedError

    def encode(self, seq, print_progress=False):
        """Encoding a message (a sequence of symbols)"""
        if print_progress:
            print("..Start encoding")
        start_time = time.time()

        for i, sym in enumerate(seq):
            self.encode_sym(sym)
            self.num_encoded += 1
            if print_progress and self.num_encoded % PRINT_FREQ == 0:
                print("\r..encoded {}/{} symbols, net length: {:.3f} bits/dim, "
                      "total length {:.3f} bits/dim, encoding speed {:.3f}s/sym"
                      .format(self.num_encoded, len(seq),
                              self.net_bit_length / np.prod(sym.shape) / self.num_encoded,
                              self.bit_length / np.prod(sym.shape) / self.num_encoded,
                              (time.time() - start_time) / self.num_encoded))

    def decode(self, num, print_progress=False):
        """Decoding a message (a sequence of symbols)"""
        if print_progress:
            print("..Start decoding")
        start_time = time.time()

        seq = []
        num_decoded = 0
        while num_decoded < num:
            sym = self.decode_sym()
            seq.insert(0, sym)
            self.num_encoded -= 1
            num_decoded += 1
            if print_progress and self.num_encoded % PRINT_FREQ == 0:
                print("\r..decoded {}/{} symbols, decoding speed {:.3f}s/sym"
                      .format(num_decoded, num,
                              (time.time() - start_time) / num_decoded))

        return seq


class SimpleCoder(Coder):
    """The simple coder which assumes access to the marginal distribution p(x)"""

    def __init__(self, count_stat_func, **kwargs):
        super(SimpleCoder, self).__init__(**kwargs)
        self.count_stat_func = count_stat_func  # the marginal count stat func

    def encode_sym(self, sym):
        self.stack = self.append(sym, self.stack, self.count_stat_func)

    def decode_sym(self):
        sym, self.stack = self.pop(self.stack, self.count_stat_func)
        return sym


class LatentVariableCoder(Coder):
    """The base latent variable coder"""

    def __init__(self, get_prior_count_stat_func=None, get_cond_count_stat_func=None, get_prop_count_stat_func=None,
                 **kwargs):
        """Initialize the latent variable coder.

        Args:
            get_prior_count_stat_func: function for getting the stat func of prior dist., signature: None -> p(z)
            get_cond_count_stat_func: function for getting the stat func of conditional dist., signature: z -> p(x|z)
            get_prop_count_stat_func: function for getting the stat func of proposal dist., signature: x -> q(z|x)
        """
        super(LatentVariableCoder, self).__init__(**kwargs)

        self.get_prior_count_stat_func = get_prior_count_stat_func
        self.get_cond_count_stat_func = get_cond_count_stat_func
        self.get_prop_count_stat_func = get_prop_count_stat_func


class StochasticCoder(LatentVariableCoder):
    """The naive stochastic coder without bits-back

    Note that this coder suffers from discretization error which increases the net bitrate by  -log(precision).
    """

    def __init__(self, **kwargs):
        super(StochasticCoder, self).__init__(**kwargs)

    def encode_sym(self, sym):
        prop_count_stat_func = self.get_prop_count_stat_func(sym)
        # sample z ~ q(z|x)
        latent = self.sample(prop_count_stat_func)
        # encode x with p(x|z)
        self.stack = self.append(sym, self.stack, self.get_cond_count_stat_func(latent))
        # encode z with p(z)
        self.stack = self.append(latent, self.stack, self.get_prior_count_stat_func())

    def decode_sym(self):
        # decode z with p(z)
        latent, self.stack = self.pop(self.stack, self.get_prior_count_stat_func())
        # decode x with p(x|z)
        sym, self.stack = self.pop(self.stack, self.get_cond_count_stat_func(latent))
        return sym


class BitsBackCoder(LatentVariableCoder):
    """The bits-back coder (BB-ELBO)"""

    def __init__(self, **kwargs):
        super(BitsBackCoder, self).__init__(**kwargs)

    def encode_sym(self, sym):
        prop_count_stat_func = self.get_prop_count_stat_func(sym)
        # decode z with q(z|x)
        latent, self.stack = self.pop(self.stack, prop_count_stat_func)

        # encode x with p(x|z)
        cond_count_stat_func = self.get_cond_count_stat_func(latent)
        self.stack = self.append(sym, self.stack, cond_count_stat_func)
        # encode z with p(z)
        prior_count_stat_func = self.get_prior_count_stat_func()
        self.stack = self.append(latent, self.stack, prior_count_stat_func)

    def decode_sym(self):
        # decode z with p(z)
        latent, self.stack = self.pop(self.stack, self.get_prior_count_stat_func())
        # decode x with p(x|z)
        sym, self.stack = self.pop(self.stack, self.get_cond_count_stat_func(latent))
        # encode z with q(z|x)
        self.stack = self.append(latent, self.stack, self.get_prop_count_stat_func(sym))
        return sym


class AISBitsBackCoder(BitsBackCoder):
    """The bits-back coder with annealed importance sampling (BB-AIS)

    As the coding scheme of BB-AIS is very similar to Bits-Back for hierarchical models (see
    https://arxiv.org/pdf/1905.06845.pdf for details). We also implement the BitSwap version for reducing the
    initial bit cost.
    """

    def __init__(self, get_joint_count_stat_func,
                 betas, get_trans_count_stat_func=None, bitswap=False, **kwargs):
        """ Initialize the BB-AIS coder.

        Args:
            get_joint_count_stat_func: function for getting the stat func of joint dist.
            betas: the array of length `num of ais steps + 1` specifying the intermediate distributions as
                            ``f_i(z) \propto q(z| x)^(1-betas[i]) * p(x,z)^betas[i]``
            get_trans_count_stat_func: function for getting the stat func of transition dist. that leaves the
            intermediate dist invariant
            bitswap: whether apply the BitSwap trick for reducing the initial bit cost
        """

        super(AISBitsBackCoder, self).__init__(**kwargs)
        assert betas[0] == 1
        assert betas[-1] == 0
        self.get_joint_count_stat_func = get_joint_count_stat_func  # p(x, z)
        self.betas = betas  #
        self.bitswap = bitswap

        if get_trans_count_stat_func is None:
            self.get_trans_count_stat_func = self.mh_transition_count_stat_func
        else:
            self.get_trans_count_stat_func = get_trans_count_stat_func

        assert not self.multidim, "BB-AIS is not supported for high-dimension yet!"
        assert not self.stack.use_statfunc, "Using stat func for BB-AIS coder is not implemented yet! Please specify " \
                                            "the stat funcs in counts!"

    def mh_transition_count_stat_func(self, latent, counts, **kwargs):
        """The Metropolis-Hastings transition kernel with a uniform proposal"""
        n = len(counts)
        transition_counts = [0] * n
        i = latent
        for j in range(n):
            if i != j:
                transition_counts[j] = min(1, counts[j] / counts[i]) / (n - 1)
        transition_counts[i] = 1 - sum(transition_counts)
        return util.Categorical(self.stack.default_mprec, transition_counts, use_make_sum_to_M=True)

    def get_inter_counts(self, sym, i):
        """Get the counts for intermediate distributions"""
        b = self.betas[i]
        prop_b = np.power(self.get_prop_count_stat_func(sym).prob, 1 - b)
        joint_b = np.power(self.get_joint_count_stat_func(x=sym).prob, b)
        return prop_b * joint_b

    def encode_sym(self, sym):
        if self.bitswap:
            return self.encode_bitswap(sym)
        else:
            return self.encode_vanilla(sym)

    def decode_sym(self):
        if self.bitswap:
            return self.decode_bitswap()
        else:
            return self.decode_vanilla()

    def encode_vanilla(self, sym):
        n = len(self.betas) - 1

        i = n - 1
        prop_count_stat_func = self.get_prop_count_stat_func(sym)
        # decode z_n-1 ~ p_n(z)
        latent, self.stack = self.pop(self.stack, prop_count_stat_func)
        latents = [latent]

        while i > 0:
            inter_counts = self.get_inter_counts(sym, i)
            # compute T_i(.|z_i)
            mh_count_stat_func = self.get_trans_count_stat_func(latent, inter_counts)

            # decode z_i-1 ~ T_i(z | z_i) for i = n-1, ..., 1
            latent, self.stack = self.pop(self.stack, mh_count_stat_func)
            latents.insert(0, latent)
            i -= 1

        i = n - 1
        while i > 0:
            inter_counts = self.get_inter_counts(sym, i)
            mh_count_stat_func = self.get_trans_count_stat_func(latents[i - 1], inter_counts,
                                                                reversal=True)

            # encode z_i with revT_i(z | z_i-1) for i = n-1, ..., 1
            latent = latents.pop()
            self.stack = self.append(latent, self.stack, mh_count_stat_func)
            i -= 1

        latent = latents.pop()
        # encode x with p(x | z_0)
        cond_count_stat_func = self.get_joint_count_stat_func(z=latent)
        self.stack = self.append(sym, self.stack, cond_count_stat_func)
        # encode z_0 with p(z_0), note f_0(z) = p(z) p(x | z)
        latent_count_stat_func = self.get_prior_count_stat_func()
        self.stack = self.append(latent, self.stack, latent_count_stat_func)

    def decode_vanilla(self):
        n = len(self.betas) - 1
        # decode z_0 with p(z_0), note f_0(z) = p(z) p(x | z)
        latent, self.stack = self.pop(self.stack, self.get_prior_count_stat_func())
        # decode x with p(x | z_0)
        sym, self.stack = self.pop(self.stack, self.get_joint_count_stat_func(z=latent))
        latents = [latent]

        i = 1
        while i < n:
            inter_counts = self.get_inter_counts(sym, i)
            mh_count_stat_func = self.get_trans_count_stat_func(latents[i - 1], inter_counts,
                                                                reversal=True)
            # decode z_i with revT_i(z | z_i-1) for i = n-1, ..., 1
            latent, self.stack = self.pop(self.stack, mh_count_stat_func)
            latents.append(latent)
            i += 1

        i = 1
        while i < n:
            inter_counts = self.get_inter_counts(sym, i)
            mh_count_stat_func = self.get_trans_count_stat_func(latents[i], inter_counts)
            # encode z_i-1 with T_i(z | z_i) for i = n-1, ..., 1
            self.stack = self.append(latents[i - 1], self.stack, mh_count_stat_func)
            i += 1

        # encode z_n-1 ~ p_n(z)
        self.stack = self.append(latents[n - 1],
                                 self.stack,
                                 self.get_prop_count_stat_func(sym))
        return sym

    def encode_bitswap(self, sym):
        n = len(self.betas) - 1

        i = n - 1
        prop_count_stat_func = self.get_prop_count_stat_func(sym)
        # decode z_n-1 ~ p_n(z)
        latent, self.stack = self.pop(self.stack, prop_count_stat_func)

        while i > 0:
            inter_counts = self.get_inter_counts(sym, i)
            mh_count_stat_func = self.get_trans_count_stat_func(latent, inter_counts)
            # decode z_i-1 ~ T_i(z | z_i) for i = n-1, ..., 1
            next_latent, self.stack = self.pop(self.stack, mh_count_stat_func)
            # sample z1 given z2 using T2, encode z2 given z1 with revT2
            # the bits are clean, if z2 | z1 has distribution revT2 as n->infty
            # because f1 gets infinitesimally close to f2, beta1 -> beta2

            mh_count_stat_func = self.get_trans_count_stat_func(next_latent, inter_counts,
                                                                reversal=True)
            # encode z_i with revT_i(z | z_i-1) for i = n-1, ..., 1
            self.stack = self.append(latent, self.stack, mh_count_stat_func)
            # not clean bits for finite n
            i -= 1
            latent = next_latent

        # encode x with p(x | z_0)
        cond_count_stat_func = self.get_joint_count_stat_func(z=latent)
        self.stack = self.append(sym, self.stack, cond_count_stat_func)
        # encode z_0 with p(z_0), note f_0(z) = p(z) p(x | z)
        latent_count_stat_func = self.get_prior_count_stat_func()
        self.stack = self.append(latent, self.stack, latent_count_stat_func)

    def decode_bitswap(self):
        n = len(self.betas) - 1
        # decode z_0 with p(z_0), note f_0(z) = p(z) p(x | z)
        latent, self.stack = self.pop(self.stack, self.get_prior_count_stat_func())
        # decode x with p(x | z_0)
        sym, self.stack = self.pop(self.stack, self.get_joint_count_stat_func(z=latent))

        i = 1
        while i < n:
            inter_counts = self.get_inter_counts(sym, i)
            mh_count_stat_func = self.get_trans_count_stat_func(latent, inter_counts, reversal=True)

            # decode z_i with revT_i(z | z_i-1) for i = n-1, ..., 1
            next_latent, self.stack = self.pop(self.stack, mh_count_stat_func)

            mh_count_stat_func = self.get_trans_count_stat_func(next_latent, inter_counts)
            # encode z_i-1 with T_i(z | z_i) for i = n-1, ..., 1
            self.stack = self.append(latent, self.stack, mh_count_stat_func)
            latent = next_latent
            i += 1

        # encode z_n-1 ~ p_n(z)
        self.stack = self.append(latent, self.stack, self.get_prop_count_stat_func(sym))
        return sym


class ISBitsBackCoder(BitsBackCoder):
    """The bits-back coder with importance sampling (BB-IS)"""

    def __init__(self, num_particles, batch_compute=False, **kwargs):
        """Initialize the BB-IS coder.

        Args:
            num_particles: the number of particles of BB-IS for compression
            batch_compute: whether compute the statistical functions of conditional likelihood in a batch over
            particles. This might lead to decode check failure since the batched computation results (at encoding time)
            may be slightly different from the unbatched ones computed individually (at decoding time) in PyTorch due
            to non-determinism.
        """
        super(ISBitsBackCoder, self).__init__(**kwargs)
        self.num_particles = num_particles
        self.batch_compute = batch_compute

        # the uniform distribution for encoding the special particle index j
        self.iw_uniform_count_stat_func = util.Categorical(self.stack.default_mprec, np.ones(num_particles),
                                                           use_make_sum_to_M=True)

    def log_importance_weight(self, x, z, cond_count_stat_func, latent_count_stat_func, prop_count_stat_func):
        """Compute the log-importance weight of a single particle"""
        p_z = latent_count_stat_func.get_log_prob(z)
        q_z = prop_count_stat_func.get_log_prob(z)
        p_x = cond_count_stat_func.get_log_prob(x)
        return p_x + p_z - q_z

    def importance_weights(self, sym, latents, prop_count_stat_func, latent_count_stat_func, cond_count_stat_funcs):
        """Compute the importance weights of particles."""
        # TODO: parallelize the computation here over particles
        log_w_unnorm = np.array([self.log_importance_weight(sym, latent,
                                                            cond_count_stat_func,
                                                            latent_count_stat_func,
                                                            prop_count_stat_func
                                                            ) for latent, cond_count_stat_func in
                                 zip(latents, cond_count_stat_funcs)])
        # log_sum_exp trick
        max_log_w = np.max(log_w_unnorm)
        log_w = log_w_unnorm - max_log_w
        w = np.exp(log_w)
        log_sum_w = logsumexp(log_w_unnorm)
        return w, log_sum_w

    def encode_sym(self, sym):
        # POP STEP
        # compute q(z|x) and p(z) which could be reused
        prop_count_stat_func = self.get_prop_count_stat_func(sym)
        prior_count_stat_func = self.get_prior_count_stat_func()

        latents = []
        for _ in range(self.num_particles):
            # decode z_i with q(z|x)
            latent, self.stack = self.pop(self.stack, prop_count_stat_func)
            latents.append(latent)

        # compute p(x|z_i) for each i where p(x|z_j) could be reused
        if self.batch_compute and self.num_particles > 1:
            cond_count_stat_funcs = list(self.get_cond_count_stat_func(latents))
        else:
            cond_count_stat_funcs = [self.get_cond_count_stat_func(latent) for latent in latents]

        # decode j with Cat(w)
        iw_counts, log_sum_w = self.importance_weights(sym, latents, prop_count_stat_func,
                                                       prior_count_stat_func, cond_count_stat_funcs)
        iw_count_stat_func = util.Categorical(self.stack.default_mprec, iw_counts, use_make_sum_to_M=True)
        j, self.stack = self.pop_fun(self.stack, iw_count_stat_func)

        # APPEND STEP
        # encode z_k with q(z|x) for all k!=j
        for k in range(self.num_particles):
            if k != j:
                self.stack = self.append(latents[k], self.stack, prop_count_stat_func)
        # encode j with Cat(1/num_particles)
        self.stack = self.append_fun(j, self.stack, self.iw_uniform_count_stat_func)
        # encode x with p(x|z_j)
        self.stack = self.append(sym, self.stack, cond_count_stat_funcs[j])
        # encode z_j with p(z)
        self.stack = self.append(latents[j], self.stack, prior_count_stat_func)

    def decode_sym(self):
        # POP STEP
        # decode z_j with p(z)
        prior_count_stat_func = self.get_prior_count_stat_func()
        latent_j, self.stack = self.pop(self.stack, prior_count_stat_func)
        # compute p(x|z_j) which could be reused
        cond_count_stat_func_j = self.get_cond_count_stat_func(latent_j)
        # decode x with p(x|z_j)
        sym, self.stack = self.pop(self.stack, cond_count_stat_func_j)

        # decode j with Cat(1/num_particles)
        j, self.stack = self.pop_fun(self.stack, self.iw_uniform_count_stat_func)

        # decode z_k with q(z|x) for all k!=j
        latents = []
        prop_count_stat_func = self.get_prop_count_stat_func(sym)  # q(z|x)
        for k in range(self.num_particles - 1):
            latent, self.stack = self.pop(self.stack, prop_count_stat_func)
            latents.insert(0, latent)
        latents.insert(j, latent_j)

        # compute p(x|z_i) for each i
        if self.batch_compute and self.num_particles > 1:
            cond_count_stat_funcs = list(self.get_cond_count_stat_func(latents))
        else:
            cond_count_stat_funcs = [self.get_cond_count_stat_func(latent) for latent in latents]

        # APPEND STEP
        # encode j with Cat(w)
        iw_counts, _ = self.importance_weights(sym, latents, prop_count_stat_func, prior_count_stat_func,
                                               cond_count_stat_funcs)
        iw_count_stat_func = util.Categorical(self.stack.default_mprec, iw_counts, use_make_sum_to_M=True)
        self.stack = self.append_fun(j, self.stack, iw_count_stat_func)

        # encode z_i with q(z|x)
        for latent in reversed(latents):
            self.stack = self.append(latent, self.stack, prop_count_stat_func)

        return sym


unif_count_stat_func_cache = {}  # cache the uniform count stat func


class CISBitsBackCoder(ISBitsBackCoder):
    """The bits-back coder with coupled importance sampling (BB-CIS)"""

    def __init__(self, bijective_operators, **kwargs):
        """Intialize the BB-CIS coder

        Args:
            bijective_operators: the bijective operators applying to the single decoded uniform. The number of operators
            should equal to the number of particles and each operator should be a tuple (operator, inverse operator).
        """
        super(CISBitsBackCoder, self).__init__(**kwargs)

        self.operators = bijective_operators
        assert isinstance(self.operators, list)
        assert len(self.operators) == self.num_particles, \
            "The number of operators should equal to the number of particles"
        assert all([isinstance(op, tuple) and len(op) == 2 for op in self.operators]), \
            "Each operator should be a two-tuple as (operator, inverse operator)"

        if self.multidim:
            self.get_coupled_uniform_count_stat_func = self._get_coupled_uniform_count_stat_func_multi
        else:
            self.get_coupled_uniform_count_stat_func = self._get_coupled_uniform_count_stat_func

    def _get_coupled_uniform_count_stat_func(self, prop_count_stat_func, z=None):
        """Get the statistical function of the uniform distribution for the scalar-valued coupled uniform

        If `z` is None, the uniform distribution is in [0, 2^prop_mprec), otherwise [prop_cdf(z), prop_cdf(z+1)). The
        lower bound of the range (i.e., either 0 or prop_cdf(z)) is also returned, which is used as the bias for
        encoding the coupled uniform
        """
        if z is None:  # return a uniform distribution over [1, 2^prop_mprec)
            prop_mprec = prop_count_stat_func.precision
            if prop_mprec in unif_count_stat_func_cache:
                unif_count_stat_func = unif_count_stat_func_cache[prop_mprec]
            else:
                uniform_count = np.ones(1 << prop_mprec)
                unif_count_stat_func = util.Categorical(prop_mprec,
                                                        prob=uniform_count / sum(uniform_count),
                                                        count_bucket=uniform_count,
                                                        cumulative_bucket=np.insert(np.cumsum(uniform_count),
                                                                                    0, 0))
                unif_count_stat_func_cache[prop_mprec] = unif_count_stat_func
            lower = 0
        else:  # return a uniform distribution over [cdf(z), cdf(z+1))
            lower = prop_count_stat_func.cdf(z)
            count_z = prop_count_stat_func.cdf(z + 1) - lower
            mprec = int(np.ceil(np.log2(count_z)))
            if 2 ** mprec != count_z:
                # if count_z is not a power of 2, after discretizd distribution is not uniform
                # thus increase mprec to reduce discretization error
                mprec += 10
                uniform_count = np.ones(count_z)
                unif_count_stat_func = util.Categorical(mprec,
                                                        prob=uniform_count / sum(uniform_count))
            else:
                uniform_count = np.ones(1 << mprec)
                unif_count_stat_func = util.Categorical(mprec,
                                                        prob=uniform_count / sum(uniform_count),
                                                        count_bucket=uniform_count,
                                                        cumulative_bucket=np.insert(np.cumsum(uniform_count),
                                                                                    0, 0))
                unif_count_stat_func_cache[mprec] = unif_count_stat_func
        return unif_count_stat_func, lower

    def _get_coupled_uniform_count_stat_func_multi(self, prop_count_stat_func, z=None):
        """Get the statistical functions of the uniform distributions for the vector-valued coupled uniform"""
        # TODO: this can be parallized
        if z is None:
            dim = prop_count_stat_func.dim
            count_stat_func, lower = self._get_coupled_uniform_count_stat_func(prop_count_stat_func, None)
            unif_count_stat_funcs = [count_stat_func] * dim
            lowers = [lower] * dim
        else:
            unif_count_stat_funcs = []
            lowers = []
            for i, f in enumerate(prop_count_stat_func.count_stat_funcs):
                count_stat_func, lower = self._get_coupled_uniform_count_stat_func(f, z[i])
                unif_count_stat_funcs.append(count_stat_func)
                lowers.append(lower)
        return unif_count_stat_funcs, np.stack(lowers)

    def get_all_latents(self, prop_count_stat_func, u_j, j):
        """Get all latents from a single uniform"""
        # get the underlying base latent by inverse mapping
        u_base = self.operators[j][1](u_j, prop_count_stat_func)

        assert np.allclose(u_j, self.operators[j][0](u_base, prop_count_stat_func))

        us = []
        # all the other particles are transformed from the base latent by forward mapping
        for k in range(self.num_particles):
            if k != j:
                us.append(self.operators[k][0](u_base, prop_count_stat_func))

        us.insert(j, u_j)

        latents = []
        # TODO: this can be parallelized
        for k in range(self.num_particles):
            if self.multidim:
                latent = []
                for i in range(len(u_j)):
                    z = prop_count_stat_func.ppf[i](us[k][i])
                    latent.append(z)
                latents.append(np.stack(latent))
            else:
                latent = prop_count_stat_func.ppf(us[k])
                latents.append(latent)
        return latents, us

    def encode_sym(self, sym):
        # POP STEP
        # compute q(z|x) and p(z) which could be reused
        prop_count_stat_func = self.get_prop_count_stat_func(sym)  # q(z|x)
        prior_count_stat_func = self.get_prior_count_stat_func()

        # decode coupled uniform u with [0, 2^prop_mprec)
        u_count_stat_func, _ = self.get_coupled_uniform_count_stat_func(prop_count_stat_func)
        u, self.stack = self.pop(self.stack, u_count_stat_func)

        latents, us = self.get_all_latents(prop_count_stat_func, u, 0)

        # compute p(x|z_i) for each i where p(x| z_j) could be reused
        if self.batch_compute and self.num_particles > 1:
            cond_count_stat_funcs = list(self.get_cond_count_stat_func(latents))
        else:
            cond_count_stat_funcs = [self.get_cond_count_stat_func(latent) for latent in latents]

        # decode j with Cat(w)
        iw_counts, log_sum_w = self.importance_weights(sym, latents, prop_count_stat_func,
                                                       prior_count_stat_func, cond_count_stat_funcs)
        iw_count_stat_func = util.Categorical(self.stack.default_mprec, iw_counts, use_make_sum_to_M=True)
        j, self.stack = self.pop_fun(self.stack, iw_count_stat_func)

        # APPEND STEP
        # encode u_j with [cdf(z_j), cdf(z_j)+1)
        u_j, latent_j = us[j], latents[j]
        u_j_count_stat_func, lower_j = self.get_coupled_uniform_count_stat_func(prop_count_stat_func, z=latent_j)
        self.stack = self.append(u_j - lower_j, self.stack, u_j_count_stat_func)

        # encode j with Cat(1/num_particles)
        self.stack = self.append_fun(j, self.stack, self.iw_uniform_count_stat_func)
        # encode x with p(x|z_j)
        self.stack = self.append(sym, self.stack, cond_count_stat_funcs[j])
        # encode z_j with p(z)
        self.stack = self.append(latent_j, self.stack, prior_count_stat_func)

    def decode_sym(self):
        # POP STEP
        # decode z_j with p(z)
        prior_count_stat_func = self.get_prior_count_stat_func()
        latent_j, self.stack = self.pop(self.stack, prior_count_stat_func)
        # compute p(x|z_j) which could be reused
        cond_count_stat_func_j = self.get_cond_count_stat_func(latent_j)
        # decode x with p(x|z_j)
        sym, self.stack = self.pop(self.stack, cond_count_stat_func_j)

        # decode j with Cat(1/num_particles)
        j, self.stack = self.pop_fun(self.stack, self.iw_uniform_count_stat_func)

        prop_count_stat_func = self.get_prop_count_stat_func(sym)  # q(z|x)

        # decode u_j with [cdf(z_j), cdf(z_j)+1)
        u_j_count_stat_func, lower_j = self.get_coupled_uniform_count_stat_func(prop_count_stat_func, z=latent_j)
        u_j, self.stack = self.pop(self.stack, u_j_count_stat_func)
        u_j += lower_j

        latents, us = self.get_all_latents(prop_count_stat_func, u_j, j)

        # compute p(x|z_i) for each i
        if self.batch_compute and self.num_particles > 1:
            cond_count_stat_funcs = list(self.get_cond_count_stat_func(latents))
        else:
            cond_count_stat_funcs = [self.get_cond_count_stat_func(latent) for latent in latents]

        # APPEND STEP
        # encode j with Cat(w)
        iw_counts, _ = self.importance_weights(sym, latents, prop_count_stat_func, prior_count_stat_func,
                                               cond_count_stat_funcs)
        iw_count_stat_func = util.Categorical(self.stack.default_mprec, iw_counts, use_make_sum_to_M=True)
        self.stack = self.append_fun(j, self.stack, iw_count_stat_func)

        # encode coupled uniform u with [0, 2^prop_mprec)
        u = us[0]
        u_count_stat_func, _ = self.get_coupled_uniform_count_stat_func(prop_count_stat_func)
        self.stack = self.append(u, self.stack, u_count_stat_func)

        return sym


class SMCBitsBackCoder(ISBitsBackCoder):
    """The bits-back coder with sequential Monte Carlo (BB-SMC)"""

    def __init__(self, num_particles, get_trans_count_stat_func, default_symlen, resample=True, adaptive=False,
                 resample_crit=None, init_state=None, update_state=None, **kwargs):
        """Initialize the BB-SMC coder.

        Args:
            num_particles: the number of particles of the SMC coder
            get_trans_count_stat_func: function for getting the stat func of transition dist., signature: z_1:n-1, state -> p(z_n|...)
            default_symlen: the default length of a single symbol (actualluy a sequence of symbols), can be overrided in
                            encoding/decoding stages
            resample: whether apply resampling for compression. If False, it reduces to BB-IS
            adaptive: whether apply adaptive resampling
            resample_crit: the criterion for adaptive resampling
            init_state: the function for initializing the state (e.g., used with VRNN)
            update_state: the function for updating the state (e.g., used with VRNN)

        BB-SMC is also amenable for the coupling technique for reducing the initial bit cost, see detailed discussion
        in our paper.
        """
        super(SMCBitsBackCoder, self).__init__(num_particles, **kwargs)

        self.get_trans_count_stat_func = get_trans_count_stat_func
        assert self.get_prior_count_stat_func is None, "Please use `get_trans_count_stat_func` instead for SMC!"
        self.default_symlen = default_symlen
        self.resample = resample
        self.adaptive = adaptive
        self.resample_crit = resample_crit

        if init_state is None:  # if we do not use a VRNN model
            self.init_state = lambda: None
        else:
            self.init_state = init_state

        if update_state is None:  # if we do not use a VRNN model
            self.update_state = lambda cur_state=None, x_n=None, z_n=None: None
        else:
            self.update_state = update_state

        if self.adaptive:
            assert self.resample
            if self.resample_crit is None:
                def ess_crit(w):
                    return (1.0 / np.sum(w ** 2)) < (len(w) / 2)

                self.resample_crit = ess_crit

    def importance_weights(self, sym, latents, prop_count_stat_funcs, latent_count_stat_funcs, cond_count_stat_funcs,
                           weights=None):
        """Compute the incremental importance weights of particles."""
        log_w_unnorm = np.array([self.log_importance_weight(sym, latent,
                                                            cond_count_stat_func,
                                                            latent_count_stat_func,
                                                            prop_count_stat_func
                                                            ) for
                                 latent, cond_count_stat_func, latent_count_stat_func, prop_count_stat_func in
                                 zip(latents, cond_count_stat_funcs, latent_count_stat_funcs, prop_count_stat_funcs)])

        # log_sum_exp trick
        max_log_w = np.max(log_w_unnorm)
        log_w = log_w_unnorm - max_log_w
        w = np.exp(log_w)  # incremental importance weights
        if weights is None:
            weights = np.ones((self.num_particles,), np.float) / self.num_particles

        log_sum_w = logsumexp(log_w_unnorm, b=weights)  # log sum of accumulative weights
        return w, log_sum_w

    def encode(self, seq, print_progress=False):
        """Encoding a message (a sequence of `symbols`) and each `symbol` is a sequence of symbols"""
        if print_progress:
            print("..Start encoding")
        start_time = time.time()
        total_encoded_symlens = 0
        total_symlens = sum([len(sym) for sym in seq])

        # For SMC, a single symbol corresponds to a sequence (x_1, x_2, ..., x_length)
        for t, sym in enumerate(seq):
            sym_length = len(sym)
            # To enable both single- or multi- dimensional latents, here we set the array `z` and `z_prev` to be type
            # object, it could be either `int` (scalar) or `np.ndarray` (multidim).
            # TODO: there should be some better ways to implement this
            z = np.empty((self.num_particles, sym_length),
                         dtype=object)  # simulated particles, total size will be N x T
            z_prev = np.empty((self.num_particles, sym_length),
                              dtype=object)  # ancestral lineage of each particle, total size will be N x T
            z_states = np.empty((self.num_particles,), dtype=object)  # current particle states, `None`s if not needed

            A = np.zeros((self.num_particles, sym_length - 1),
                         dtype=np.int)  # resampling parent indices, total size will be N x (T - 1)

            # We store the statistical functions of all distributions computed in the POP step which will be reused
            # in the APPEND step
            q_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)  # proposal stat funcs
            f_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)  # prior/transition stat funcs
            g_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)  # cond likelihood stat funcs
            iw_count_stat_funcs = np.empty((sym_length - 1,), dtype=object)  # categorical stat funcs

            # cumulative importance weights
            w_cum = np.ones((self.num_particles,), np.float) / self.num_particles
            # resample decisions used for adaptive resampling
            resample_decision = np.zeros((sym_length - 1,), dtype=bool)

            # POP STEP
            for n in range(0, sym_length):
                if n != 0:
                    do_resample = self.resample and (
                            (not self.adaptive) or (self.adaptive and self.resample_crit(w_cum)))
                    if do_resample:
                        # decode A_n-1 ~ Cat(w_n-1) for each particle
                        A_n = []
                        iw_count_stat_func = util.Categorical(self.stack.default_mprec, w_cum, use_make_sum_to_M=True)
                        for i in range(self.num_particles):
                            parent, self.stack = self.pop_fun(self.stack, iw_count_stat_func)
                            A_n.append(parent)
                        A_n = np.array(A_n)
                        z_prev = z_prev[A_n]  # re-assign the ancestral particles
                        z_states = z_states[A_n]  # re-assign the particle states

                        A[:, n - 1] = A_n
                        w_cum = np.ones((self.num_particles,), np.float) / self.num_particles  # reset weights
                        resample_decision[n - 1] = True
                        iw_count_stat_funcs[n - 1] = iw_count_stat_func
                    else:
                        # no resampling, simply inherit from itself for each particle
                        A_n = np.arange(self.num_particles)
                        A[:, n - 1] = A_n
                        resample_decision[n - 1] = False

                x_n = sym[n]
                z_n = []
                q_count_stat_funcs_n = []
                f_count_stat_funcs_n = []
                g_count_stat_funcs_n = []
                for i in range(self.num_particles):
                    # compute q(z_n | x_n, z_1:n-1^i), p(z_n | z_1:n-1^i) for each i which could be reused
                    if n == 0:
                        # get prior dist
                        cur_state = self.init_state()
                        latent_count_stat_func = self.get_trans_count_stat_func(None,
                                                                                cur_state)  # p(z_0^i)
                    else:
                        # get transition dist
                        cur_state = z_states[i]
                        latent_count_stat_func = self.get_trans_count_stat_func(z_prev[i, n - 1],
                                                                                cur_state)  # p(z_n^i | z_1:n-1^i)

                    prop_count_stat_func = self.get_prop_count_stat_func(x_n, cur_state)  # q(z_n^i | x_n, z_1:n-1^i)

                    # decode z_n^i ~ q(z_n | x_n, z_1:n-1^i) i.i.d. for each particle
                    latent, self.stack = self.pop(self.stack, prop_count_stat_func)

                    # compute p(x_n | z_n^i) for each i which could be reused
                    cond_count_stat_func = self.get_cond_count_stat_func(latent, cur_state)

                    next_state = self.update_state(cur_state, x_n, latent)
                    z_states[i] = next_state
                    z_n.append(latent)
                    q_count_stat_funcs_n.append(prop_count_stat_func)
                    f_count_stat_funcs_n.append(latent_count_stat_func)
                    g_count_stat_funcs_n.append(cond_count_stat_func)
                w_n, log_sum_w = self.importance_weights(x_n, z_n, q_count_stat_funcs_n, f_count_stat_funcs_n,
                                                         g_count_stat_funcs_n,
                                                         weights=w_cum)  # compute importance weights
                z[:, n] = z_n
                z_prev[:, n] = z[:, n].copy()
                q_count_stat_funcs[:, n] = q_count_stat_funcs_n
                f_count_stat_funcs[:, n] = f_count_stat_funcs_n
                g_count_stat_funcs[:, n] = g_count_stat_funcs_n
                w_cum = w_cum * np.array(w_n)  # update cumulative weights
                w_cum = w_cum / np.sum(w_cum)  # normalize cumulative weights

            # decode j with Cat(w_T)
            iw_count_stat_func = util.Categorical(self.stack.default_mprec, w_cum, use_make_sum_to_M=True)
            j, self.stack = self.pop_fun(self.stack, iw_count_stat_func)
            z_prev_j = z_prev[j]  # the ancestral lineage of the j_th particle
            Bj = []  # the index of the ancestral lineage of the j_th particle

            # APPEND STEP
            for n in reversed(range(sym_length)):
                if n == (sym_length - 1):
                    Bj_n = j  # the ancestral index of the j_th particle at step n
                else:
                    Bj_n = A[Bj_n, n]  # update rule: B_n^j = A_n^(B_n+1^j)
                Bj.insert(0, Bj_n)

                for i in range(self.num_particles):
                    if i != Bj_n:
                        # encode z_n^i with q(z_n^i | x_n, z_1:n-1^i)
                        self.stack = self.append(z[i, n], self.stack, q_count_stat_funcs[i, n])

                if n != 0 and resample_decision[n - 1]:
                    for i in range(self.num_particles):
                        if i != Bj_n:
                            # encode A_n-1^i with Cat(w_n-1)
                            self.stack = self.append_fun(A[i, n - 1], self.stack, iw_count_stat_funcs[n - 1])

                # encode x_n with g(x_n | z_n^(B_n^j))
                assert np.all(z[Bj_n, n] == z_prev_j[n])  # sanity check
                self.stack = self.append(sym[n], self.stack, g_count_stat_funcs[Bj_n, n])

                # encode z_n^(B_n^j) with f(z_n^(B_n^j) | z_n-1^(B_n-1^j))
                self.stack = self.append(z_prev_j[n], self.stack, f_count_stat_funcs[Bj_n, n])

                if n == 0 or resample_decision[n - 1]:
                    # encode B_n^j = A_n^(B_n+1^j) with Cat(1/N)
                    # always encode B_1^j with Cat(1/N)
                    self.stack = self.append_fun(Bj_n, self.stack, self.iw_uniform_count_stat_func)

            total_encoded_symlens += sym_length
            self.num_encoded += 1
            if print_progress and self.num_encoded % PRINT_FREQ == 0:
                print("\r..encoded {}/{} sequences, totalling {}/{} encoded symbols, net length: {:.3f} bits/sym, "
                      "total length {:.3f} bits/sym, encoding speed {:.3f}s/sym"
                      .format(self.num_encoded, len(seq),
                              total_encoded_symlens, total_symlens,
                              self.net_bit_length / total_encoded_symlens,
                              self.bit_length / total_encoded_symlens,
                              (time.time() - start_time) / total_encoded_symlens))

    def decode(self, num, sym_lengths=None, print_progress=False):
        """Encoding a message (a sequence of `symbols`) and each `symbol` is a sequence of symbols"""
        # `num` should equal to the length of the encoded sequence
        # `sym_lengths` is the length for each symbol which should be of shape (num,). If not provided, the default
        # length is used instead

        if print_progress:
            print("..Start decoding")
        start_time = time.time()

        seq = []
        total_decoded_symlens = 0
        if sym_lengths is not None:
            total_symlens = sum(sym_lengths)
        else:
            total_symlens = num * self.default_symlen

        while num > 0:
            if sym_lengths is not None:
                sym_length = int(sym_lengths[num - 1])
            else:
                sym_length = self.default_symlen
            sym = []
            z = np.empty((self.num_particles, sym_length),
                         dtype=object)  # simulated particles, total size will be N x T
            z_prev = np.empty((self.num_particles, sym_length),
                              dtype=object)  # ancestral lineage of each particle, total size will be N x T
            z_states = np.empty((self.num_particles,), dtype=object)  # current particle states, `None`s if not needed

            A = np.zeros((self.num_particles, sym_length - 1),
                         dtype=np.int)  # resampling parent indices, total size will be N x (T - 1)

            # Similar to encoding, we store the stat funcs in POP step
            q_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)
            f_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)
            g_count_stat_funcs = np.empty((self.num_particles, sym_length), dtype=object)
            iw_count_stat_funcs = np.empty((sym_length - 1,), dtype=object)

            # cumulative importance weights
            w_cum = np.ones((self.num_particles,), np.float) / self.num_particles
            # resample decisions used for adaptive resampling
            resample_decision = np.zeros((sym_length - 1,), dtype=bool)

            # POP step
            z_prev_j = []
            Bj = []
            for n in range(sym_length):
                particle_states_n = []
                if n == 0:
                    # decode B_1^j with Cat(1/N)
                    Bj_n, self.stack = self.pop_fun(self.stack, self.iw_uniform_count_stat_func)
                    state_j_n = self.init_state()
                    latent_count_stat_func_j = self.get_trans_count_stat_func(None,
                                                                              state_j_n)  # p(z_0^i)
                else:
                    do_resample = self.resample and (
                            (not self.adaptive) or (self.adaptive and self.resample_crit(w_cum)))

                    # if resampled, decode B_n^j with Cat(1/N)
                    if do_resample:
                        Bj_n, self.stack = self.pop_fun(self.stack, self.iw_uniform_count_stat_func)
                    # else copy Bj_n from the last step

                    state_j_n = z_states[Bj[n - 1]]
                    latent_count_stat_func_j = self.get_trans_count_stat_func(z_prev_j[-1],
                                                                              state_j_n)  # p(z_n^i | z_1:n-1^i)

                Bj.append(Bj_n)
                # decode z_n^(B_n^j) with f(z_n^(B_n^j) | z_n-1^(B_n-1^j))
                z_prev_j_n, self.stack = self.pop(self.stack, latent_count_stat_func_j)
                z_prev_j.append(z_prev_j_n)

                # decode x_n with g(x_n | z_n^(B_n^j))
                cond_count_stat_func_j = self.get_cond_count_stat_func(z_prev_j_n, state_j_n)
                x_n, self.stack = self.pop(self.stack, cond_count_stat_func_j)
                sym.append(x_n)

                prop_count_stat_func_j = self.get_prop_count_stat_func(x_n, state_j_n)  # q(z_1 | x_1)
                next_state_j_n = self.update_state(state_j_n, x_n, z_prev_j_n)

                if n != 0:
                    if do_resample:
                        A_n = []
                        iw_count_stat_func = util.Categorical(self.stack.default_mprec, w_cum, use_make_sum_to_M=True)
                        for i in range(self.num_particles):
                            if i != Bj_n:
                                # decode A_n-1^i with Cat(w_n-1)
                                parent, self.stack = self.pop_fun(self.stack, iw_count_stat_func)
                                A_n.insert(0, parent)
                        A_n.insert(Bj[n], Bj[n - 1])  # insert the ancestral index for the j_th particle
                        A_n = np.array(A_n)
                        z_prev = z_prev[A_n]  # re-assign the ancestral particles
                        z_states = z_states[A_n]  # re-assign the particle states
                        A[:, n - 1] = A_n
                        w_cum = np.ones((self.num_particles,), np.float) / self.num_particles  # reset weights
                        resample_decision[n - 1] = True
                        iw_count_stat_funcs[n - 1] = iw_count_stat_func
                    else:
                        A_n = np.arange(self.num_particles)
                        A[:, n - 1] = A_n
                        resample_decision[n - 1] = False

                z_n = []
                q_count_stat_funcs_n = []
                f_count_stat_funcs_n = []
                g_count_stat_funcs_n = []

                for i in reversed(range(self.num_particles)):
                    if i != Bj_n:
                        if n == 0:
                            cur_state = self.init_state()  # get inital state h_0
                            latent_count_stat_func = self.get_trans_count_stat_func(None,
                                                                                    cur_state)  # p(z_0^i)

                        else:
                            cur_state = z_states[i]
                            latent_count_stat_func = self.get_trans_count_stat_func(z_prev[i, n - 1],
                                                                                    cur_state)  # p(z_n^i | z_1:n-1^i)

                        prop_count_stat_func = self.get_prop_count_stat_func(x_n,
                                                                             cur_state)  # q(z_n^i | x_n, z_1:n-1^i)

                        # decode z_n^i ~ q(z_n | x_n, z_1:n-1^i) i.i.d. for each particle
                        latent, self.stack = self.pop(self.stack, prop_count_stat_func)
                        cond_count_stat_func = self.get_cond_count_stat_func(latent, cur_state)
                        next_state = self.update_state(cur_state, x_n, latent)
                        z_n.insert(0, latent)
                        q_count_stat_funcs_n.insert(0, prop_count_stat_func)
                        f_count_stat_funcs_n.insert(0, latent_count_stat_func)
                        g_count_stat_funcs_n.insert(0, cond_count_stat_func)
                        particle_states_n.insert(0, next_state)

                z_n.insert(Bj_n, z_prev_j_n)  # insert the j_th particle

                # special particle
                particle_states_n.insert(Bj_n, next_state_j_n)

                q_count_stat_funcs_n.insert(Bj_n, prop_count_stat_func_j)
                f_count_stat_funcs_n.insert(Bj_n, latent_count_stat_func_j)
                g_count_stat_funcs_n.insert(Bj_n, cond_count_stat_func_j)
                w_n, _ = self.importance_weights(x_n, z_n, q_count_stat_funcs_n, f_count_stat_funcs_n,
                                                 g_count_stat_funcs_n, weights=w_cum)  # compute importance weights
                z[:, n] = z_n
                z_prev[:, n] = z[:, n].copy()
                z_states[:] = particle_states_n

                q_count_stat_funcs[:, n] = q_count_stat_funcs_n
                f_count_stat_funcs[:, n] = f_count_stat_funcs_n  # not necessary
                g_count_stat_funcs[:, n] = g_count_stat_funcs_n  # not necessary
                w_cum = w_cum * np.array(w_n)
                w_cum = w_cum / np.sum(w_cum)  # normalize

            # APPEND STEP
            # Encode j with Cat(w_T)
            iw_count_stat_func = util.Categorical(self.stack.default_mprec, w_cum, use_make_sum_to_M=True)
            self.stack = self.append_fun(Bj_n, self.stack, iw_count_stat_func)

            for n in reversed(range(sym_length)):
                for i in reversed(range(self.num_particles)):
                    # encode z_n^i with q(z_n^i | x_n, z_1:n-1^i)
                    self.stack = self.append(z[i, n], self.stack, q_count_stat_funcs[i, n])

                if n != 0 and resample_decision[n - 1]:
                    for i in reversed(range(self.num_particles)):
                        # encode A_n-1^i with Cat(w_n-1)
                        self.stack = self.append_fun(A[i, n - 1], self.stack, iw_count_stat_funcs[n - 1])

            seq.insert(0, sym)
            num -= 1
            self.num_encoded -= 1
            total_decoded_symlens += sym_length
            if print_progress and self.num_encoded % PRINT_FREQ == 0:
                print("\r..decoded {}/{} sequences, totalling {}/{} decoded symbols, decoding speed {:.3f}s/sym"
                      .format(self.num_encoded, len(seq),
                              total_decoded_symlens, total_symlens,
                              (time.time() - start_time) / total_decoded_symlens))
        return seq
