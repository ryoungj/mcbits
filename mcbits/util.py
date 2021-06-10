from __future__ import division
from __future__ import print_function

import sys

is_py2 = sys.version[0] == '2'

import numpy as np
from scipy.stats import norm, beta, binom
from scipy.special import gammaln

if is_py2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue


# ----------------------------------------------------------------------------
# Functions for categorical and discrete Gaussian distributions
# Some are adpated from https://github.com/bits-back/bits-back
# ----------------------------------------------------------------------------

def _nearest_int(arr):
    # This will break when vectorized
    return int(np.around(arr))


std_gaussian_bucket_cache = {}  # Stores bucket endpoints
std_gaussian_centres_cache = {}  # Stores bucket centres


def std_gaussian_buckets(precision):
    """
    Return the endpoints of buckets partitioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_bucket_cache:
        return std_gaussian_bucket_cache[precision]
    else:
        cumsum_range = np.arange((1 << precision) + 1) / (1 << precision)
        buckets = norm.ppf(cumsum_range)
        std_gaussian_bucket_cache[precision] = buckets
        return buckets


def std_gaussian_centres(precision):
    """
    Return the centres of mass of buckets partitioning the domain of the prior.
    Each bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_centres_cache:
        return std_gaussian_centres_cache[precision]
    else:
        cumsum_range = (np.arange((1 << precision)) + 0.5) / (1 << precision)
        centres = norm.ppf(cumsum_range)
        std_gaussian_centres_cache[precision] = centres
        return centres


def create_categorical_buckets(probs, precision):
    """
    Create categorical buckets with the sum approximated probability counts equal to (1 << precision)
    """
    probs = np.array(probs)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
        batch = False
    else:
        batch = True
    buckets = np.floor(probs * ((1 << precision) - probs.shape[1])) + np.ones(probs.shape)
    bucket_sum = np.sum(buckets, axis=1)

    idxs_max = np.argmax(buckets, axis=1)
    buckets[np.arange(probs.shape[0]), idxs_max] += (1 << precision) - bucket_sum
    assert (np.sum(buckets, axis=1) == 1 << precision).all()
    if not batch:
        return buckets[0], np.insert(np.cumsum(buckets[0]), 0, 0)  # this could be slightly wrong
    else:
        return buckets, np.insert(np.cumsum(buckets, axis=1), 0, 0, axis=1)  # this could be slightly wrong


def make_sum_to_M(counts, M):
    """
    Make a distribution of counts sum to M with very little error.
    Credit is to Charles Bloom:
    http://cbloomrants.blogspot.com/2014/02/02-11-14-understanding-ans-10.html
    Args
        counts: a list of ints
        M: an integer
    Returns
        mcounts: a list of ints such that sum(mcounts) = M.
    """
    T = sum(counts)
    mcounts = []
    for sym in range(len(counts)):
        from_scaled = M * counts[sym] / float(T)
        from_scaled = np.maximum(from_scaled, 1e-16)
        down = max(int(np.floor(from_scaled)), 1)
        mcounts.append(
            down if from_scaled * from_scaled <= down * (down + 1) else down + 1)

    q = PriorityQueue()
    corr = M - sum(mcounts)
    corrsign = 1 if corr > 0 else -1

    for sym in range(len(counts)):
        Cs = counts[sym]
        Fs = mcounts[sym]
        if (Cs > 0) and ((Fs > 1) or (corrsign == 1)):
            Ls_delta = Cs * np.log2(Fs / (Fs + corrsign))
            q.put((Ls_delta, sym))

    while corr != 0:
        _, sym = q.get()

        mcounts[sym] += corrsign
        corr -= corrsign
        Fs = mcounts[sym]
        Cs = counts[sym]
        if (Fs > 1) or (corrsign == 1):
            Ls_delta = Cs * np.log2(Fs / (Fs + corrsign))
            q.put((Ls_delta, sym))

    return mcounts


# ----------------------------------------------------------------------------
# Count Stat Func Classes for Discrete Distribution
# ----------------------------------------------------------------------------


class CountStatFunc(object):
    """The base class of statistical functions of scaler-valued discrete distributions"""

    def __init__(self, precision):
        self.precision = precision

    @property
    def cdf(self):
        """The discretized cdf function: bucket index -> cdf count in the range [0, 1 << precision]"""
        raise NotImplementedError

    @property
    def ppf(self):
        """The discretized ppf (inverse cdf) function: cdf count in the range [0, 1 << precision] -> bucket index"""
        raise NotImplementedError

    @property
    def count(self):
        """The discretized probability count"""
        raise NotImplementedError

    @property
    def stat_func(self):
        """The statistical functions of the discrete distribution (cdf, ppf)"""
        return NotImplementedError

    def get_log_prob(self, idx):
        """Compute the log discrete probability"""
        raise NotImplementedError

    def count2stat_func(self, count):
        """Convert probability counts to statistical functions (cdf, pdf)"""

        assert sum(count) == (1 << self.precision), "Counts must be pre-computed and sum to 2^mprec"

        cumcount = np.insert(np.cumsum(count), 0, 0)  # this could be slightly wrong

        def cdf(s):
            return int(cumcount[s])

        def ppf(cf):
            return np.searchsorted(cumcount, cf, 'right') - 1

        return cdf, ppf

    def stat_func2count(self, stat_func):
        """Convert statistical functions (cdf, pdf) to probability counts"""

        cdf, _ = stat_func

        cumcount = []
        i = 0
        while True:
            try:
                c = cdf(i)
                cumcount.append(c)
                i += 1
            except:
                break

        count = [end - beg for end, beg in zip(cumcount[1:], cumcount[:-1])]
        assert sum(count) == (1 << self.precision), sum(count)

        return count


class DiscreteGaussian(CountStatFunc):
    """The Discrete Gaussian whose buckets have the equal mass under the standard Gaussian prior"""

    def __init__(self, precision, log_num_bucket, mean, stdd):
        """Initialize a DiscreteGaussian Distribution

        Args:
            precision: the precision for approximating the probability counts
            log_num_bucket: the log of the number of buckets
            mean: the mean of the Gaussian distribution
            stdd: the std of the Gaussian distribution
        """
        super(DiscreteGaussian, self).__init__(precision)
        self.gaussian_buckets = std_gaussian_buckets(log_num_bucket)
        self.mean = mean
        self.stdd = stdd

    @property
    def cdf(self):
        def gaussian_cdf(idx):
            x = self.gaussian_buckets[idx]
            return _nearest_int(norm.cdf(x, self.mean, self.stdd) * (1 << self.precision))

        return gaussian_cdf

    @property
    def ppf(self):
        def gaussian_ppf(cf):
            x = norm.ppf((cf + 0.5) / (1 << self.precision), self.mean, self.stdd)
            return np.searchsorted(self.gaussian_buckets, x, 'right') - 1

        return gaussian_ppf

    @property
    def stat_func(self):
        return (self.cdf, self.ppf)

    @property
    def count(self):
        cumcount = [_nearest_int(norm.cdf(x, self.mean, self.stdd) * (1 << self.precision)) for x in
                    self.gaussian_buckets]
        count = np.array(cumcount[1:]) - np.array(cumcount[:-1])

        return count

    def get_log_prob(self, idx):
        # use stat func is computationally cheaper for discretized Gaussian
        count = self.cdf(idx + 1) - self.cdf(idx)
        return np.log(count / (1 << self.precision) + 1e-16)


class QuantizedGaussian(CountStatFunc):
    """The Quantized Gaussian which is uniformly quantized in the latent space"""

    def __init__(self, precision, mean, stdd, interval=1.0):
        """Initialize a QuantizedGaussian Distribution
        
        Args:
            precision: the precision for approximating the probability counts
            mean: the mean of the Gaussian distribution
            stdd: the std of the Gaussian distribution
            interval: the quantization interval
        """
        super(QuantizedGaussian, self).__init__(precision)
        self.mean = mean
        self.stdd = stdd
        self.interval = interval

    @property
    def cdf(self):
        def gaussian_cdf(idx):
            return _nearest_int(norm.cdf((idx - 0.5) * self.interval, self.mean, self.stdd) * (1 << self.precision))

        return gaussian_cdf

    @property
    def ppf(self):
        def gaussian_ppf(cf):
            x = norm.ppf((cf + 0.5) / (1 << self.precision), self.mean, self.stdd)
            return _nearest_int(x / self.interval)

        return gaussian_ppf

    @property
    def stat_func(self):
        return (self.cdf, self.ppf)

    @property
    def count(self):
        raise NotImplementedError

    def get_log_prob(self, idx):
        # use stat func is computationally cheaper for discretized Gaussian
        count = self.cdf(idx + 1) - self.cdf(idx)
        return np.log(count / (1 << self.precision) + 1e-16)


class Categorical(CountStatFunc):
    """The Categorical distribution"""

    def __init__(self, precision, prob, use_make_sum_to_M=False, count_bucket=None, cumulative_bucket=None):
        """Initialize the  Categorical distribution

        Args:
            precision: the precision for approximating the probability counts
            prob: the original probability counts (sum to 1)
            use_make_sum_to_M: whether use the `make_sum_to_M` for approximating the probability count
            count_bucket: the pre-computed count bucket
            cumulative_bucket: the pre-computed cumulative count bucket
        """
        super(Categorical, self).__init__(precision)
        self.prob = prob

        if (count_bucket is not None) and (cumulative_bucket is not None):
            self.count_bucket = count_bucket
            self.cumulative_bucket = cumulative_bucket
        else:
            if use_make_sum_to_M:
                self.count_bucket = make_sum_to_M(self.prob, 1 << precision)
                self.cumulative_bucket = np.cumsum([0] + self.count_bucket)
            else:
                self.count_bucket, self.cumulative_bucket = create_categorical_buckets(self.prob, precision)

    @property
    def cdf(self):
        def cat_cdf(idx):
            return int(self.cumulative_bucket[idx])

        return cat_cdf

    @property
    def ppf(self):
        def cat_ppf(cf):
            return np.searchsorted(self.cumulative_bucket, cf, 'right') - 1

        return cat_ppf

    @property
    def stat_func(self):
        return (self.cdf, self.ppf)

    @property
    def count(self):
        return self.count_bucket

    def get_log_prob(self, idx):
        # use count is computationally cheaper for Categorical
        count = self.count[idx]
        return np.log(count / (1 << self.precision) + 1e-16)


class CountStatFuncMulti(CountStatFunc):
    """The base class of statistical functions of vector-valued (multi-dim) discrete distributions"""

    def __init__(self, dim, precision):
        super(CountStatFuncMulti, self).__init__(precision)
        self.dim = dim
        self.precision = precision
        self.count_stat_funcs = []

    @property
    def cdf(self):
        return [count_stat_func.cdf for count_stat_func in self.count_stat_funcs]

    @property
    def ppf(self):
        return [count_stat_func.ppf for count_stat_func in self.count_stat_funcs]

    @property
    def stat_func(self):
        return [count_stat_func.stat_func for count_stat_func in self.count_stat_funcs]

    @property
    def count(self):
        return [count_stat_func.count for count_stat_func in self.count_stat_funcs]


class DiscreteGaussianMulti(CountStatFuncMulti):
    """The multi-dim Discrete Gaussian"""

    def __init__(self, dim, precision, log_num_bucket, means, stdds):
        """Initialize the multi-dim Discrete Gaussian distribution

        Args:
            dim: the dimension of the distribution
            precision: the precision for approximating the probability counts
            log_num_bucket: the log of the number of buckets
            means: the means of the multi-dim Gaussian distribution
            stdds: the stds of the multi-dim Gaussian distribution
        """
        super(DiscreteGaussianMulti, self).__init__(dim, precision)
        self.means = means
        self.stdds = stdds
        self.log_num_bucket = log_num_bucket
        assert len(self.means) == self.dim, (len(self.means), self.dim)
        assert len(self.stdds) == self.dim, (len(self.stdds), self.dim)
        self.count_stat_funcs = [DiscreteGaussian(precision, log_num_bucket, self.means[i], self.stdds[i])
                                 for i in range(self.dim)]

    def get_log_prob(self, idxs):
        # vectorize the log prob computation
        idxs = np.array(idxs).astype(np.int)
        xs = std_gaussian_buckets(self.log_num_bucket)[idxs]
        next_xs = std_gaussian_buckets(self.log_num_bucket)[idxs + 1]
        cumcount = np.around(norm.cdf(xs, self.means, self.stdds) * (1 << self.precision)).astype(int)
        next_cumcount = np.around(norm.cdf(next_xs, self.means, self.stdds) * (1 << self.precision)).astype(int)

        counts = next_cumcount - cumcount
        return np.sum(np.log(counts / (1 << self.precision) + 1e-16))


class QuantizedGaussianMulti(CountStatFuncMulti):
    """The multi-dim Quantized Gaussian"""

    def __init__(self, dim, precision, means, stdds, interval=1.0):
        """Initialize the multi-dim Quantized Gaussian

        Args:
            dim: the dimension of the distribution
            precision: the precision for approximating the probability counts
            means: the means of the multi-dim Gaussian distribution
            stdds: the stds of the multi-dim Gaussian distribution
            interval: the quantization interval
        """
        super(QuantizedGaussianMulti, self).__init__(dim, precision)
        self.means = means
        self.stdds = stdds
        self.interval = interval
        assert len(self.means) == self.dim, (len(self.means), self.dim)
        assert len(self.stdds) == self.dim, (len(self.stdds), self.dim)
        self.count_stat_funcs = [QuantizedGaussian(precision, self.means[i], self.stdds[i], self.interval)
                                 for i in range(self.dim)]

    def get_log_prob(self, idxs):
        # vectorize the log prob computation
        idxs = np.array(idxs).astype(np.int)
        cumcount = np.around(
            norm.cdf((idxs - 0.5) * self.interval, self.means, self.stdds) * (1 << self.precision)).astype(int)
        next_cumcount = np.around(
            norm.cdf((idxs + 0.5) * self.interval, self.means, self.stdds) * (1 << self.precision)).astype(int)

        counts = next_cumcount - cumcount
        return np.sum(np.log(counts / (1 << self.precision) + 1e-16))


class CategoricalMulti(CountStatFuncMulti):
    """The multi-dim Categorical distribution"""

    def __init__(self, dim, precision, probs, use_make_sum_to_M=False):
        """ Initialize the multi-dim Categorical distribution

        Args:
            dim: the dimension of the distribution
            precision: the precision for approximating the probability counts
            probs: the original probability counts (sum to 1)
            use_make_sum_to_M: whether use the `make_sum_to_M` for approximating the probability count
        """
        super(CategoricalMulti, self).__init__(dim, precision)
        self.probs = probs
        self.count_stat_funcs = []

        assert len(self.probs) == self.dim, (len(self.probs), self.dim)

        self.count_buckets, self.cumulative_buckets = create_categorical_buckets(self.probs, precision)
        for i in range(self.dim):
            self.count_stat_funcs.append(Categorical(precision, probs[i], count_bucket=self.count_buckets[i],
                                                     cumulative_bucket=self.cumulative_buckets[i]))

    def get_log_prob(self, idxs):
        # vectorize the log prob computation
        idxs = np.array(idxs).astype(np.int)
        counts = self.count_buckets[np.arange(self.dim), idxs]
        return np.sum(np.log(counts / (1 << self.precision) + 1e-16))
