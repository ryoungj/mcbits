"""
Closely based on https://fgiesen.wordpress.com/2014/02/02/rans-notes/ by
Fabian Giesen.

NOTE:

(a << b) is the same as a * (2 ** b)
(a >> b) is the same as a / (2 ** b)
(a % (2 ** b)) is the same as (a & ((1 << b) - 1))

These are just fast bit-wise manipulations.
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import bisect
import math


def get_sym(b, c):
    """
    Find the index i in cumcounts such that b[i] <= c < b[i+1], unless
    c > b[-1], in which case return len(b) - 1.
    Args:
        b : sorted, increasing list of ints
        c : int
    Returns
        i : index in b
    """
    i = bisect.bisect_right(b, c)
    return i - 1


def get_count_by_cdf(sym, cdf):
    start = cdf(sym)
    count = cdf(sym + 1) - start
    return start, count

def encode(x, count, cumcount, prec):
    """
    Implements the encode function of rANS, as discussed by Fabian Giesen.
    C(x, s) = floor(x / count) * 2^prec + cumcount + (x % count)
    Ideally:
    for s in seq:
        x = encode(x, s)
    for i in range(len(seq)):
        s, x = decode(x)
    Args:
        x : int representing the state of the ANS encoder
        sym : int representing the symbol to encode into x
        count: int representing the unnorm. prob. of sym
        cumcount: cumsum of all counts for symbols s s.t. s < sym
        prec: sum(counts) = 2^prec
    Returns:
        nextx : x with sym encoded into it
    """
    return (int(x // count) << prec) + cumcount + (x % count)


def decode_by_counts(x, counts, cumcounts, prec):
    """
    Implements the decode function of rANS, as discussed by Fabian Giesen.
    D(x) = (floor(x / 2^prec) * counts[s] + (x % 2^prec) - cumcounts[s], s)
    where s is such that cumcounts[s] <= x % 2^prec < cumcounts[s] + counts[s]
    Assumption: sum(counts) = 2^prec
                cumcounts = np.cumsum([0] + counts)[:-1].tolist()
    Args:
        x : int representing the state of the ANS encoder
        counts : list of ints representing the unnorm. probs.
        cumcounts : cumsum of counts starting at 0, i.e.,
                    cumcounts = np.cumsum([0] + counts)[:-1].tolist()
        prec: sum(counts) = 2^prec
    Returns:
        x : x with sym decoded from it
        s : the decoded sym
    """

    y = x & ((1 << prec) - 1)
    sym = get_sym(cumcounts, y)
    x = counts[sym] * (x >> prec) + y - cumcounts[sym]
    return (x, sym)


def decode_by_statfunc(x, cdf, ppf, prec):
    """
    Implements the decode function of rANS, using stat func.
    Assumption: cdf has already been scaled to 2^prec
    Args:
        x : int representing the state of the ANS encoder
        cdf : cumulative count function where cdf[idx] = sum(pdf[:idx])
        ppf: probability (count) percentile function where ppf[state] returns the index idx such that
             cdf[idx] <= state < cdf[idx]
        prec: sum(counts) = 2^prec
    Returns:
        x : x with sym decoded from it
        s : the decoded sym
    """

    y = x & ((1 << prec) - 1)
    sym = ppf(y)
    cumcount, count = get_count_by_cdf(sym, cdf)
    x = count * (x >> prec) + y - cumcount
    return (x, sym)


class XStack(object):
    """This is a stack for ints that starts returning random ints in the
    range [0, 2 ** bprec) once it is empty."""

    def __init__(self, bprec=32, initial_xstack=None):
        self.bprec = bprec  # <- precision of each emitted integer
        self.initial_xstack = initial_xstack
        if initial_xstack is None:
            self.stack = []
        else:
            self.stack = initial_xstack
        self.empty_pop = 0  # to measure the initial extra bits used in BitsBack

        # assert bprec in [8, 16, 32, 64]

        # debug bits
        # self.stack = (np.array(
        #        [1182768797, 1702838126, 391580084,
        #         1169432897, 1214198040, 1839743260, 1606532335,
        #         1029170860]).tolist())[::-1]

    def append(self, x):
        self.stack.append(x)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            # todo: can we do this smarter ??
            self.empty_pop += 1
            return int(np.random.randint(1 << self.bprec))

    def is_empty(self):
        return len(self.stack) == 0

    def __len__(self):
        return len(self.stack)

    def reset(self):
        if self.initial_xstack is None:
            self.stack = []
        else:
            self.stack = self.initial_xstack
        self.empty_pop = 0


class RANSStack(object):

    def __init__(self, counts=None, stat_func=None,
                 use_statfunc=True, mprec=8, lprec=32, bprec=32, initial_x=None, initial_xstack=None):
        """
        An rANS Stack keeping its state inside the interval
            x in [L, b*L) = {L, L+1, ..., b*L - 1}

        Args:
            counts: list of counts

        In Frank's notation:
        M := (1 << mprec) # mprec <- precision of the distribution
        b := (1 << bprec) # bprec <- precision of each emitted integer
        L := (1 << lprec) # <- lower bound on x
        Require bprec > 0, lprec >= mprec

        In a perfect world, this stack would simple keep track of the integer x
        and apply encode encode encode, then decode decode decode. But, x will
        quickly grow without bound, so this wrapper ensures that x remains in
        the interval [L, b*L) while emitting just the necessary bits to reconstruct
        the whole path.
        """
        assert bprec > 0
        assert lprec >= mprec
        # assert bprec in [1, 8, 16, 32]  # not necessary
        # assert lprec + bprec in [8, 16, 32, 64]  # not necessary

        self.initial_x = initial_x
        if self.initial_x is None:
            self.x = (1 << lprec)  # self.x is our state, it is an int
            self.initial_x = (1 << lprec)
            # self.x = 7713286347646032629 # debug bits
        else:
            assert (1 << lprec) <= self.initial_x < (1 << (lprec + bprec))
            self.x = self.initial_x

        # These are bits we "store" to keep self.x in [L, b*L)
        if initial_xstack is not None:
            self.initial_xstack = initial_xstack.copy()
        else:
            self.initial_xstack = []
        self.xstack = XStack(bprec, initial_xstack)
        self.default_mprec = mprec
        self.lprec = lprec
        self.bprec = bprec
        self.num = 0

        if counts:
            self.update_counts(counts)
        else:
            self.counts = None

        if stat_func:
            self.cdf, self.ppf = stat_func
        else:
            self.cdf, self.ppf = None, None

        self.use_statfunc = use_statfunc

    def update_count_stat_func(self, count_stat_func, mprec=None):
        self.mprec = mprec if mprec is not None else self.default_mprec

        if self.use_statfunc:
            self.update_statfunc(count_stat_func.stat_func)
        else:
            self.update_counts(count_stat_func.count)

    def update_statfunc(self, stat_func):
        self.cdf, self.ppf = stat_func

    def update_counts(self, counts):
        """Changes the encoding / decoding distribution to counts."""
        self.counts = [int(count) for count in counts]
        assert sum(self.counts) == (1 << self.mprec), "The counts are not discretized!"
        self.cumcounts = np.cumsum([0] + self.counts)[:-1].tolist()

    def get_count(self, sym):
        if self.use_statfunc:
            if self.cdf is None and self.ppf is None:
                raise Exception("Need to specify stat_func when `use_statfunc`=True!")

            cumcount, count = get_count_by_cdf(sym, self.cdf)
        else:
            if self.counts is None:
                raise Exception("Need to specify counts when `use_statfunc`=False!")

            cumcount, count = self.cumcounts[sym], self.counts[sym]

        return cumcount, count

    def append(self, sym):
        """Push sym onto the stack. Sym is distributed according
        to counts and cumcounts, which are such that sum(counts) = 2^prec."""
        assert (1 << self.lprec) <= self.x < (1 << (self.bprec + self.lprec))

        cumcount, count = self.get_count(sym)

        assert count != 0, "Count should not be zero"

        x_max = count << (self.lprec - self.mprec + self.bprec)
        while self.x >= x_max:
            self.xstack.append(self.x & ((1 << self.bprec) - 1))  # (x mod 2**bprec)
            self.x = (self.x >> self.bprec)  # (x / 2**bprec)
            # bprec = 1
            # x = 0111
            # put 0 onto our stack
            # set x = 111
        self.x = encode(self.x, count, cumcount, self.mprec)

    def pop(self):
        """Pop sym from the stack. Sym is distributed according
        to counts and cumcounts, which are such that sum(counts) = 2^prec."""
        assert (1 << self.lprec) <= self.x < (1 << (self.bprec + self.lprec))

        if self.use_statfunc:
            if self.cdf is None and self.ppf is None:
                raise Exception("Need to specify stat_func when `use_statfunc`=True!")

            self.x, sym = decode_by_statfunc(self.x, self.cdf, self.ppf, self.mprec)
        else:
            if self.counts is None:
                raise Exception("Need to specify counts when `use_statfunc`=False!")

            self.x, sym = decode_by_counts(self.x, self.counts, self.cumcounts, self.mprec)

        while self.x < (1 << self.lprec):  # if x < L, want x >= L = 2 ** lprec
            nextx = self.xstack.pop()  # remove from xstack
            self.x = (self.x << self.bprec) + nextx  # put them back into x
            # so that x >= L, nextx has bprec bits.
            # e.g. x = (001010110101110)_b, let's say bprec = 4
            # x -> (0000001010110101110)_b + (1100)_b, where (1100)_b is nextx
            # x = (1100001010110101110)_b

        return sym

    @property
    def bit_length(self):
        return self.bprec * len(self.xstack) + np.log2(float(self.x))
        # -> entropy * num_symbols if the sequence of symbols are iid from counts.

    @property
    def net_bit_length(self):
        # To study the cleaness issue, we might want to use net bit length instead of bit length
        return np.log2(float(self.x)) + self.bprec * (len(self.xstack) - self.xstack.empty_pop - len(self.initial_xstack)) \
               - np.log2(float(self.initial_x))
        # -> assume we have enough extra information to send,
        #    we can also use these bits for encoding the first symbol and have bits back

    @property
    def net_bit_length_townsend(self):
        return self.bprec * (len(self.xstack) - len(self.initial_xstack) - self.xstack.empty_pop)

    def reset(self):
        if self.initial_x is None:
            self.x = (1 << self.lprec)
        else:
            self.x = self.initial_x
        self.xstack.reset()
