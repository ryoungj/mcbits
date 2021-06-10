import sys
from mcbits import coders, util
import numpy as np
import collections
import random
from models.toy_model import ToyModel
from models.toy_hmm_model import HMMToyModel
from matplotlib import pyplot as plt
from numpy.random import randint, choice
import time


def test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message):
    message_length = len(seq)

    print(print_message)
    coder = coder_class(**coder_kwargs)
    initial_state = coder.stack.x
    start_time = time.time()
    coder.encode(seq)
    print(f"\tentropy : {entropy}")
    print(f"\tideal bit length per symb : {ideal_bound}")
    print(f"\tnet bit length per symb : {coder.net_bit_length / message_length}")
    print(f"\ttotal bit length per symb : {coder.bit_length / message_length}")
    print(f"\tencoding time per symb: {(time.time() - start_time) / message_length}")
    outseq = coder.decode(message_length)
    assert np.all(np.equal(seq, outseq))
    assert coder.stack.x == initial_state
    print("\tencode/decode successful")


def run_toy_mixture_model(alphabet_size=16, latent_alphabet_size=8, message_length=1000, lprec=32, bprec=32, mprec=16,
                          num_particles=16, ais_steps=16):
    """Test coders on the toy mixture model"""

    model = ToyModel(latent_alphabet_size, alphabet_size)
    entropy = model.entropy
    seq = model.sample_message(message_length)

    coder_kwargs = {"mprec": mprec, "lprec": lprec, "bprec": bprec, "multidim": False, "use_statfunc": False}

    # test simple coder
    simple_coder_kwargs = dict(**coder_kwargs, count_stat_func=util.Categorical(mprec, model.x_probs,
                                                                                use_make_sum_to_M=True))
    coder_class = coders.SimpleCoder
    print_message = f"Test {coder_class.__name__} Encode/Decode"
    ideal_bound = model.entropy
    test_encode_decode(seq, coder_class, simple_coder_kwargs, entropy, ideal_bound, print_message)

    # prepare for latent variable coders
    # prop_counts = np.tile(model.z_probs, (1, alphabet_size))  # use prior as proposal
    prop_counts = (np.ones((latent_alphabet_size, alphabet_size)) / latent_alphabet_size)  # use uniform proposal

    coder_kwargs.update({
        "get_prior_count_stat_func": lambda: util.Categorical(mprec, model.z_probs, use_make_sum_to_M=True),
        "get_prop_count_stat_func": lambda x: util.Categorical(mprec, prop_counts[:, x], use_make_sum_to_M=True),
        "get_cond_count_stat_func": lambda z: util.Categorical(mprec, model.cond_probs[z], use_make_sum_to_M=True),
    })

    # test naive coder
    coder_class = coders.StochasticCoder
    print_message = f"Test {coder_class.__name__} Encode/Decode"
    ideal_bound = model.naive_code(prop_counts)
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)

    # test bits-back coder
    coder_class = coders.BitsBackCoder
    print_message = f"Test {coder_class.__name__} Encode/Decode"
    ideal_bound = model.elbo_code(prop_counts)
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)

    # test is bits-back coder
    coder_class = coders.ISBitsBackCoder
    is_coder_kwargs = dict(**coder_kwargs, num_particles=num_particles)
    print_message = f"Test {coder_class.__name__} Encode/Decode, Num Particles = {num_particles}"
    ideal_bound = model.is_code(prop_counts, num_particles=num_particles)
    test_encode_decode(seq, coder_class, is_coder_kwargs, entropy, ideal_bound, print_message)

    # prepare for cis
    # for cis, we use shift sampling operators to generate other latents
    def shift_sampling(shift, precision):
        assert 0.0 <= shift <= 1.0
        upper = 1 << precision
        shift_scaled = int(shift * upper)

        def operator(u, prop_count_stat_func):
            return (u + shift_scaled) % upper

        def inverse_operator(u, prop_count_stat_func):
            return (u - shift_scaled) % upper

        return (operator, inverse_operator)

    sampling_shifts = np.random.rand(num_particles - 1, )
    sampling_shifts = np.insert(sampling_shifts, 0, 0.0)
    shift_sampling_operators = [shift_sampling(shift, mprec) for shift in sampling_shifts]

    # test cis bits-back coder
    coder_class = coders.CISBitsBackCoder
    cis_coder_kwargs = dict(**is_coder_kwargs, bijective_operators=shift_sampling_operators)
    print_message = f"Test {coder_class.__name__} Encode/Decode, Num Particles = {num_particles}"
    ideal_bound = model.coupled_is_code(prop_counts, sampling_shifts, num_particles=num_particles)
    test_encode_decode(seq, coder_class, cis_coder_kwargs, entropy, ideal_bound, print_message)

    # prepare for ais
    def get_joint_count_stat_func(z=None, x=None):
        if z is not None and x is None:
            return util.Categorical(mprec, model.joint_probs[z], use_make_sum_to_M=True)
        elif x is not None and z is None:
            return util.Categorical(mprec, model.joint_probs[:, x], use_make_sum_to_M=True)
        else:
            raise Exception("Bad arg to joint_count_stat_func_fn.")

    # test ais bits-back coder
    coder_class = coders.AISBitsBackCoder
    ais_coder_kwargs = dict(**coder_kwargs, betas=np.flip(np.linspace(0, 1, ais_steps + 1)),
                            get_joint_count_stat_func=get_joint_count_stat_func)
    print_message = f"Test {coder_class.__name__} Encode/Decode, Num AIS Steps = {ais_steps}"
    ideal_bound = model.ais_code(prop_counts, betas=ais_coder_kwargs["betas"])
    test_encode_decode(seq, coder_class, ais_coder_kwargs, entropy, ideal_bound, print_message)

    # test ais bits-back coder with bitswap
    ais_coder_kwargs.update({'bitswap': True})
    print_message = f"Test {coder_class.__name__} (BitSwap) Encode/Decode, Num AIS Steps = {ais_steps}"
    test_encode_decode(seq, coder_class, ais_coder_kwargs, entropy, ideal_bound, print_message)


def run_toy_hmm_model(alphabet_size=16, latent_alphabet_size=8, length=10, message_length=100,
                      mprec=16, lprec=32, bprec=32, num_particles=16):
    model = HMMToyModel(latent_alphabet_size, alphabet_size, length)
    seq = model.sample_message(message_length)
    log_prob = model.log_prob(seq)
    entropy = -log_prob / message_length  # use the exact neg. log prob of the message as the entropy here

    # prop_count = np.tile(model.z_probs, (1, alphabet_size))  # use prior as proposal
    prop_count = lambda x, state=None: np.ones((model.lals,)) / model.lals

    def get_trans_count_stat_func(z_prev, state=None):
        if z_prev is None:
            return util.Categorical(mprec, model.prior_probs, use_make_sum_to_M=True)
        else:
            return util.Categorical(mprec, model.transition_probs[z_prev], use_make_sum_to_M=True)

    coder_kwargs = {
        "mprec": mprec, "lprec": lprec, "bprec": bprec,
        "multidim": False, "use_statfunc": False,
        "init_state": lambda: None,
        "update_state": lambda cur_state=None, x_t=None, z_t=None: None,
        "get_prop_count_stat_func": lambda x_t, cur_state=None: util.Categorical(mprec, prop_count(x_t, cur_state),
                                                                                 use_make_sum_to_M=True),
        "get_cond_count_stat_func": lambda z_t, cur_state=None: util.Categorical(mprec, model.emission_probs[z_t],
                                                                                 use_make_sum_to_M=True),
        "get_trans_count_stat_func": get_trans_count_stat_func,
        "num_particles": num_particles,
        "default_symlen": length,
    }

    # test smc bits-back coder
    coder_class = coders.SMCBitsBackCoder
    print_message = f"Test BB-SMC Encode/Decode, Num Particles = {num_particles}"
    ideal_bound = - model.log_prob(seq, method="smc", N=num_particles, resampling=True,
                                   adaptive=False, prop_counts=prop_count) / message_length
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)

    # test smc bits-back coder (adaptive)
    coder_kwargs["adaptive"] = True
    print_message = f"Test BB-SMC (Adaptive) Encode/Decode, Num Particles = {num_particles}"
    ideal_bound = - model.log_prob(seq, method="smc", N=num_particles, resampling=True,
                                   adaptive=True, prop_counts=prop_count) / message_length
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)

    # test sis bits-back coder
    print_message = f"Test BB-SIS Encode/Decode, Num Particles = {num_particles}"
    coder_kwargs["resample"] = False
    coder_kwargs["adaptive"] = False
    ideal_bound = - model.log_prob(seq, method="smc", N=num_particles, resampling=False,
                                   adaptive=False, prop_counts=prop_count) / message_length
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)

    # test bits-back coder
    num_particles = 1
    print_message = f"Test BB-ELBO Encode/Decode, Num Particles = {num_particles}"
    coder_kwargs["resample"] = False
    coder_kwargs["adaptive"] = False
    coder_kwargs["num_particles"] = num_particles
    ideal_bound = - model.log_prob(seq, method="smc", N=num_particles, resampling=False,
                                   adaptive=False, prop_counts=prop_count) / message_length
    test_encode_decode(seq, coder_class, coder_kwargs, entropy, ideal_bound, print_message)


def main():
    np.random.seed(123)
    random.seed(123)
    run_toy_mixture_model()
    run_toy_hmm_model()


if __name__ == '__main__':
    main()
