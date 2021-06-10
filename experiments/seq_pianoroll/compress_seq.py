#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import random
import time
import tensorflow as tf

from mcbits import coders, util

# location of fivo repo, see README for more information
FIVO_CODEREPO = os.getenv('FIVO_CODEREPO')
sys.path.append(FIVO_CODEREPO)
from fivo import runners
from fivo.models.vrnn import TrainableVRNNState
from fivo.models import base
from fivo.models import vrnn
from mcbits.argsparser import str2bool, get_train_parser, get_compress_parser, get_train_args, get_compress_args

from collections import namedtuple

PureVRNNState = namedtuple("VRNNState", "rnn_state rnn_out")


def main_worker(args):
    with tf.Graph().as_default():
        print("=> Loading {}-{}-{} set from {} and construct messages...".format(args.dataset_type, args.data,
                                                                                 args.split, args.dataset_path))
        inputs, targets, seq_lengths, model, dataset_mean = runners.create_dataset_and_model(
            args, split=args.split, shuffle=False, repeat=False)

        # CREATE GRAPH
        global_step = tf.train.get_or_create_global_step()
        _init_state = model.zero_state(tf.constant(1), tf.float32)
        _cur_rnn_state_c = tf.placeholder(dtype=tf.float32, shape=(1, model.rnn_cell.state_size.c))
        _cur_rnn_state_h = tf.placeholder(dtype=tf.float32, shape=(1, model.rnn_cell.state_size.h))
        _cur_rnn_out = tf.placeholder(dtype=tf.float32, shape=(1, model.rnn_cell.output_size))
        _cur_rnn_state = tf.nn.rnn_cell.LSTMStateTuple(_cur_rnn_state_c, _cur_rnn_state_h)
        _cur_state = PureVRNNState(rnn_state=_cur_rnn_state,
                                   rnn_out=_cur_rnn_out)

        _cur_latent = tf.placeholder(dtype=tf.float32, shape=(1, args.latent_size))
        _cur_latent_encoded = model.latent_encoder(_cur_latent)
        _cur_target = tf.placeholder(dtype=tf.float32, shape=(1, args.data_dimension))
        # the mean centered target and also the next input in the original repo
        _cur_target_processed = tf.placeholder(dtype=tf.float32, shape=(1, args.data_dimension))

        # for updating the state
        _new_rnn_out, _new_rnn_state = model.run_rnn(_cur_state.rnn_state,
                                                     _cur_latent_encoded,
                                                     _cur_target_processed)
        _new_state = PureVRNNState(rnn_state=_new_rnn_state,
                                   rnn_out=_new_rnn_out)

        # for computing the stat functions
        p_zt = model.transition(_cur_state.rnn_out)
        p_zt_mean, p_zt_scale = p_zt.mean(), p_zt.scale
        q_zt = model._proposal(_cur_state.rnn_out, model.data_encoder(_cur_target), prior_mu=p_zt.mean())
        q_zt_mean, q_zt_scale = q_zt.mean(), q_zt.scale
        p_xt_given_zt, _ = model.emission(_cur_latent, _cur_state.rnn_out)

        saver = tf.train.Saver()
        with tf.train.SingularMonitoredSession() as sess:
            # RESTORE MODEL
            checkpoint = tf.train.get_checkpoint_state(args.logdir)
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            full_checkpoint_path = os.path.join(args.logdir, checkpoint_name)
            saver.restore(sess, full_checkpoint_path)
            step = sess.run(global_step)
            print("=> Model restored from {}, step {}.".format(args.logdir, step))

            # SPECIFY MESSAGE
            data_mean = sess.run(dataset_mean)

            # vrnn_inputs = []  # mean centered and shifted targets
            messages = []
            lengths = []

            while True:
                try:
                    np_out = sess.run([inputs, targets, seq_lengths])
                    messages.append(np_out[1].squeeze(1))  # original shape: (length, 1, dim)
                    lengths.append(int(np_out[2]))
                except:
                    break

            # sort the message by length to reduce the initial bit cost
            sort_idx = np.argsort(lengths)[:args.num_compress]
            messages = [messages[i].astype(int) for i in sort_idx]
            lengths = [lengths[i] for i in sort_idx]

            def get_target_input(x_t):
                '''
                Args:
                    x_t: the integer-valued symbol
                Return:
                    target_t: the target value for the VRNN model
                    input_t: the input value for the VRNN model processed from the target value
                '''
                target_t = x_t
                input_t = (x_t - data_mean)[None, :]  # mean centered target

                return (target_t, input_t)

            def get_latent(z_t):
                '''
                Args:
                    z_t: the index of the latent
                Return:
                    y_t: the true value of the latent after quantization
                '''
                y_t = util.std_gaussian_centres(args.log_num_bucket)[np.array(z_t)]

                return y_t

            # SPECIFY CODER
            # STEP 1: quantize latent and observation distribution
            # Latent cdf inputs are indices of buckets of equal width under the 'prior',
            # assumed for the purposes of bits back to be in the same family. They
            # lie in the range of ints [0, 1 << log_num_bucket).

            def init_state():
                initial_state = sess.run(_init_state)  # actually h_{-1}
                padded_input = np.zeros((1, args.data_dimension))
                new_state = sess.run(_new_state, feed_dict={_cur_rnn_state: initial_state.rnn_state,
                                                            _cur_rnn_out: initial_state.rnn_out,
                                                            _cur_target_processed: padded_input,
                                                            _cur_latent_encoded: initial_state.latent_encoded})
                return new_state

            def update_state(cur_state, x_t, z_t):
                y = get_latent(z_t)[None, :]
                _, input_t = get_target_input(x_t)
                new_state = sess.run(_new_state, feed_dict={_cur_state: cur_state,
                                                            _cur_target_processed: input_t,
                                                            _cur_latent: y})
                return new_state

            def get_trans_count_stat_func(unused_z_prev, cur_state):
                trans_mean, trans_stdd = sess.run([p_zt_mean, p_zt_scale], feed_dict={_cur_state: cur_state})
                trans_mean, trans_stdd = np.ravel(trans_mean), np.ravel(trans_stdd)
                count_stat_func = util.DiscreteGaussianMulti(args.latent_size,
                                                             args.prior_mprec,
                                                             args.log_num_bucket,
                                                             trans_mean, trans_stdd)
                return count_stat_func

            def get_prop_count_stat_func(x_t, cur_state):
                target_t, _, = get_target_input(x_t)
                post_mean, post_stdd = sess.run([q_zt_mean, q_zt_scale], feed_dict={_cur_state: cur_state,
                                                                                    _cur_target: target_t[None, :]})
                post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
                count_stat_func = util.DiscreteGaussianMulti(args.latent_size,
                                                             args.prop_mprec,
                                                             args.log_num_bucket,
                                                             post_mean, post_stdd)
                return count_stat_func

            def get_cond_count_stat_func(z_t, cur_state):
                y = get_latent(z_t)[None, :]

                probs = sess.run(p_xt_given_zt.probs, feed_dict={_cur_state: cur_state,
                                                                 _cur_latent: y})

                probs = np.stack((1. - probs, probs), axis=-1)
                counts = np.reshape(probs, (-1, np.shape(probs)[-1]))
                count_stat_func = util.CategoricalMulti(args.data_dimension, args.cond_mprec, counts)
                return count_stat_func

            # STEP 2: Built coder
            coder_kwargs = {
                # rANS params
                "lprec": args.lprec,
                "bprec": args.bprec,
                "use_statfunc": True,
                "multidim": True,
                # funcs for specifying BB-SMC coder
                "get_prior_count_stat_func": None,
                "get_cond_count_stat_func": get_cond_count_stat_func,
                "get_prop_count_stat_func": get_prop_count_stat_func,
                "get_trans_count_stat_func": get_trans_count_stat_func,
                "update_state": update_state,
                "init_state": init_state,
                # specific params for BB-SMC
                "num_particles": args.num_particles,
                "default_symlen": None,
                "resample": args.resample,
                "adaptive": args.adaptive,
            }
            assert args.coder == "SMCBitsBackCoder"
            coder = coders.__dict__[args.coder](**coder_kwargs)

            # ENCODE MESSAGE
            print("=> Encoding messages...")
            encode_start_time = time.time()
            coder.encode(messages, print_progress=True)

            # PRINT RESULT
            message_length = len(messages)
            total_symbols = sum(lengths)
            print("=> Encoded {} sequences and {} symbols in {:.2f}s".format(message_length, total_symbols,
                                                                             time.time() - encode_start_time))
            print("Net bit length:")
            print("\t{:.3f} bits".format(coder.net_bit_length))
            print("\t{:.4f} bits/seq".format(coder.net_bit_length / message_length))
            print("\t{:.4f} bits/sym".format(coder.net_bit_length / total_symbols))
            print("Total bit length:")
            print("\t{:.3f} bits".format(coder.bit_length))
            print("\t{:.4f} bits/seq".format(coder.bit_length / message_length))
            print("\t{:.4f} bits/sym".format(coder.bit_length / total_symbols))

            # DECODE MESSAGE
            if args.decode_check:
                decode_start_time = time.time()
                dec_messages = coder.decode(len(messages), lengths, print_progress=False)
                print("Decoded {} sequences and {} symbols in {:.2f}s\n"
                      "=> Decode check successful!".format(message_length,
                                                           total_symbols,
                                                           time.time() - decode_start_time))
                assert np.all([np.allclose(dec, enc) for (dec, enc) in zip(dec_messages, messages)]), \
                    "Decoded message does not match encoded message"


def main():
    np.seterr(all='raise')

    compress_parser = get_compress_parser()

    # add specific arguments
    compress_parser.add_argument(
        "--resample",
        type=str2bool,
        default=False,
        help="whether apply resampling for BB-SMC. If False, BB-SMC reduces to BB-IS.",
    )
    compress_parser.add_argument(
        "--adaptive",
        type=str2bool,
        default=False,
        help="whether apply *adaptive* resampling for BB-SMC",
    )
    compress_parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
        help="the latent size of the trained model",
    )
    compress_parser.add_argument(
        "--bound",
        type=str,
        default='elbo',
        choices=['elbo', 'iwae', 'fivo'],
        help="the variational bound used for the training the model",
    )

    compress_args = get_compress_args(compress_parser)
    print("=> Arguments:", compress_args)

    # set seed
    np.random.seed(compress_args.seed)
    random.seed(compress_args.seed)

    # add additional arguments used by the original FIVO repo
    compress_args.dataset_type = "pianoroll"
    compress_args.data_dimension = 88
    compress_args.observation_size = 88
    compress_args.data = compress_args.dataset
    compress_args.split = "test"
    compress_args.model = "vrnn"
    compress_args.proposal_type = "filtering"
    compress_args.batch_size = 1

    main_worker(compress_args)


if __name__ == '__main__':
    main()
