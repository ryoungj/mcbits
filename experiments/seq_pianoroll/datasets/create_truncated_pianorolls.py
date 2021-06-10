#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import itertools

import tensorflow as tf

tf.app.flags.DEFINE_string('in_file', None,
                           'Filename of the pickled pianoroll dataset to load.')
tf.app.flags.DEFINE_string('out_file', None,
                           'Name of the output pickle file. Defaults to in_file, '
                           'updating the input pickle file.')
tf.app.flags.DEFINE_integer('seq_len', 100,
                            'Length at which each sequences is going to be '
                            'truncated.')

tf.app.flags.mark_flag_as_required('in_file')

FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
    n = FLAGS.seq_len
    if FLAGS.out_file is None:
        FLAGS.out_file = FLAGS.in_file
    with tf.gfile.Open(FLAGS.in_file, 'r') as f:
        pianorolls = pickle.load(f)

    truncated_rolls = {}
    for split in ['train', 'valid', 'test']:
        truncated_rolls[split] = [[song[i:i + n] for i in range(0, len(song), n)] for song in pianorolls[split]]
        truncated_rolls[split] = [chunk for chunk in itertools.chain(*truncated_rolls[split])]
    # Write out the whole pickle file, including the train mean.
    pickle.dump(truncated_rolls, open(FLAGS.out_file, 'wb'))


if __name__ == '__main__':
    tf.app.run()
