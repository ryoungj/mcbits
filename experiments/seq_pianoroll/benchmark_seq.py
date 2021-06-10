import sys

assert sys.version[0] == '3', "Please use the `mcbits` environment with Python 3 for this script!"

import gzip
import bz2
import lzma
import numpy as np
import os

from scipy.sparse import coo_matrix
import pickle


def sparse_pianoroll_to_dense(pianoroll, min_note, num_notes):
  """Converts a sparse pianoroll to a dense numpy array.

  Given a sparse pianoroll, converts it to a dense numpy array of shape
  [num_timesteps, num_notes] where entry i,j is 1.0 if note j is active on
  timestep i and 0.0 otherwise.

  Args:
    pianoroll: A sparse pianoroll object, a list of tuples where the i'th tuple
      contains the indices of the notes active at timestep i.
    min_note: The minimum note in the pianoroll, subtracted from all notes so
      that the minimum note becomes 0.
    num_notes: The number of possible different note indices, determines the
      second dimension of the resulting dense array.
  Returns:
    dense_pianoroll: A [num_timesteps, num_notes] numpy array of floats.
    num_timesteps: A python int, the number of timesteps in the pianoroll.
  """
  num_timesteps = len(pianoroll)
  inds = []
  for time, chord in enumerate(pianoroll):
    # Re-index the notes to start from min_note.
    inds.extend((time, note-min_note) for note in chord)
  shape = [num_timesteps, num_notes]
  values = [1.] * len(inds)
  sparse_pianoroll = coo_matrix(
      (values, ([x[0] for x in inds], [x[1] for x in inds])),
      shape=shape)
  return sparse_pianoroll.toarray(), num_timesteps

def get_pianoroll_binarized(dataset, base_dir="./datasets",
                            split="test",min_note=21, max_note=108):
    num_notes = max_note - min_note + 1

    path = os.path.join(base_dir, "{}.pkl".format(dataset))
    with open(path, "rb") as f:
        raw_data = pickle.load(f, encoding="latin1")
    pianorolls = raw_data[split]

    messages = []
    lengths = []
    for sparse_pianoroll in pianorolls:
        message, length = sparse_pianoroll_to_dense(sparse_pianoroll, min_note, num_notes)
        messages.append(message.astype(bool))
        lengths.append(length)

    messages = np.concatenate(messages, axis=0)
    timesteps = sum(lengths)
    assert len(messages) == sum(lengths)
    return messages, timesteps

def bench_compressor(compress_fun, compressor_name, messages, messages_name, timesteps):
    byts = compress_fun(messages)
    n_bits = len(byts) * 8
    bits_per_pixel = n_bits / timesteps
    print("Dataset: {}. Total timesteps: {}, Compressor: {}. Rate: {:.2f} bits per timesteps.".
          format(messages_name, timesteps, compressor_name, bits_per_pixel))

def gzip_compress(messages):
    messages = np.packbits(messages) if messages.dtype is np.dtype(bool) else messages
    assert messages.dtype is np.dtype('uint8')
    return gzip.compress(messages.tobytes())

def bz2_compress(messages):
    messages = np.packbits(messages) if messages.dtype is np.dtype(bool) else messages
    assert messages.dtype is np.dtype('uint8')
    return bz2.compress(messages.tobytes())

def lzma_compress(messages):
    messages = np.packbits(messages) if messages.dtype is np.dtype(bool) else messages
    assert messages.dtype is np.dtype('uint8')
    return lzma.compress(messages.tobytes())


if __name__ == "__main__":
    dataset_list = ['trunc_musedata', 'trunc_jsb', 'trunc_nottingham', 'trunc_piano-midi.de']
    for dataset in dataset_list:
        messages, timesteps = get_pianoroll_binarized(dataset)
        bench_compressor(gzip_compress, "gzip", messages, dataset, timesteps)
        bench_compressor(bz2_compress, "bz2", messages, dataset, timesteps)
        bench_compressor(lzma_compress, "lzma", messages, dataset, timesteps)
