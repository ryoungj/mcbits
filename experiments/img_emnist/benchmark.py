#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is adapted from
https://github.com/bits-back/bits-back
"""
import io
import gzip
import bz2
import lzma
import numpy as np
import os

from torchvision import datasets, transforms
import PIL
import PIL.Image as pimg
import scipy.io

assert PIL.PILLOW_VERSION <= '5.4.1', f"Need to use Pillow <=\"5.4.1\" (current version {PIL.PILLOW_VERSION}) otherwise the PNG and WebP results are bad!!!"


def mnist_raw():
    mnist = datasets.MNIST(
        'data/mnist', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    return mnist.test_data.numpy()


def mnist_binarized(rng):
    raw_probs = mnist_raw() / 255
    return rng.random_sample(np.shape(raw_probs)) < raw_probs


def omniglot_binarized(rng):
    omni_raw = scipy.io.loadmat(
        os.path.join('data/omniglot', 'chardata.mat'))['testdata'].T.astype('float32')
    raw_probs = omni_raw.reshape((-1, 28, 28)).transpose(0, 2, 1)
    return rng.random_sample(np.shape(raw_probs)) < raw_probs


def emnist_raw(split='mnist'):
    emnist = datasets.EMNIST(
        'data/emnist', split=split, train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    return emnist.test_data.numpy()


def emnist_binarized(rng, split='mnist'):
    raw_probs = emnist_raw(split=split) / 255
    return rng.random_sample(np.shape(raw_probs)) < raw_probs


def bench_compressor(compress_fun, compressor_name, images, images_name):
    byts = compress_fun(images)
    n_bits = len(byts) * 8
    bits_per_pixel = n_bits / np.size(images)
    print("Dataset: {}. Compressor: {}. Rate: {:.4f} bits per channel.".
          format(images_name, compressor_name, bits_per_pixel))


def gzip_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return gzip.compress(images.tobytes())


def bz2_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return bz2.compress(images.tobytes())


def lzma_compress(images):
    original_size = np.size(images)
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype is np.dtype('uint8')
    return lzma.compress(images.tobytes())


def pimg_compress(format='PNG', **params):
    def compress_fun(images):
        compressed_data = bytearray()
        for n, image in enumerate(images):
            image = pimg.fromarray(image)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=format, **params)
            compressed_data.extend(img_bytes.getvalue())
        return compressed_data

    return compress_fun


def gz_and_pimg(images, format='PNG', **params):
    pimg_compressed_data = pimg_compress(images, format, **params)
    return gzip.compress(pimg_compressed_data)


if __name__ == "__main__":
    #  #MNIST_raw
    # images = mnist_raw()
    # name = 'raw mnist'
    # bench_compressor(gzip_compress, "gzip", images, name)
    # bench_compressor(bz2_compress, "bz2", images, name)
    # bench_compressor(lzma_compress, "lzma", images, name)
    # bench_compressor(
    #     pimg_compress("PNG", optimize=True), "PNG", images, name)
    # bench_compressor(
    #     pimg_compress('WebP', lossless=True, quality=100), "WebP", images, name)

    # # MNIST binarized
    # rng = np.random.RandomState(0)
    # name = 'binarized mnist'
    # images = mnist_binarized(rng)
    # bench_compressor(gzip_compress, "gzip", images, name)
    # bench_compressor(bz2_compress, "bz2", images, name)
    # bench_compressor(lzma_compress, "lzma", images, name)
    # bench_compressor(
    #     pimg_compress("PNG", optimize=True), "PNG", images, name)
    # bench_compressor(
    #     pimg_compress('WebP', lossless=True, quality=100), "WebP", images, name)

    # EMNIST binarized (mnist)
    rng = np.random.RandomState(0)
    name = 'binarized emnist (mnist)'
    images = emnist_binarized(rng, split='mnist')
    bench_compressor(gzip_compress, "gzip", images, name)
    bench_compressor(bz2_compress, "bz2", images, name)
    bench_compressor(lzma_compress, "lzma", images, name)
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, name)
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, name)

    # EMNIST binarized (letters)
    rng = np.random.RandomState(0)
    name = 'binarized emnist (letters)'
    images = emnist_binarized(rng, split='letters')
    bench_compressor(gzip_compress, "gzip", images, name)
    bench_compressor(bz2_compress, "bz2", images, name)
    bench_compressor(lzma_compress, "lzma", images, name)
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, name)
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, name)

    # # OMNIGLOT binarized
    # rng = np.random.RandomState(0)
    # name = 'binarized omniglot'
    # images = omniglot_binarized(rng)
    # bench_compressor(gzip_compress, "gzip", images, name)
    # bench_compressor(bz2_compress, "bz2", images, name)
    # bench_compressor(lzma_compress, "lzma", images, name)
    # bench_compressor(
    #     pimg_compress("PNG", optimize=True), "PNG", images, name)
    # bench_compressor(
    #     pimg_compress('WebP', lossless=True, quality=100), "WebP", images, name)
