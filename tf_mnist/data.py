from pathlib import Path
from typing import Tuple

import requests
from urllib.parse import urlparse
from hashlib import md5
from subprocess import check_call
import gzip
import shutil
import numpy as np
import struct
import tensorflow as tf

from tqdm import tqdm

INPUT = Path('./input')

FILES_GZ = [
    ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
     '6bbc9ace898e44ae57da46a324031adb'),
    ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
     'a25bea736e30d166cdddb491f175f624'),
    ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
     '2646ac647ad5339dbf082846283269ea'),
    ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
     '27ae3e4e09519cfbb04c329615203637')
]

IMAGES = {'train': INPUT / 'train-images-idx3-ubyte',
          'val': INPUT / 't10k-images-idx3-ubyte'}
LABELS = {'train': INPUT / 'train-labels-idx1-ubyte',
          'val': INPUT / 't10k-labels-idx1-ubyte'}


def md5sum(file: Path):
    data = file.open('rb').read()
    return md5(data).hexdigest()


def get_data(**kwargs):
    """
    Get MNIST data from Yann LeCun site. Check for existence first.
    """
    for raw_url, file_hash in FILES_GZ:
        url = urlparse(raw_url)
        # store data in INPUT
        dest = INPUT / Path(url.path).name

        # check if we already have the unpacked data
        dest_unpacked = dest.with_suffix('')
        if dest_unpacked.exists() and md5sum(dest_unpacked) == file_hash:
            tqdm.write(f'Already downloaded {dest_unpacked}')
            continue

        # do download with neat progress bars
        r = requests.get(raw_url, stream=True)
        file_size = int(r.headers.get('content-length', 0))
        tqdm.write(f'Downloading {raw_url}')
        if file_size:
            bar = tqdm(total=file_size)
        else:
            bar = tqdm()
        with dest.open('wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    bar.update(len(chunk))
        bar.close()

        # use gzip module to unpack downloaded files
        tqdm.write(f'Unpacking {dest}')
        with gzip.open(str(dest), 'rb') as gz_src:
            with dest_unpacked.open('wb') as gz_dst:
                shutil.copyfileobj(gz_src, gz_dst)

        dest.unlink()


def read_mnist_images(split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST images data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = IMAGES[split].open('rb')
    magic, size, h, w = struct.unpack('>iiii', fd.read(4 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, h, w, 1)
    fd.close()

    return data


def read_mnist_labels(split) -> np.ndarray:
    """
    Create tf.data.Dataset out of MNIST labels data
    :param split: one of 'train' or 'val' for training or validation data
    """
    assert split in ['train', 'val']

    # read data as numpy array. The data structure is specified in Yann LeCun
    # site.
    fd = LABELS[split].open('rb')
    magic, size, = struct.unpack('>ii', fd.read(2 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, 1)
    fd.close()

    return data


def normalize(images):
    """
    Normalize images to [-1,1]
    """

    images = tf.cast(images, tf.float32)
    images /= 255.
    images -= 0.5
    images *= 2
    return images


def transform_train(images, labels):
    """
    Apply transformations to MNIST data for use in training.

    To images: random zoom and crop to 28x28, then normalize to [-1, 1]
    To labels: one-hot encode.
    """
    zoom = 0.9 + np.random.random() * 0.2  # random between 0.9-1.1
    size = int(round(zoom * 28))
    images = tf.image.resize_bilinear(images, (size, size))
    images = tf.image.resize_image_with_crop_or_pad(images, 28, 28)
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def transform_val(images, labels):
    """
    Normalize MNIST images and one-hot encode labels.
    """
    images = normalize(images)
    labels = tf.one_hot(labels, 10)
    labels = tf.squeeze(labels, 1)
    return images, labels


def create_mnist_dataset(batch_size, split) -> Tuple[tf.data.Dataset, int]:
    """
    Creates an Dataset for MNIST Data.

    This function create the correct tf.data.Dataset for a given split, transforms and
    batch inputs.
    """
    images = read_mnist_images(split)
    labels = read_mnist_labels(split)

    def gen():
        for image, label in zip(images, labels):
            yield image, label

    ds = tf.data.Dataset.from_generator(gen, (tf.uint8, tf.uint8), ((28, 28, 1), (1,)))

    if split == 'train':
        return ds.batch(batch_size).map(transform_train), len(labels)
    elif split == 'val':
        return ds.batch(batch_size).map(transform_val), len(labels)
