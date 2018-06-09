from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import tensorflow as tf
import math
from six.moves import urllib
import csv
import glob
import re

IMAGENET_PATH = './Datasets/ILSVRC2012/'
DATA_DIR = './Datasets/'
URLs = {
    'cifar10': 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz',
    'cifar100': 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
    }


def __maybe_download(data_url, dest_directory, apply_func=None):
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if apply_func is not None:
        apply_func(filepath)

def __read_cifar(filenames, shuffle=True, cifar100=False):
  """Reads and parses examples from CIFAR data files.
  """
  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle,num_epochs=None)

  label_bytes = 1  # 2 for CIFAR-100
  if cifar100:
      label_bytes = 2
  height = 32
  width = 32
  depth = 3
  image_bytes = height * width * depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [depth, height, width])
  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.transpose(depth_major, [1, 2, 0])

  return tf.cast(image, tf.float32), label

def __read_imagenet(path, shuffle=True, save_file = 'imagenet_files.csv'):
    if not os.path.exists(save_file):
        def class_index(fn):
            class_id = re.search(r'(n\d+)', fn).group(1)
            return synset_map[class_id]['index']

        file_list = glob.glob(path+'/*/*.JPEG')
        label_indexes = []
        with open(save_file, 'wb') as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
            for f in file_list:
                idx = class_index(f)
                label_indexes.append(idx)
                wr.writerow([f, idx])

    with open(save_file, 'rb') as f:
        reader = csv.reader(f)
        file_list = list(reader)
    file_tuple, label_tuple = zip(*file_list)

    filename, labels = tf.train.slice_input_producer([list(file_tuple), list(label_tuple)], shuffle=shuffle)
    images = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    images = tf.div(tf.add(tf.to_float(images), -127), 128)
    return images, tf.string_to_number(labels, tf.int32)


def __load_jpeg(filenames, channels=3):
    return tf.image.decode_jpeg(tf.read_file(filename), channels=3), filenames

class DataProvider:
    def __init__(self, data, size=None, training=True):
        self.size = size or [None]*4
        self.data = data
        self.training = training

    def generate_batches(self, batch_size, min_queue_examples=1000, num_threads=8):
        """Construct a queued batch of images and labels.

        Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.

        Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.

        image, label = self.data
        if self.training:
            images, label_batch = tf.train.shuffle_batch(
            [preprocess_training(image, height=self.size[1], width=self.size[2]), label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
            [preprocess_evaluation(image, height=self.size[1], width=self.size[2]), label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size)

        return images, tf.reshape(label_batch, [batch_size])



def preprocess_evaluation(img, height=None, width=None, normalize=False):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]
    preproc_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
    if normalize:
         # Subtract off the mean and divide by the variance of the pixels.
        preproc_image = tf.image.per_image_whitening(preproc_image)
    else:
        distorted_image=tf.image.per_image_standardization(preproc_image)
    return preproc_image

def preprocess_training(img, height=None, width=None, normalize=False):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(img, [height-3, width-3, 3])
    distorted_image = tf.image.resize_image_with_crop_or_pad(distorted_image, height, width)
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                             max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                           lower=0.2, upper=1.8)
    if normalize:
        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_whitening(distorted_image)
    else:
        distorted_image=tf.image.per_image_standardization(distorted_image)
    return distorted_image

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img



def get_data_provider(name, training=True):
    if name == 'imagenet':
        return DataProvider(__read_imagenet(IMAGENET_PATH), [1280000, 224, 224, 3], True)
    elif name == 'cifar10':
        path = os.path.join(DATA_DIR,'cifar10')
        url = URLs['cifar10']
        def post_f(f): return tarfile.open(f, 'r:gz').extractall(path)
        __maybe_download(url, path,post_f)
        data_dir = os.path.join(path, 'cifar-10-batches-bin/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'data_batch_%d.bin' % i)
                                    for i in range(1, 6)]), [50000, 32,32,3], True)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test_batch.bin')]),
                                [10000, 32,32, 3], False)
    elif name == 'cifar100':
        path = os.path.join(DATA_DIR,'cifar100')
        url = URLs['cifar100']
        def post_f(f): return tarfile.open(f, 'r:gz').extractall(path)
        __maybe_download(url, path,post_f)
        data_dir = os.path.join(path, 'cifar-100-batches-bin/')
        if training:
            return DataProvider([os.path.join(data_dir, 'train.bin')],
                                    50000, True, __read_cifar)
        else:
            return DataProvider([os.path.join(data_dir, 'test.bin')],
                                10000, False, __read_cifar)
