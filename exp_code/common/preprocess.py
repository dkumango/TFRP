###############################################################################
# Execution Environment: any (Python 3.7 / 3.12) -- numpy only
###############################################################################
"""Shared image cache and model-specific preprocessing for all three environments.

Images are decoded and resized once (PIL, bilinear) and cached as uint8 arrays, so the
TensorFlow, PyTorch and Python 3.7 (iNNvestigate) runs see bit-identical inputs.
Preprocessing reproduces keras.applications exactly:
  caffe : RGB -> BGR, subtract ImageNet BGR means      (VGG16, ResNet50)
  tf    : x / 127.5 - 1                                (InceptionV3, MobileNetV2)
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# image cache written by exp_code/e1_prepare.py, and the folder with the 1,000 imagen JPEGs
CACHE_DIR = os.environ.get('TFRP_CACHE', os.path.join(ROOT, 'cache'))
IMAGE_DIR_DEFAULT = os.environ.get('TFRP_IMAGE_DIR', os.path.join(ROOT, 'data', 'imagen'))

MODEL_INFO = {  # name: (input size, preprocessing mode)
    'vgg16': (224, 'caffe'),
    'resnet50': (224, 'caffe'),
    'inception_v3': (299, 'tf'),
    'mobilenet_v2': (224, 'tf'),
}
CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def image_list():
    with open(os.path.join(CACHE_DIR, 'images.txt'), encoding='utf-8') as f:
        return f.read().split()


def sample_indices(n, seed=42):
    """Sorted random subset of cache indices (the cache is ordered by file name, i.e. by class,
    so a prefix would cover only a few classes). n >= N returns all indices."""
    total = len(image_list())
    if n >= total:
        return list(range(total))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=n, replace=False).tolist())


def rgb_cache(size):
    """uint8 array (N, size, size, 3) in the order of image_list()."""
    return np.load(os.path.join(CACHE_DIR, 'rgb_%d.npy' % size), mmap_mode='r')


def preprocess(rgb_uint8, mode):
    x = np.asarray(rgb_uint8, dtype=np.float32)
    if mode == 'caffe':
        return x[..., ::-1] - CAFFE_MEAN_BGR
    if mode == 'tf':
        return x / 127.5 - 1.0
    raise ValueError(mode)


def model_input(model_name, index):
    """Preprocessed input (1, H, W, 3) float32, NHWC."""
    size, mode = MODEL_INFO[model_name]
    return np.ascontiguousarray(preprocess(rgb_cache(size)[index], mode)[None])
