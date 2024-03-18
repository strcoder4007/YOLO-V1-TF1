import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

from ..utils import compose
from .darknet19 import (DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body)

sys.path.append('..')

voc_anchors = np.array([
    [1.08, 1.19], 
    [3.42, 4.41], 
    [6.63, 11.38], 
    [9.42, 5.11], 
    [16.62, 10.52]
])

voc_classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# def space_to_depth_x2(x):
#     """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
#     # Import currently required to make Lambda work.
#     # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
#     import tensorflow as tf
#     return tf.nn.space_to_depth(x, block_size=2)