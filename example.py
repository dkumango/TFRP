###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, matplotlib
###############################################################################
"""Minimal TFRP example: z+ relevance of the top-1 class of ResNet50 for one image.

Usage (from the repository root):  python example.py path/to/image.jpg
Output: printed conservation check and the first entries of the conservation log; relevance.png
"""

import sys

sys.path.insert(0, 'exp_code/tfrp')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from general_TFRP_v2 import get_relevance_map_generalized

# 1. A Keras model, used as is (no rewriting, no canonization)
model = tf.keras.applications.ResNet50(weights='imagenet')

# 2. An input image, preprocessed as for the model
img = tf.keras.utils.load_img(sys.argv[1], target_size=(224, 224))
x = tf.keras.applications.resnet50.preprocess_input(np.array(img, dtype=np.float32)[None])

# 3. Relevance of the top-1 class with the z+ rule (default)
R, seed, total, leak = get_relevance_map_generalized(model, x)
R = np.asarray(R)
print(f'explained score (bias-free logit) = {seed:.4f}')
print(f'sum of input relevance            = {total:.4f}')
print(f'relative conservation error       = {abs(total - seed) / abs(seed) * 100:.2e} %')

# 4. The per-operation conservation log: (operation, type, relevance in, relevance out)
*_, log = get_relevance_map_generalized(model, x, return_layer_log=True)
for name, op_type, r_in, r_out in log[:3]:
    print(f'  {name:22s} {op_type:24s} {r_in:10.4f} {r_out:10.4f}')

# 5. Heat map (sum over colour channels, symmetric colour scale)
heat = R[0].sum(-1)
lim = np.percentile(np.abs(heat), 99.5)
plt.imshow(heat, cmap='bwr', vmin=-lim, vmax=lim)
plt.axis('off')
plt.savefig('relevance.png', bbox_inches='tight')
