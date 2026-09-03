###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""Precompute and Export LRP Attribution Maps on ImageNet Samples.

This script runs conventional LRP (z+, epsilon, alpha-beta, and composite preset)
on 5 representative ImageNet sample images using ResNet50, and saves the resulting
attribution tensors as NumPy `.npy` files for visualization and benchmarking.

Outputs:
- `exp_result/LRP_5/result_LRP_z_plus.npy`
- `exp_result/LRP_5/result_LRP_epsilon.npy`
- `exp_result/LRP_5/result_LRP_alpha_beta.npy`
- `exp_result/LRP_5/result_LRP_composite.npy`
"""

import os
import random
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from keras.preprocessing import image
import innvestigate
import eval_lib as EVAL

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

model_type = "resnet50"
os.makedirs("exp_result/LRP_5", exist_ok=True)
model = ResNet50(weights='imagenet')


def set_seeds(seed=42):
    """Enforce deterministic operations and fix random seeds.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


set_seeds(42)

# Remove softmax activation to analyze pre-softmax logits
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

image_path = "imagen_5/"
file_list = []

for file_name in os.listdir(image_path):
    full_path = os.path.join(image_path, file_name)
    if os.path.isfile(full_path):
        file_list.append(file_name)

n_images = len(file_list)
img_shape = (224, 224, 3)

org_imgs = np.zeros((n_images, 224, 224, 3))
input_imgs = np.zeros((n_images, 224, 224, 3))

result_z_plus = np.zeros((n_images, 224, 224, 3))
result_epsilon = np.zeros((n_images, 224, 224, 3))
result_alpha_beta = np.zeros((n_images, 224, 224, 3))
result_composite = np.zeros((n_images, 224, 224, 3))

for i in range(n_images):
    print(i, end=",")
    img = image.load_img(os.path.join(image_path, file_list[i]), target_size=(224, 224))
    x = image.img_to_array(img)
    org_imgs[i] = x.copy()
    x = preprocess_input(x)
    input_imgs[i] = x.copy()

analyzer_plus = innvestigate.create_analyzer(
    "lrp.z_plus", model, neuron_selection_mode="max_activation"
)
analyzer_epsilon = innvestigate.create_analyzer(
    "lrp.epsilon", model, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
)
analyzer_alpha_beta = innvestigate.create_analyzer(
    "lrp.alpha_beta", model, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
)
analyzer_composite = innvestigate.create_analyzer(
    "lrp.sequential_preset_a", model, neuron_selection_mode="max_activation"
)

for i in range(n_images):
    x = np.expand_dims(input_imgs[i], axis=0)

    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    probs = softmax(preds, axis=1)
    top_class_label = decode_predictions(probs, top=1)[0][0][1]

    print(f"\n {i}, Analyzing {file_list[i]} with top class '{top_class_label}'")

    # LRP-z+
    result = analyzer_plus.analyze(x)
    result_z_plus[i] = result[0]

    # LRP-epsilon
    result = analyzer_epsilon.analyze(x)
    result_epsilon[i] = result[0]

    # LRP-alpha-beta
    result = analyzer_alpha_beta.analyze(x)
    result_alpha_beta[i] = result[0]

    # LRP-composite
    result = analyzer_composite.analyze(x)
    result_composite[i] = result[0]

np.save("exp_result/LRP_5/result_LRP_z_plus.npy", result_z_plus)
np.save("exp_result/LRP_5/result_LRP_epsilon.npy", result_epsilon)
np.save("exp_result/LRP_5/result_LRP_alpha_beta.npy", result_alpha_beta)
np.save("exp_result/LRP_5/result_LRP_composite.npy", result_composite)
print("\nLRP .npy attribution files saved to exp_result/LRP_5/")
