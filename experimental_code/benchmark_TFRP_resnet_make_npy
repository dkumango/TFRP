###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Precompute and Export TFRP Attribution Maps on ImageNet Samples.

This script executes Generalized TFRP (z+, epsilon, alpha-beta, and composite preset)
on 5 ImageNet sample images using a ResNet50 model, saving the generated relevance heatmaps
into NumPy `.npy` files for visualization and benchmarking.

Outputs:
- `exp_result/TFRP_5/result_TFRP_z_plus.npy`
- `exp_result/TFRP_5/result_TFRP_epsilon.npy`
- `exp_result/TFRP_5/result_TFRP_alpha_beta.npy`
- `exp_result/TFRP_5/result_TFRP_composite.npy`
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import general_TFRP_v1 as TFRP
import eval_lib as EVAL

model_type = "resnet50"
img_shape = (224, 224, 3)
os.makedirs("exp_result/TFRP_5", exist_ok=True)
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
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


set_seeds(42)

image_path = 'imagen_5/'
file_list = []

for file_name in os.listdir(image_path):
    full_path = os.path.join(image_path, file_name)
    if os.path.isfile(full_path):
        file_list.append(file_name)

random.seed(42)
n_images = len(file_list)

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

preds = model.predict(input_imgs)
top_classes = np.argmax(preds, axis=1)
top_class_labels = decode_predictions(preds, top=1)

global_rule = ['z_plus', 'epsilon', 'alpha_beta']
composite_preset = {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}

for i in range(n_images):
    x = np.expand_dims(input_imgs[i], axis=0)

    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    probs = softmax(preds, axis=1)
    top_class_label = decode_predictions(probs, top=1)[0][0][1]

    print(f"\n {i}, Analyzing {file_list[i]} with top class '{top_class_label}'")

    for g_rule in global_rule:
        result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
            model, x, target_class_idx=None,
            use_logit=True, global_rule=g_rule, composite_preset=None,
            alpha=2.0, beta=1.0, epsilon=1e-2
        )

        if g_rule == 'z_plus':
            result_z_plus[i] = result[0]
        elif g_rule == 'epsilon':
            result_epsilon[i] = result[0]
        elif g_rule == 'alpha_beta':
            result_alpha_beta[i] = result[0]

    # Evaluate composite preset
    result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
        model, x, target_class_idx=None,
        use_logit=True, global_rule='z_plus', composite_preset=composite_preset,
        alpha=2.0, beta=1.0, epsilon=1e-7
    )
    result_composite[i] = result[0]

np.save("exp_result/TFRP_5/result_TFRP_z_plus.npy", result_z_plus)
np.save("exp_result/TFRP_5/result_TFRP_epsilon.npy", result_epsilon)
np.save("exp_result/TFRP_5/result_TFRP_alpha_beta.npy", result_alpha_beta)
np.save("exp_result/TFRP_5/result_TFRP_composite.npy", result_composite)
print("\nTFRP .npy attribution files saved to exp_result/TFRP_5/")
