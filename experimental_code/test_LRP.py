###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""Baseline LRP Benchmark on ImageNet Using iNNvestigate.

Environment:
    - Python Version: Python 3.7 (.venv_37)
    - Required Dependencies: iNNvestigate, TensorFlow 1.15.x / Keras 2.2.x (static graph mode)

This experiment evaluates traditional Layer-wise Relevance Propagation (LRP) implementations
via the `innvestigate` library across standard CNN architectures (VGG16, ResNet50, InceptionV3, MobileNetV2)
on ImageNet sample images.

Evaluated Rules:
- LRP-z+ (`lrp.z_plus`)
- LRP-epsilon (`lrp.epsilon`, epsilon=0.01)
- LRP-alpha-beta (`lrp.alpha_beta`, alpha=2, beta=1)
- Composite sequential preset A (`lrp.sequential_preset_a`)

Metrics Evaluated:
- Total Conservation Leakage (%)
- Relevance Sparsity (fraction of inactive pixels)
- Heatmap Shannon Entropy
- Execution Latency (seconds per explanation)
- Structure Tolerance against architectural components:
  * Residual branch additions (Add layers)
  * Zero-padding artifacts (ZeroPadding2D layers)
  * Spatial pooling (GlobalAveragePooling2D layers)

Results are exported to CSV in `exp_result/`.
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

from keras.applications import ResNet50, VGG16, InceptionV3, MobileNetV2
from keras.applications.inception_v3 import preprocess_input, decode_predictions

# Select target model architecture
# Options: "vgg16", "resnet50", "inception_v3", "mobilenet_v2"
model_type = "inception_v3"
model = InceptionV3(weights='imagenet')

out_fname = f"exp_result/lrp_{model_type}_500.csv"
os.makedirs("exp_result", exist_ok=True)


def set_seeds(seed=42):
    """Enforce deterministic operations and fix random seeds for bit-level reproducibility.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        None
    """
    # 1. Fix base Python and Hash seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    # 2. Fix NumPy seed
    np.random.seed(seed)

    # 3. Fix TensorFlow seed
    tf.random.set_random_seed(seed)

    # 4. Enforce deterministic TensorFlow operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


set_seeds(42)

# Remove softmax activation to analyze pre-softmax logit scores
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

# Dataset path and random sample selection
image_path = "imagen_500"
file_list = []

for file_name in os.listdir(image_path):
    full_path = os.path.join(image_path, file_name)
    if os.path.isfile(full_path):
        file_list.append(file_name)

random.seed(42)
images = random.sample(file_list, min(100, len(file_list)))
n_images = len(images)

if 'inception' in model.name.lower():
    img_width, img_height = (299, 299)
else:
    img_width, img_height = (224, 224)

img_shape = (img_width, img_height, 3)

# Initialize metric accumulators
imgs = []
global_rules = []
results = []
leakage_percents = []
sparsity = []
heatmap_entropy = []
st_add = []
st_padding = []
st_gap = []
times = []

# Instantiate synthetic models for structure tolerance evaluation
model_add = EVAL.create_add_test_model(img_shape)
model_padding = EVAL.create_padding_test_model(img_shape)
model_gap = EVAL.create_gap_test_model(img_shape)

# Create iNNvestigate analyzers for target model
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

# Create analyzers for structure tolerance models
analyzer_plus_add = innvestigate.create_analyzer(
    "lrp.z_plus", model_add, neuron_selection_mode="max_activation"
)
analyzer_plus_padding = innvestigate.create_analyzer(
    "lrp.z_plus", model_padding, neuron_selection_mode="max_activation"
)
analyzer_plus_gap = innvestigate.create_analyzer(
    "lrp.z_plus", model_gap, neuron_selection_mode="max_activation"
)
analyzer_epsilon_add = innvestigate.create_analyzer(
    "lrp.epsilon", model_add, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
)
analyzer_epsilon_padding = innvestigate.create_analyzer(
    "lrp.epsilon", model_padding, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
)
analyzer_epsilon_gap = innvestigate.create_analyzer(
    "lrp.epsilon", model_gap, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
)
analyzer_alpha_beta_add = innvestigate.create_analyzer(
    "lrp.alpha_beta", model_add, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
)
analyzer_alpha_beta_padding = innvestigate.create_analyzer(
    "lrp.alpha_beta", model_padding, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
)
analyzer_alpha_beta_gap = innvestigate.create_analyzer(
    "lrp.alpha_beta", model_gap, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
)
analyzer_composite_add = innvestigate.create_analyzer(
    "lrp.sequential_preset_a", model_add, neuron_selection_mode="max_activation"
)
analyzer_composite_padding = innvestigate.create_analyzer(
    "lrp.sequential_preset_a", model_padding, neuron_selection_mode="max_activation"
)
analyzer_composite_gap = innvestigate.create_analyzer(
    "lrp.sequential_preset_a", model_gap, neuron_selection_mode="max_activation"
)

for i in range(n_images):
    # Load and preprocess input image
    img = image.load_img(os.path.join(image_path, images[i]), target_size=(img_width, img_height))
    x = image.img_to_array(img)
    org_img = x.copy()
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Model inference
    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    probs = softmax(preds, axis=1)
    top_class_label = decode_predictions(probs, top=1)[0][0][1]
    final_relevance_sum = preds[0][top_class_idx]

    print(f"\n {i}, Analyzing {images[i]} with top class '{top_class_label}'")

    # 1. Evaluate LRP-z+
    start_time = time.time()
    result = analyzer_plus.analyze(x)
    end_time = time.time()

    imgs.append(images[i])
    global_rules.append('z_plus')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)
    del result

    # Structure-tolerance tests for LRP-z+
    if model_type == "resnet50":
        result = analyzer_plus_add.analyze(x)
        pred_st = model_add.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type in ["resnet50", "mobilenet_v2"]:
        result = analyzer_plus_padding.analyze(x)
        pred_st = model_padding.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type not in ["vgg16"]:
        result = analyzer_plus_gap.analyze(x)
        pred_st = model_gap.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    # 2. Evaluate LRP-epsilon
    start_time = time.time()
    result = analyzer_epsilon.analyze(x)
    end_time = time.time()

    imgs.append(images[i])
    global_rules.append('epsilon')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)
    del result

    # Structure-tolerance tests for LRP-epsilon
    if model_type == "resnet50":
        result = analyzer_epsilon_add.analyze(x)
        pred_st = model_add.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type in ["resnet50", "mobilenet_v2"]:
        result = analyzer_epsilon_padding.analyze(x)
        pred_st = model_padding.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type not in ["vgg16"]:
        result = analyzer_epsilon_gap.analyze(x)
        pred_st = model_gap.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    # 3. Evaluate LRP-alpha-beta
    start_time = time.time()
    result = analyzer_alpha_beta.analyze(x)
    end_time = time.time()

    imgs.append(images[i])
    global_rules.append('alpha_beta')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)
    del result

    # Structure-tolerance tests for LRP-alpha-beta
    if model_type == "resnet50":
        result = analyzer_alpha_beta_add.analyze(x)
        pred_st = model_add.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type in ["resnet50", "mobilenet_v2"]:
        result = analyzer_alpha_beta_padding.analyze(x)
        pred_st = model_padding.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type not in ["vgg16"]:
        result = analyzer_alpha_beta_gap.analyze(x)
        pred_st = model_gap.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    # 4. Evaluate Composite Sequential Preset A
    start_time = time.time()
    result = analyzer_composite.analyze(x)
    end_time = time.time()

    imgs.append(images[i])
    global_rules.append('composite')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)
    del result

    # Structure-tolerance tests for Composite Preset
    if model_type == "resnet50":
        result = analyzer_composite_add.analyze(x)
        pred_st = model_add.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type in ["resnet50", "mobilenet_v2"]:
        result = analyzer_composite_padding.analyze(x)
        pred_st = model_padding.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type not in ["vgg16"]:
        result = analyzer_composite_gap.analyze(x)
        pred_st = model_gap.predict(x)
        target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else:
        leakage_percent_st = None
    st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    del x, preds, img, org_img

    # Export intermediate results
    df = pd.DataFrame({
        "Image": imgs,
        "Global_Rule": global_rules,
        "Total_Leakage_(%)": leakage_percents,
        "Sparsity": sparsity,
        "Heatmap_Entropy": heatmap_entropy,
        "Execution_Time": times,
        "Structure-Tolerance_Add": st_add,
        "Structure-Tolerance_Padding": st_padding,
        "Structure-Tolerance_Gap": st_gap
    })
    df.to_csv(out_fname, index=False)

del model