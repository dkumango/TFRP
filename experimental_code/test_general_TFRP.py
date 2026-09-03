###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Generalized TFRP Benchmark on ImageNet.

Environment:
    - Python Version: Python 3.12 (.venv_312)

This experiment evaluates the Generalized Token/Feature Relevance Propagation (TFRP) method
across standard CNN architectures (VGG16, ResNet50, InceptionV3, MobileNetV2) on ImageNet samples.

Metrics Evaluated:
- Total Conservation Leakage (%)
- Relevance Sparsity (fraction of inactive pixels)
- Heatmap Shannon Entropy (spatial focus vs. fragmentation)
- Execution Latency (seconds per explanation)
- Structure Tolerance against architectural components:
  * Residual branch additions (Add layers)
  * Zero-padding artifacts (ZeroPadding2D layers)
  * Spatial pooling (GlobalAveragePooling2D layers)

Results are recorded and exported as a summary CSV in `exp_result/`.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

import general_TFRP_v1 as TFRP
import eval_lib as EVAL


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
    tf.random.set_seed(seed)

    # 4. Enforce deterministic TensorFlow operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


set_seeds(42)

# Load target model
# Available options: "vgg16", "resnet50", "inception_v3", "mobilenet_v2"
model_type = "mobilenet_v2"
model = MobileNetV2(weights='imagenet')

if 'inception' in model.name.lower():
    img_width, img_height = (299, 299)
else:
    img_width, img_height = (224, 224)
img_shape = (img_width, img_height, 3)

out_fname = f"exp_result/tfrp_{model_type}_500.csv"
os.makedirs("exp_result", exist_ok=True)

# Image directory and sample selection
image_path = 'imagen_500'
file_list = []

for file_name in os.listdir(image_path):
    full_path = os.path.join(image_path, file_name)
    if os.path.isfile(full_path):
        file_list.append(file_name)

random.seed(42)
images = random.sample(file_list, min(100, len(file_list)))
n_images = len(images)

# Conservation and execution benchmark setup
global_rule = ['z_plus', 'epsilon', 'alpha_beta']
composite_preset = {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}
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

    # Evaluate individual global rules
    for g_rule in global_rule:
        start_time = time.time()
        result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
            model, x, target_class_idx=None,
            use_logit=True, global_rule=g_rule, composite_preset=None,
            alpha=2.0, beta=1.0, epsilon=1e-2
        )
        end_time = time.time()

        results.append(result)
        imgs.append(images[i])
        global_rules.append(g_rule)
        leakage_percents.append(np.abs(leakage_percent))
        times.append(end_time - start_time)
        sparsity.append(EVAL.relevance_sparsity(result))
        heatmap_entropy.append(EVAL.heatmap_entropy(result))

        # Structure-tolerance test: Residual Addition
        if model_type == "resnet50":
            result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
                model_add, x, target_class_idx=None,
                use_logit=True, global_rule=g_rule, composite_preset=None,
                alpha=2.0, beta=1.0, epsilon=1e-2
            )
        else:
            leakage_percent_st = None
        st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

        # Structure-tolerance test: Zero Padding
        if model_type in ["resnet50", "mobilenet_v2"]:
            result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
                model_padding, x, target_class_idx=None,
                use_logit=True, global_rule=g_rule, composite_preset=None,
                alpha=2.0, beta=1.0, epsilon=1e-2
            )
        else:
            leakage_percent_st = None
        st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

        # Structure-tolerance test: Global Average Pooling (GAP)
        if model_type not in ["vgg16"]:
            result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
                model_gap, x, target_class_idx=None,
                use_logit=True, global_rule=g_rule, composite_preset=None,
                alpha=2.0, beta=1.0, epsilon=1e-2
            )
        else:
            leakage_percent_st = None
        st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    # Evaluate composite preset rule
    start_time = time.time()
    result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
        model, x, target_class_idx=None,
        use_logit=True, global_rule='z_plus', composite_preset=composite_preset,
        alpha=2.0, beta=1.0, epsilon=1e-7
    )
    end_time = time.time()

    results.append(result)
    imgs.append(images[i])
    global_rules.append('composite')
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)

    # Structure-tolerance evaluation for composite preset
    if model_type == "resnet50":
        result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
            model_add, x, target_class_idx=None,
            use_logit=True, global_rule='z_plus', composite_preset=None,
            alpha=2.0, beta=1.0, epsilon=1e-2
        )
    else:
        leakage_percent_st = None
    st_add.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type in ["resnet50", "mobilenet_v2"]:
        result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
            model_padding, x, target_class_idx=None,
            use_logit=True, global_rule='z_plus', composite_preset=None,
            alpha=2.0, beta=1.0, epsilon=1e-2
        )
    else:
        leakage_percent_st = None
    st_padding.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    if model_type not in ["vgg16"]:
        result_st, initial_sum_st, final_sum_st, leakage_percent_st = TFRP.get_relevance_map_generalized(
            model_gap, x, target_class_idx=None,
            use_logit=True, global_rule='z_plus', composite_preset=None,
            alpha=2.0, beta=1.0, epsilon=1e-2
        )
    else:
        leakage_percent_st = None
    st_gap.append(np.abs(leakage_percent_st) if leakage_percent_st is not None else None)

    # Compile and export current benchmark results
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
    del x, preds, img, org_img

del model
