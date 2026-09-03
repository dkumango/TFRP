###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""Relative Conservation Error (RCE) Benchmark for Conventional LRP (iNNvestigate).

This experiment measures Relative Conservation Error (RCE) for conventional LRP implementations
using the iNNvestigate library on 500 ImageNet samples with a ResNet50 model.

Evaluated Rules:
- LRP-z+
- LRP-epsilon (epsilon=0.01)
- LRP-alpha-beta (alpha=2, beta=1)
- Sequential Composite Preset A

Outputs:
- CSV report: `exp_result/LRP_5/lrp_relative_conservation_error.csv`
- Precomputed .npy attribution tensors for downstream analysis
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import innvestigate
import eval_lib as EVAL

# Enforce static graph execution for iNNvestigate
tf.compat.v1.disable_eager_execution()


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


def calculate_rce(r_in, r_out, eps=1e-9):
    """Compute Relative Conservation Error (RCE).

    Args:
        r_in (float): Input attribution sum (R_in).
        r_out (float): Target class output logit score (R_out).
        eps (float, optional): Epsilon numerical stabilizer. Defaults to 1e-9.

    Returns:
        float: Relative Conservation Error value.
    """
    abs_diff = np.abs(r_in - r_out)
    denominator = np.abs(r_out) + eps
    return abs_diff / denominator


# Load and strip softmax from model
model = ResNet50(weights='imagenet')
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

image_path = "imagen_500/"
file_list = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
random.seed(42)
images = random.sample(file_list, min(500, len(file_list)))
n_images = len(images)

input_imgs = np.zeros((n_images, 224, 224, 3))
org_imgs = np.zeros((n_images, 224, 224, 3))

result_z_plus = np.zeros((n_images, 224, 224, 3))
result_epsilon = np.zeros((n_images, 224, 224, 3))
result_alpha_beta = np.zeros((n_images, 224, 224, 3))
result_composite = np.zeros((n_images, 224, 224, 3))

for i in range(n_images):
    img = image.load_img(os.path.join(image_path, file_list[i]), target_size=(224, 224))
    x = image.img_to_array(img)
    org_imgs[i] = x.copy()
    input_imgs[i] = preprocess_input(x)

# Instantiate iNNvestigate analyzers
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

rules = [
    ('z_plus', analyzer_plus, result_z_plus),
    ('epsilon', analyzer_epsilon, result_epsilon),
    ('alpha_beta', analyzer_alpha_beta, result_alpha_beta),
    ('composite', analyzer_composite, result_composite)
]

lrp_rce_records = []

print("\n=== Evaluating Relative Conservation Error (RCE) across LRP Rules ===")

for i in range(n_images):
    x = np.expand_dims(input_imgs[i], axis=0)
    img_name = images[i]

    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    initial_sum = preds[0][top_class_idx]

    print(f"\n[{i}] Analyzing {img_name} -> Top Class Index: {top_class_idx}")

    for rule_name, analyzer, storage_array in rules:
        result = analyzer.analyze(x)
        storage_array[i] = result[0]

        final_sum = np.sum(result)
        rce = calculate_rce(r_in=final_sum, r_out=initial_sum, eps=1e-9)

        lrp_rce_records.append({
            'Image_Index': i,
            'File_Name': img_name,
            'Rule': rule_name,
            'Initial_Sum(R_out)': initial_sum,
            'Final_Sum(R_in)': final_sum,
            'Absolute_Error': np.abs(final_sum - initial_sum),
            'RCE_Value': rce,
            'RCE_Percentage(%)': rce * 100
        })
        print(f"Image {i} [{rule_name}] -> RCE: {rce:.2e} ({rce * 100:.5f}%)")

# Save attribution maps and tabular records
output_dir = "exp_result/LRP_5"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "result_lrp_z_plus.npy"), result_z_plus)
np.save(os.path.join(output_dir, "result_lrp_epsilon.npy"), result_epsilon)
np.save(os.path.join(output_dir, "result_lrp_alpha_beta.npy"), result_alpha_beta)
np.save(os.path.join(output_dir, "result_lrp_composite.npy"), result_composite)

df_lrp_rce = pd.DataFrame(lrp_rce_records)
csv_output_path = os.path.join(output_dir, "lrp_relative_conservation_error.csv")
df_lrp_rce.to_csv(csv_output_path, index=False)
print(f"\nResults saved to: '{csv_output_path}'")

summary_df = df_lrp_rce.groupby('Rule')[['Absolute_Error', 'RCE_Value', 'RCE_Percentage(%)']].mean()
print("\n" + "=" * 60)
print("Conventional LRP Relative Conservation Error (RCE) Summary")
print("=" * 60)
print(summary_df)
print("=" * 60)