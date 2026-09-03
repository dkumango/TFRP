###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Relative Conservation Error (RCE) Benchmark for Generalized TFRP.

This experiment evaluates the Relative Conservation Error (RCE) of Generalized TFRP
across 500 ImageNet samples on a pretrained ResNet50 backbone.

Evaluated Rules:
- LRP-z+
- LRP-epsilon
- LRP-alpha-beta
- Composite rule preset

Metric Formulation:
    RCE = |R_in - R_out| / (|R_out| + eps)
    where R_out is the top class target logit and R_in is the sum of input-level attributions.

Outputs:
- CSV report: `exp_result/TFRP_5/tfrp_relative_conservation_error.csv`
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
    tf.config.experimental.enable_op_determinism()


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


# Load pretrained model and prepare dataset
model = ResNet50(weights='imagenet')
image_path = 'imagen_500/'

file_list = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
random.seed(42)
images = random.sample(file_list, min(500, len(file_list)))
n_images = len(images)

input_imgs = np.zeros((n_images, 224, 224, 3))
for i in range(n_images):
    img = image.load_img(os.path.join(image_path, file_list[i]), target_size=(224, 224))
    x = image.img_to_array(img)
    input_imgs[i] = preprocess_input(x)

# Obtain model predictions
preds = model.predict(input_imgs)
top_classes = np.argmax(preds, axis=1)

global_rules = ['z_plus', 'epsilon', 'alpha_beta']
composite_preset = {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}
rce_records = []

print("\n=== Evaluating Relative Conservation Error (RCE) across TFRP Rules ===")

for i in range(n_images):
    x = np.expand_dims(input_imgs[i], axis=0)
    img_name = images[i]

    # Evaluate individual global rules
    for g_rule in global_rules:
        result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
            model, x, target_class_idx=top_classes[i], use_logit=True,
            global_rule=g_rule, composite_preset=None, alpha=2.0, beta=1.0, epsilon=1e-2
        )

        rce = calculate_rce(r_in=final_sum, r_out=initial_sum, eps=1e-9)

        rce_records.append({
            'Image_Index': i,
            'File_Name': img_name,
            'Rule': g_rule,
            'Initial_Sum(R_out)': initial_sum,
            'Final_Sum(R_in)': final_sum,
            'Absolute_Error': np.abs(final_sum - initial_sum),
            'RCE_Value': rce,
            'RCE_Percentage(%)': rce * 100
        })
        print(f"Image {i} [{g_rule}] -> RCE: {rce:.2e} ({rce * 100:.5f}%)")

    # Evaluate composite preset rule
    result, initial_sum, final_sum, leakage_percent = TFRP.get_relevance_map_generalized(
        model, x, target_class_idx=top_classes[i], use_logit=True,
        global_rule='z_plus', composite_preset=composite_preset, alpha=2.0, beta=1.0, epsilon=1e-7
    )

    rce_comp = calculate_rce(r_in=final_sum, r_out=initial_sum, eps=1e-9)

    rce_records.append({
        'Image_Index': i,
        'File_Name': img_name,
        'Rule': 'composite',
        'Initial_Sum(R_out)': initial_sum,
        'Final_Sum(R_in)': final_sum,
        'Absolute_Error': np.abs(final_sum - initial_sum),
        'RCE_Value': rce_comp,
        'RCE_Percentage(%)': rce_comp * 100
    })
    print(f"Image {i} [composite] -> RCE: {rce_comp:.2e} ({rce_comp * 100:.5f}%)")

# Save results
df_rce = pd.DataFrame(rce_records)
os.makedirs("exp_result/TFRP_5", exist_ok=True)
csv_path = "exp_result/TFRP_5/tfrp_relative_conservation_error.csv"
df_rce.to_csv(csv_path, index=False)

summary_df = df_rce.groupby('Rule')[['Absolute_Error', 'RCE_Value', 'RCE_Percentage(%)']].mean()
print("\n" + "=" * 60)
print("TFRP Relative Conservation Error (RCE) Summary")
print("=" * 60)
print(summary_df)
print("=" * 60)
