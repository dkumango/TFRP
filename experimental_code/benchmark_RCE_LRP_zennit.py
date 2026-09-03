###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Relative Conservation Error (RCE) Benchmark for LRP using Zennit (PyTorch).

This script measures Relative Conservation Error (RCE) on ResNet50 across 500 ImageNet samples
using the PyTorch Zennit library for layer-wise relevance propagation.

Evaluated Rules:
- LRP-z+
- LRP-epsilon (epsilon=0.01)
- LRP-alpha-beta (alpha=2, beta=1)
- Composite (EpsilonPlus)

Outputs:
- CSV report: `exp_result/LRP_500/lrp_relative_conservation_error_zennit.csv`
- Precomputed .npy attribution tensors in `exp_result/LRP_500/`
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torchvision.models as models
import zennit.composites as composites
import zennit.rules as rules
from zennit.attribution import Gradient


def set_seeds(seed=42):
    """Enforce deterministic operations and fix random seeds for PyTorch.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# Load pretrained PyTorch ResNet50
print("Loading PyTorch ResNet50...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()
print("ResNet50 model loaded.")

image_path = "imagen_500/"
file_list = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
random.seed(42)
images = random.sample(file_list, min(500, len(file_list)))
n_images = len(images)

result_z_plus = np.zeros((n_images, 224, 224, 3))
result_epsilon = np.zeros((n_images, 224, 224, 3))
result_alpha_beta = np.zeros((n_images, 224, 224, 3))
result_composite = np.zeros((n_images, 224, 224, 3))

# Configure Zennit LRP composites
rules_meta = [
    ('z_plus', composites.LayerMapComposite(layer_map=[(nn.Conv2d, rules.ZPlus()), (nn.Linear, rules.ZPlus())]), result_z_plus),
    ('epsilon', composites.LayerMapComposite(layer_map=[(nn.Conv2d, rules.Epsilon(epsilon=0.01)), (nn.Linear, rules.Epsilon(epsilon=0.01))]), result_epsilon),
    ('alpha_beta', composites.LayerMapComposite(layer_map=[(nn.Conv2d, rules.AlphaBeta(alpha=2, beta=1)), (nn.Linear, rules.AlphaBeta(alpha=2, beta=1))]), result_alpha_beta),
    ('composite', composites.EpsilonPlus(), result_composite)
]

lrp_rce_records = []

print("\nStarting Zennit LRP Relative Conservation Error (RCE) Benchmark on ResNet50...")
print("=" * 80)

for i in range(n_images):
    img_name = images[i]
    img_full_path = os.path.join(image_path, img_name)
    img = Image.open(img_full_path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)

    # Preprocessing (RGB -> BGR and Mean Subtraction)
    x_arr = np.array(img, dtype=np.float32)[..., ::-1].copy()
    x_arr[..., 0] -= 103.939
    x_arr[..., 1] -= 116.779
    x_arr[..., 2] -= 123.68

    x_tensor = torch.from_numpy(x_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)
    x_tensor.requires_grad_(True)

    preds = model(x_tensor)
    top_class_idx = preds.argmax(dim=1).item()
    initial_sum = preds[0, top_class_idx].item()

    print(f"\n[{i}] Analyzing {img_name} -> Top Class Index: {top_class_idx}")

    target_mask = torch.zeros_like(preds)
    target_mask[0, top_class_idx] = 1.0

    for rule_name, comp, storage_array in rules_meta:
        with Gradient(model=model, composite=comp) as attr:
            _, relevance = attr(x_tensor, target_mask)

        rel_np = relevance[0].permute(1, 2, 0).detach().cpu().numpy()
        storage_array[i] = rel_np

        final_sum = np.sum(rel_np)
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
output_dir = "exp_result/LRP_500"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "result_LRP_z_plus_zennit.npy"), result_z_plus)
np.save(os.path.join(output_dir, "result_LRP_epsilon_zennit.npy"), result_epsilon)
np.save(os.path.join(output_dir, "result_LRP_alpha_beta_zennit.npy"), result_alpha_beta)
np.save(os.path.join(output_dir, "result_LRP_composite_zennit.npy"), result_composite)

df_lrp_rce = pd.DataFrame(lrp_rce_records)
csv_output_path = os.path.join(output_dir, "lrp_relative_conservation_error_zennit.csv")
df_lrp_rce.to_csv(csv_output_path, index=False)
print(f"\nResults saved to: '{csv_output_path}'")

summary_df = df_lrp_rce.groupby('Rule')[['Absolute_Error', 'RCE_Value', 'RCE_Percentage(%)']].mean()
print("\n" + "=" * 80)
print("Zennit LRP Relative Conservation Error (RCE) Summary on ResNet50")
print("=" * 80)
print(summary_df.to_string())
print("=" * 80)
