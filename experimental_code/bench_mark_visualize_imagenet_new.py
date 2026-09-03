###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Qualitative Comparison Visualization for LRP and TFRP Variants.

This script loads precomputed LRP and TFRP attribution maps on 5 ImageNet samples
and constructs a 5x9 multi-panel comparison grid:
Columns:
1. Original Image
2. LRP z+ (Raw)
3. LRP z+ (Postprocessed)
4. TFRP z+ (Raw)
5. TFRP z+ (Postprocessed)
6. LRP Composite (Raw)
7. LRP Composite (Postprocessed)
8. TFRP Composite (Raw)
9. TFRP Composite (Postprocessed)

Output figure is exported to: `exp_result/qualitative_comparison_variants.png`.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import general_TFRP_v1 as TFRP

# Model and path configuration (using relative paths for portability)
base_path = "./"
image_path = os.path.join(base_path, 'imagen_5')
tfrp_image_path = os.path.join(base_path, 'exp_result', 'TFRP_5')
lrp_image_path = os.path.join(base_path, 'exp_result', 'LRP_5')
out_dir = os.path.join(base_path, 'exp_result')
os.makedirs(out_dir, exist_ok=True)

model = ResNet50(weights='imagenet')

# 1. Load original input images
n_images = 5
file_list = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]

org_imgs = np.zeros((n_images, 224, 224, 3))
input_imgs = np.zeros((n_images, 224, 224, 3))

for i in range(n_images):
    img = image.load_img(os.path.join(image_path, file_list[i]), target_size=(224, 224))
    x = image.img_to_array(img)
    org_imgs[i] = x.copy()
    x = preprocess_input(x)
    input_imgs[i] = x.copy()

# 2. Load precomputed attribution maps (.npy)
LRP_zplus = np.load(os.path.join(lrp_image_path, "result_lrp_z_plus.npy"))
TFRP_zplus = np.load(os.path.join(tfrp_image_path, "result_TFRP_z_plus.npy"))

LRP_epsilon = np.load(os.path.join(lrp_image_path, "result_lrp_epsilon.npy"))
TFRP_epsilon = np.load(os.path.join(tfrp_image_path, "result_TFRP_epsilon.npy"))

LRP_alphabeta = np.load(os.path.join(lrp_image_path, "result_lrp_alpha_beta.npy"))
TFRP_alphabeta = np.load(os.path.join(tfrp_image_path, "result_TFRP_alpha_beta.npy"))

LRP_composite = np.load(os.path.join(lrp_image_path, "result_lrp_composite.npy"))
TFRP_composite = np.load(os.path.join(tfrp_image_path, "result_TFRP_composite.npy"))


def normalize_heatmap(heatmap, clip_percentile: float = 99.5) -> np.ndarray:
    """Normalize attribution heatmap with percentile clipping and max-abs scaling.

    Args:
        heatmap (np.ndarray): Attribution tensor.
        clip_percentile (float, optional): Percentile for outlier clipping. Defaults to 99.5.

    Returns:
        np.ndarray: Normalized 2D attribution heatmap array.
    """
    if heatmap is None:
        return heatmap

    if hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
    heatmap = np.array(heatmap, dtype=np.float32)

    if heatmap.size == 0:
        return heatmap

    # Channel summation (H, W, 3) -> (H, W)
    if heatmap.ndim == 3 and heatmap.shape[-1] == 3:
        heatmap = np.sum(heatmap, axis=-1)
    elif heatmap.ndim == 4:
        heatmap = heatmap[0]
        if heatmap.ndim == 3 and heatmap.shape[-1] == 3:
            heatmap = np.sum(heatmap, axis=-1)

    # Outlier clipping and max-abs scaling
    max_val = np.percentile(np.abs(heatmap), clip_percentile)
    if max_val > 1e-9:
        heatmap = np.clip(heatmap, -max_val, max_val)
        heatmap = heatmap / max_val

    return heatmap


def postprocess_innvestigate_style(heatmap_raw: np.ndarray) -> np.ndarray:
    """Apply standard iNNvestigate visualization normalization.

    Sums over color channels, scales by maximum absolute value to [-1, 1],
    and maps linearly to [0, 1] for display.

    Args:
        heatmap_raw (np.ndarray): Raw attribution array.

    Returns:
        np.ndarray: Postprocessed 2D heatmap in range [0, 1].
    """
    if len(heatmap_raw.shape) == 3:
        heatmap = np.sum(heatmap_raw, axis=-1)
    else:
        heatmap = heatmap_raw.copy()

    vmax = np.max(np.abs(heatmap))
    if vmax > 0:
        heatmap = heatmap / vmax

    heatmap = (heatmap + 1.0) / 2.0
    return heatmap


# 3. Assemble 5x9 image grid array (45 images total)
compiled_images = np.zeros((45, 224, 224, 3))
cnt = 0

preds = model.predict(input_imgs)
top_classes = np.argmax(preds, axis=1)

for i in range(5):
    # Column 1: Original Image
    compiled_images[cnt] = org_imgs[i] / 255.0
    cnt += 1

    # Column 2: LRP Z+ (Raw)
    res_norm = postprocess_innvestigate_style(LRP_zplus[i])
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 3: LRP Z+ (Postprocessed)
    res_norm, _ = TFRP.visualize_TFRP_new(model, org_imgs[i], input_imgs[i], LRP_zplus[i], top_classes[i], alpha=0.3, sigma=1, segment_n=400, step=3)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 4: TFRP Z+ (Raw)
    res_norm = normalize_heatmap(TFRP_zplus[i], clip_percentile=99.5)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 5: TFRP Z+ (Postprocessed)
    res_norm, _ = TFRP.visualize_TFRP_new(model, org_imgs[i], input_imgs[i], TFRP_zplus[i], top_classes[i], alpha=0.3, sigma=1, segment_n=400, step=3)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 6: LRP Composite (Raw)
    res_norm = postprocess_innvestigate_style(LRP_composite[i])
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 7: LRP Composite (Postprocessed)
    res_norm, _ = TFRP.visualize_TFRP_new(model, org_imgs[i], input_imgs[i], LRP_composite[i], top_classes[i], alpha=0.3, sigma=1, segment_n=400, step=3)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 8: TFRP Composite (Raw)
    res_norm = normalize_heatmap(TFRP_composite[i], clip_percentile=99.5)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

    # Column 9: TFRP Composite (Postprocessed)
    res_norm, _ = TFRP.visualize_TFRP_new(model, org_imgs[i], input_imgs[i], TFRP_composite[i], top_classes[i], alpha=0.3, sigma=1, segment_n=400, step=3)
    compiled_images[cnt] = np.stack([res_norm, res_norm, res_norm], axis=-1)
    cnt += 1

# 4. Render and export figure
column_titles = [
    "Original Image", "LRP $Z^+$ (Raw)", r"LRP $Z^+$", r"TFRP $Z^+$ (Raw)", "TFRP $Z^+$",
    "LRP-Comp. (Raw)", r"LRP-Comp.", r"TFRP-Comp. (Raw)", "TFRP-Comp."
]

fig, axes = plt.subplots(5, 9, figsize=(15.0, 9.4))

for i, ax in enumerate(axes.flat):
    col_idx = i % 9
    if col_idx == 0:
        ax.imshow(compiled_images[i])
    else:
        ax.imshow(compiled_images[i, :, :, 0], cmap='jet')

    if i < 9:
        ax.set_title(column_titles[col_idx], fontsize=10, pad=14, fontweight='bold')

fig.suptitle("Qualitative Comparison of LRP and TFRP Variants",
             fontsize=18, y=0.98, fontweight='bold')

plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.87, wspace=0.01, hspace=0.01)

for ax in axes.flat:
    ax.axis('off')

out_fig_path = os.path.join(out_dir, 'qualitative_comparison_variants.png')
plt.savefig(out_fig_path, dpi=900, bbox_inches='tight')
plt.close()

print(f"Qualitative comparison figure saved to: '{out_fig_path}'")
