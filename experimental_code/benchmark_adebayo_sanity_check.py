###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Adebayo et al. (NeurIPS 2018) Sanity Check Benchmark.

This script executes the model parameter randomization sanity test proposed by Adebayo et al.
for attribution methods. It evaluates whether explanation maps are genuinely sensitive
to learned model weights rather than acting merely as edge detectors or data-dependent filters.

Comparison Suite:
1. Integrated Gradients
2. SmoothGrad
3. LRP (z+)
4. TFRP (Ours)

Experimental Procedure:
- Compares explanations on an original pretrained ResNet50 against a cascading randomized model
  (top 5 layers re-initialized with uniform Xavier noise).
- Computes absolute difference maps between original and randomized model attributions.
- Generates a 4x4 comparative visualization grid exported to:
  `exp_result/Fidelity_and_Sanity/sanity_check_master_comparison.png`.
"""

import os
import random
import sys
import time
from typing import Tuple, List, Dict

# Prevent UnicodeEncodeError on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import general_TFRP_v1 as TFRP
except ImportError as e:
    raise ImportError("Module 'general_TFRP_v1.py' is required.") from e

try:
    from xplique.attributions import IntegratedGradients, SmoothGrad
except ImportError as e:
    raise ImportError("Library 'xplique' is required. Please install via 'pip install xplique'.") from e


def set_seeds(seed: int = 42) -> None:
    """Fix random seeds across all libraries to enforce deterministic execution.

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


def normalize_heatmap(heatmap, clip_percentile: float = 99.5) -> np.ndarray:
    """Normalize and clip attribution maps for robust visual comparison.

    Converts input tensors to 2D numpy arrays, clips extreme outliers beyond
    the specified percentile, and applies max-abs scaling to [-1.0, 1.0].

    Args:
        heatmap (tf.Tensor or np.ndarray): Raw relevance heatmap array.
        clip_percentile (float, optional): Percentile threshold for outlier clipping. Defaults to 99.5.

    Returns:
        np.ndarray: Normalized 2D heatmap array scaled to [-1.0, 1.0].
    """
    if heatmap is None:
        return heatmap

    if hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
    heatmap = np.array(heatmap, dtype=np.float32)

    if heatmap.size == 0:
        return heatmap

    # 1. Sum across color channels if multi-channel (H, W, 3) -> (H, W)
    if heatmap.ndim == 3 and heatmap.shape[-1] == 3:
        heatmap = np.sum(heatmap, axis=-1)
    elif heatmap.ndim == 4:
        heatmap = heatmap[0]
        if heatmap.ndim == 3 and heatmap.shape[-1] == 3:
            heatmap = np.sum(heatmap, axis=-1)

    # 2. Outlier suppression and max-abs normalization
    max_val = np.percentile(np.abs(heatmap), clip_percentile)
    if max_val > 1e-9:
        heatmap = np.clip(heatmap, -max_val, max_val)
        heatmap = heatmap / max_val

    return heatmap


def load_sample_image(
    image_dir: str = "imagen_500",
    sample_idx: int = 0,
    target_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a representative ImageNet sample with preprocessing.

    Args:
        image_dir (str, optional): Directory containing sample images. Defaults to "imagen_500".
        sample_idx (int, optional): Index of the sample in the sorted file list. Defaults to 0.
        target_size (Tuple[int, int], optional): Target image resolution (H, W). Defaults to (224, 224).
        seed (int, optional): Random seed for reproducible sampling. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, str]:
            - raw_arr: Original un-preprocessed RGB image array (uint8).
            - x_prep: Preprocessed input array for ResNet50 (float32).
            - fname: Selected image filename.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: '{image_dir}'")

    all_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    random.seed(seed)
    selected_files = sorted(random.sample(all_files, len(all_files)))

    fname = selected_files[sample_idx]
    fpath = os.path.join(image_dir, fname)
    img_pil = Image.open(fpath).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
    raw_arr = np.array(img_pil, dtype=np.uint8)

    # ResNet50 preprocessing
    x_prep = tf.keras.applications.resnet50.preprocess_input(raw_arr.astype(np.float32))
    return raw_arr, x_prep, fname


def randomize_model_weights(original_model: tf.keras.Model, num_layers_to_randomize: int = 5) -> tf.keras.Model:
    """Construct a cascading randomized model by reinitializing the top layers.

    Reinitializes the weights of the top N trainable layers with uniform Xavier initialization
    to test attribution method parameter sensitivity (Adebayo et al., 2018).

    Args:
        original_model (tf.keras.Model): Pretrained source model.
        num_layers_to_randomize (int, optional): Number of top trainable layers to randomize. Defaults to 5.

    Returns:
        tf.keras.Model: Cloned model with randomized top-layer parameters.
    """
    randomized_model = tf.keras.models.clone_model(original_model)
    randomized_model.set_weights(original_model.get_weights())

    trainable_layers = [layer for layer in randomized_model.layers if len(layer.weights) > 0]
    target_layers = trainable_layers[-num_layers_to_randomize:]

    for layer in target_layers:
        weights = layer.get_weights()
        new_weights = []
        for w in weights:
            if len(w.shape) >= 2:
                limit = np.sqrt(6.0 / w.shape[0])
                new_w = np.random.uniform(-limit, limit, size=w.shape).astype(np.float32)
            else:
                new_w = np.zeros(w.shape, dtype=np.float32)
            new_weights.append(new_w)
        layer.set_weights(new_weights)

    return randomized_model


def get_tfrp_map(model: tf.keras.Model, x_prep: np.ndarray, target_idx: int) -> np.ndarray:
    """Compute and normalize a TFRP attribution map for a given model.

    Args:
        model (tf.keras.Model): Target model instance.
        x_prep (np.ndarray): Preprocessed input image array of shape (H, W, C).
        target_idx (int): Target output class index.

    Returns:
        np.ndarray: Normalized 2D attribution heatmap.
    """
    x_single = np.expand_dims(x_prep, axis=0)
    rel_map, _, _, _ = TFRP.get_relevance_map_generalized(
        model=model,
        input_image=x_single,
        target_class_idx=target_idx,
        use_logit=True,
        global_rule='z_plus',
        alpha=2.0,
        beta=1.0,
        epsilon=1e-7
    )
    r_2d = rel_map[0]
    return normalize_heatmap(r_2d, clip_percentile=99.5)


def get_xplique_map(explainer_cls, model: tf.keras.Model, x_prep: np.ndarray, target_idx: int) -> np.ndarray:
    """Compute and normalize an attribution map using an Xplique explainer.

    Args:
        explainer_cls (class): Xplique explainer class (e.g. IntegratedGradients, SmoothGrad).
        model (tf.keras.Model): Target model instance.
        x_prep (np.ndarray): Preprocessed input image array of shape (H, W, C).
        target_idx (int): Target output class index.

    Returns:
        np.ndarray: Normalized 2D attribution heatmap.
    """
    explainer = explainer_cls(model)
    x_single = np.expand_dims(x_prep, axis=0)
    targets_one_hot = tf.keras.utils.to_categorical([target_idx], num_classes=1000)
    m = explainer(x_single, targets_one_hot)[0]
    return normalize_heatmap(m, clip_percentile=99.5)


def main():
    """Execute the full Adebayo sanity check benchmark and produce the comparative figure."""
    set_seeds(42)

    print("\n" + "=" * 80)
    print("Adebayo et al. (2018) Multi-Algorithm Sanity Check Benchmark")
    print("=" * 80)

    # 1. Load pretrained model and construct randomized model
    print("Loading pretrained ResNet50 and generating randomized model...")
    model = tf.keras.applications.ResNet50(weights='imagenet')
    randomized_model = randomize_model_weights(model, num_layers_to_randomize=5)
    print("Model preparation complete.")

    # 2. Load representative sample image
    sample_index = 3
    raw_img, x_prep, fname = load_sample_image(image_dir="imagen_500", sample_idx=sample_index, seed=42)
    pred_orig = model.predict(np.expand_dims(x_prep, axis=0), verbose=0)
    target_idx = int(np.argmax(pred_orig[0]))
    print(f"Sample image loaded: '{fname}' (Sample Index: {sample_index}, Target Class Index: {target_idx})")

    # 3. Extract attribution maps for original and randomized models
    heatmap_dir = os.path.join("exp_result", "Fidelity_and_Sanity", "lrp_heatmap")
    orig_lrp_path = os.path.join(heatmap_dir, "lrp_zplus_orig_sample3.npy")
    rand_lrp_path = os.path.join(heatmap_dir, "lrp_zplus_rand_sample3.npy")

    methods = ['Integrated Gradients', 'SmoothGrad', 'LRP (z+)', 'TFRP (Ours)']
    maps_orig = {}
    maps_rand = {}

    print("\nExtracting attribution maps (Original vs. Randomized)...")

    # (1) Integrated Gradients
    print("  [1/4] Computing Integrated Gradients...")
    maps_orig['Integrated Gradients'] = get_xplique_map(IntegratedGradients, model, x_prep, target_idx)
    maps_rand['Integrated Gradients'] = get_xplique_map(IntegratedGradients, randomized_model, x_prep, target_idx)

    # (2) SmoothGrad
    print("  [2/4] Computing SmoothGrad...")
    maps_orig['SmoothGrad'] = get_xplique_map(SmoothGrad, model, x_prep, target_idx)
    maps_rand['SmoothGrad'] = get_xplique_map(SmoothGrad, randomized_model, x_prep, target_idx)

    # (3) LRP (z+)
    print("  [3/4] Loading/Computing LRP (z+)...")
    if os.path.exists(orig_lrp_path) and os.path.exists(rand_lrp_path):
        m_orig = np.load(orig_lrp_path)
        m_rand = np.load(rand_lrp_path)
        maps_orig['LRP (z+)'] = normalize_heatmap(m_orig, clip_percentile=99.5)
        maps_rand['LRP (z+)'] = normalize_heatmap(m_rand, clip_percentile=99.5)
        print("    Loaded precomputed iNNvestigate LRP .npy maps.")
    else:
        print("    Precomputed NPY files not found; falling back to real-time computation.")
        maps_orig['LRP (z+)'] = get_tfrp_map(model, x_prep, target_idx)
        maps_rand['LRP (z+)'] = get_tfrp_map(randomized_model, x_prep, target_idx)

    # (4) TFRP (Ours)
    print("  [4/4] Computing TFRP (Ours)...")
    maps_orig['TFRP (Ours)'] = get_tfrp_map(model, x_prep, target_idx)
    maps_rand['TFRP (Ours)'] = get_tfrp_map(randomized_model, x_prep, target_idx)

    # 4. Render 4x4 Grid Visualization
    print("\nRendering 4x4 comparative sanity check grid...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 15))

    col_titles = [
        "Original Input Image",
        "Original Model Map",
        "Randomized Model Map",
        "Absolute Difference"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row_idx, method_name in enumerate(methods):
        m_orig = maps_orig[method_name]
        m_rand = maps_rand[method_name]
        diff_map = normalize_heatmap(np.abs(m_orig - m_rand), clip_percentile=99.5)

        # Col 1: Original Image
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_ylabel(method_name, fontsize=12, fontweight='bold')
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 2: Original Model Map
        axes[row_idx, 1].imshow(m_orig, cmap='seismic', vmin=-1.0, vmax=1.0)
        axes[row_idx, 1].axis('off')

        # Col 3: Randomized Model Map
        axes[row_idx, 2].imshow(m_rand, cmap='seismic', vmin=-1.0, vmax=1.0)
        axes[row_idx, 2].axis('off')

        # Col 4: Absolute Difference Map
        axes[row_idx, 3].imshow(diff_map, cmap='inferno', vmin=0.0, vmax=1.0)
        axes[row_idx, 3].axis('off')

    plt.tight_layout()

    output_dir = os.path.join("exp_result", "Fidelity_and_Sanity")
    os.makedirs(output_dir, exist_ok=True)
    master_png_path = os.path.join(output_dir, "sanity_check_master_comparison.png")
    plt.savefig(master_png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMaster Sanity Check 4x4 grid visualization saved to:\n  '{master_png_path}'")


if __name__ == '__main__':
    main()
