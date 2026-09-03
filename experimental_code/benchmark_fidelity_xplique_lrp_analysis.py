###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""LRP Fidelity Benchmark via Precomputed Heatmaps (Xplique Analysis).

This script performs Stage 2 of the LRP (z+) fidelity benchmark:
- Loads precomputed LRP (z+) attribution heatmaps saved as `.npy` files (`lrp_zplus_maps_500.npy`).
- Evaluates Xplique quantitative fidelity metrics: Deletion AUC, Insertion AUC, and MuFidelity.
- Exports results to `exp_result/Fidelity_and_Sanity/fidelity_metrics_summary_lrp.csv`.
"""

import os
import random
import sys
import time
from typing import Tuple, List

# Prevent UnicodeEncodeError on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import xplique
    from xplique.metrics import Deletion, Insertion, MuFidelity
except ImportError as e:
    raise ImportError("Library 'xplique' is required. Please install via 'pip install xplique'.") from e


def set_seeds(seed: int = 42) -> None:
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


def load_imagenet_500(
    image_dir: str = "imagen_500",
    num_samples: int = 500,
    target_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """Sample and preprocess ImageNet validation images.

    Args:
        image_dir (str, optional): Directory containing sample images. Defaults to "imagen_500".
        num_samples (int, optional): Number of images to sample. Defaults to 500.
        target_size (Tuple[int, int], optional): Target resolution (H, W). Defaults to (224, 224).
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, List[str]]:
            - images_preprocessed: Preprocessed NumPy array of shape (N, 224, 224, 3).
            - selected_files: List of sampled filenames.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: '{image_dir}'")

    all_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if len(all_files) < num_samples:
        num_samples = len(all_files)

    random.seed(seed)
    selected_files = sorted(random.sample(all_files, num_samples))

    preprocessed_list = []
    for fname in selected_files:
        fpath = os.path.join(image_dir, fname)
        img = Image.open(fpath).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
        x_arr = np.array(img, dtype=np.float32)
        x_prep = tf.keras.applications.resnet50.preprocess_input(x_arr)
        preprocessed_list.append(x_prep)

    images_preprocessed = np.array(preprocessed_list, dtype=np.float32)
    return images_preprocessed, selected_files


def main():
    """Execute Stage 2 Xplique fidelity evaluation for precomputed LRP heatmaps."""
    set_seeds(42)

    heatmap_dir = os.path.join("exp_result", "Fidelity_and_Sanity", "lrp_heatmap")
    maps_path = os.path.join(heatmap_dir, "lrp_zplus_maps_500.npy")

    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Precomputed LRP heatmap file not found: '{maps_path}'")

    print(f"Loading precomputed LRP heatmaps from: '{maps_path}'")
    lrp_maps = np.load(maps_path)
    print(f"Loaded LRP (z+) heatmaps. Shape: {lrp_maps.shape}")

    print("\nLoading pretrained ResNet50 model...")
    model = tf.keras.applications.ResNet50(weights='imagenet')
    print("ResNet50 loaded.")

    images_preprocessed, selected_files = load_imagenet_500(
        image_dir="imagen_500",
        num_samples=500,
        seed=42
    )

    print("Computing target predictions on 500 samples...")
    preds = model.predict(images_preprocessed, batch_size=32)
    target_indices = np.argmax(preds, axis=1)
    targets_one_hot = tf.keras.utils.to_categorical(target_indices, num_classes=1000)

    batch_size = 16
    print("\n" + "=" * 80)
    print("Xplique LRP (z+) Quantitative Fidelity Evaluation")
    print("=" * 80)

    start_time = time.time()

    print("Evaluating Deletion AUC [1/3]...")
    deletion_metric = Deletion(model, images_preprocessed, targets_one_hot, batch_size=batch_size)
    del_score = float(deletion_metric(lrp_maps))

    print("Evaluating Insertion AUC [2/3]...")
    insertion_metric = Insertion(model, images_preprocessed, targets_one_hot, batch_size=batch_size)
    ins_score = float(insertion_metric(lrp_maps))

    print("Evaluating MuFidelity [3/3]...")
    mufid_metric = MuFidelity(model, images_preprocessed, targets_one_hot, batch_size=batch_size)
    mufid_score = float(mufid_metric(lrp_maps))

    elapsed = time.time() - start_time

    results_data = [{
        'Method': 'LRP (z+)',
        'Deletion (AUC)': del_score,
        'Insertion (AUC)': ins_score,
        'MuFidelity': mufid_score,
        'Sensitivity (Stability)': 'Calculated in benchmark_fidelity_lrp_sensitivity_only.py',
        'Time (s)': elapsed
    }]

    df_results = pd.DataFrame(results_data)

    output_dir = os.path.join("exp_result", "Fidelity_and_Sanity")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "fidelity_metrics_summary_lrp.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("LRP (z+) Xplique Quantitative Summary")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"\nReport saved to: '{csv_path}'")


if __name__ == '__main__':
    main()
