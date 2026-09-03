###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""LRP Sensitivity (AverageStability) Standalone Benchmark.

This script evaluates the sensitivity (stability under input perturbations) of conventional
LRP (z+) using iNNvestigate on 500 ImageNet samples with ResNet50.

Methodology:
- Injects Gaussian noise (std=0.05, 5 iterations) into each input image.
- Computes the L2 distance between the original attribution map and the perturbed map.
- Exports results to `exp_result/Fidelity_and_Sanity/sensitivity_lrp_only.csv`.
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

# Enforce static graph mode for iNNvestigate
tf.compat.v1.disable_eager_execution()
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
import innvestigate
import innvestigate.utils.keras.graph as kgraph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    tf.compat.v1.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def load_imagenet_500(
    image_dir: str = "imagen_500",
    num_samples: int = 500,
    target_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> Tuple[np.ndarray, List[str]]:
    """Sample and preprocess 500 ImageNet images.

    Args:
        image_dir (str, optional): Image folder path. Defaults to "imagen_500".
        num_samples (int, optional): Number of images to sample. Defaults to 500.
        target_size (Tuple[int, int], optional): Target resolution (H, W). Defaults to (224, 224).
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[np.ndarray, List[str]]: Preprocessed image tensor array and list of filenames.
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
        x_prep = preprocess_input(x_arr)
        preprocessed_list.append(x_prep)

    return np.array(preprocessed_list, dtype=np.float32), selected_files


def calculate_manual_sensitivity(
    analyzer,
    images: np.ndarray,
    noise_std: float = 0.05,
    iterations: int = 5
) -> float:
    """Calculate AverageStability sensitivity score by measuring L2 attribution shifts under noise.

    Args:
        analyzer: iNNvestigate analyzer instance.
        images (np.ndarray): Array of input image samples.
        noise_std (float, optional): Gaussian noise standard deviation. Defaults to 0.05.
        iterations (int, optional): Number of noise draws per image. Defaults to 5.

    Returns:
        float: Mean L2 attribution shift across all samples.
    """
    num_samples = len(images)
    sensitivity_scores = []

    print(f"Evaluating sensitivity under Gaussian noise (std={noise_std}, iterations={iterations}) for {num_samples} samples...")

    for idx, img in enumerate(images):
        img_input = np.expand_dims(img, axis=0)

        # Original attribution map
        orig_map = analyzer.analyze(img_input)[0]
        if orig_map.ndim == 3 and orig_map.shape[-1] == 3:
            orig_map = np.sum(orig_map, axis=-1)

        diffs = []
        for _ in range(iterations):
            noise = np.random.normal(0, noise_std, img_input.shape)
            noisy_img = img_input + noise

            noisy_map = analyzer.analyze(noisy_img)[0]
            if noisy_map.ndim == 3 and noisy_map.shape[-1] == 3:
                noisy_map = np.sum(noisy_map, axis=-1)

            diff = np.linalg.norm(orig_map - noisy_map)
            diffs.append(diff)

        sensitivity_scores.append(np.mean(diffs))

        if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
            print(f"  Progress: {idx + 1}/{num_samples} samples processed (Current Mean Sensitivity: {np.mean(sensitivity_scores):.6f})")

    return float(np.mean(sensitivity_scores))


def main():
    """Execute standalone LRP sensitivity benchmark."""
    set_seeds(42)

    print("\n" + "=" * 80)
    print("iNNvestigate LRP (z+) Standalone Sensitivity Evaluation")
    print("=" * 80)

    print("Loading ResNet50 model...")
    model = ResNet50(weights='imagenet')
    model_wo_sm = kgraph.model_wo_softmax(model)
    analyzer_zplus = innvestigate.create_analyzer("lrp.z_plus", model_wo_sm)
    print("iNNvestigate lrp.z_plus analyzer configured.")

    images_preprocessed, selected_files = load_imagenet_500(num_samples=500, seed=42)

    start_time = time.time()
    stab_score = calculate_manual_sensitivity(
        analyzer=analyzer_zplus,
        images=images_preprocessed,
        noise_std=0.05,
        iterations=5
    )
    elapsed = time.time() - start_time

    results_data = [{
        'Method': 'LRP (z+) (iNNvestigate Pure)',
        'Sensitivity (Stability)': stab_score,
        'Time (s)': elapsed
    }]

    df_results = pd.DataFrame(results_data)
    output_dir = os.path.join("exp_result", "Fidelity_and_Sanity")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sensitivity_lrp_only.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("iNNvestigate LRP (z+) Sensitivity Summary")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"\nReport saved to: '{csv_path}'")


if __name__ == '__main__':
    main()
