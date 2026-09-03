###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""LRP Heatmap Precomputation and Export for Fidelity Benchmarks.

This script performs Stage 1 of the LRP (z+) fidelity benchmark:
- Computes LRP (z+) attributions on 500 ImageNet samples using iNNvestigate.
- Exports attributions to `exp_result/Fidelity_and_Sanity/lrp_heatmap/lrp_zplus_maps_500.npy`.
- Generates and exports randomized model heatmaps for Adebayo sanity check evaluation.
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
import tensorflow as tf
from PIL import Image

# Disable TF eager execution for iNNvestigate static graph computation
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
    """Sample and preprocess ImageNet images for ResNet50.

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

    images_preprocessed = np.array(preprocessed_list, dtype=np.float32)
    print(f"Loaded {len(selected_files)} images. Shape: {images_preprocessed.shape}")
    return images_preprocessed, selected_files


def get_lrp_zplus_analyzer(model: keras.Model):
    """Build iNNvestigate lrp.z_plus analyzer stripping softmax activation.

    Args:
        model (keras.Model): Source Keras model.

    Returns:
        innvestigate.analyzer: Configured LRP-z+ analyzer.
    """
    model_wo_sm = kgraph.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer("lrp.z_plus", model_wo_sm)
    return analyzer


def compute_lrp_maps(analyzer, inputs: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Compute LRP attribution maps in batches.

    Args:
        analyzer: iNNvestigate analyzer instance.
        inputs (np.ndarray): Batch of preprocessed images.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        np.ndarray: Concatenated attribution array of shape (N, H, W, 1).
    """
    num_samples = len(inputs)
    maps_list = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_inputs = inputs[start_idx:end_idx]
        analysis_batch = analyzer.analyze(batch_inputs)

        # Sum over 3 color channels if present
        if analysis_batch.ndim == 4 and analysis_batch.shape[-1] == 3:
            rel_map = np.sum(analysis_batch, axis=-1, keepdims=True)
        else:
            rel_map = analysis_batch
        maps_list.append(rel_map)

    all_maps = np.concatenate(maps_list, axis=0)
    return all_maps


def randomize_model_top_weights(original_model: keras.Model, num_layers: int = 5) -> keras.Model:
    """Randomize the weights of the top N layers for the Adebayo sanity check.

    Args:
        original_model (keras.Model): Pretrained source model.
        num_layers (int, optional): Number of top layers to randomize. Defaults to 5.

    Returns:
        keras.Model: Model clone with randomized top layers.
    """
    print(f"\nRandomizing weights of top {num_layers} layers...")
    config = original_model.get_config()
    randomized_model = keras.Model.from_config(config)
    randomized_model.set_weights(original_model.get_weights())

    trainable_layers = [l for l in randomized_model.layers if len(l.weights) > 0]
    target_layers = trainable_layers[-num_layers:]

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
        print(f"  Randomized layer: {layer.name}")

    return randomized_model


def main():
    """Execute Stage 1 LRP heatmap precomputation and file export."""
    set_seeds(42)

    save_dir = os.path.join("exp_result", "Fidelity_and_Sanity", "lrp_heatmap")
    os.makedirs(save_dir, exist_ok=True)

    print("Loading pretrained ResNet50...")
    model = ResNet50(weights='imagenet')
    print("ResNet50 loaded.")

    images_preprocessed, selected_files = load_imagenet_500(
        image_dir="imagen_500",
        num_samples=500,
        seed=42
    )

    print("\nInstantiating iNNvestigate lrp.z_plus analyzer...")
    analyzer_zplus = get_lrp_zplus_analyzer(model)

    print("\nComputing LRP (z+) heatmaps for 500 samples...")
    start_time = time.time()
    lrp_maps = compute_lrp_maps(analyzer_zplus, images_preprocessed, batch_size=32)
    elapsed = time.time() - start_time
    print(f"LRP heatmaps computed in {elapsed:.2f}s. Shape: {lrp_maps.shape}")

    # Export to .npy
    maps_path = os.path.join(save_dir, "lrp_zplus_maps_500.npy")
    files_path = os.path.join(save_dir, "selected_files.npy")
    np.save(maps_path, lrp_maps)
    np.save(files_path, selected_files)
    print(f"Saved 500 LRP attribution maps to: '{maps_path}'")

    # Sanity check attribution maps for Sample 3
    print("\nExtracting Adebayo sanity check heatmaps (Sample Index = 3)...")
    sample_idx = 3
    sample_img = images_preprocessed[sample_idx:sample_idx + 1]

    randomized_model = randomize_model_top_weights(model, num_layers=5)
    analyzer_rand = get_lrp_zplus_analyzer(randomized_model)

    map_orig = analyzer_zplus.analyze(sample_img)[0]
    map_rand = analyzer_rand.analyze(sample_img)[0]

    np.save(os.path.join(save_dir, "lrp_zplus_orig_sample3.npy"), map_orig)
    np.save(os.path.join(save_dir, "lrp_zplus_rand_sample3.npy"), map_rand)
    print("Saved Sanity Check NPY heatmaps for sample 3.")


if __name__ == '__main__':
    main()
