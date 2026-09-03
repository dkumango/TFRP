###########################################################################
# Selected XAI methods   (tensorflow 2.x based)
#
# [Method]
# - TFRP
# - GradCAM (GC)
# - integrated gradients (IG)
# - guided_backpropergation (GB)
# - lime (LIME)
# - smooth Grad (SG)
# - rise (RISE)
#
# [Metric]
# - insertion, deletion
# - average stability 
# - Efficiency
# - Sparsity
# - mufidelity
###########################################################################
import time
import numpy as np
import tensorflow as tf

import os
os.chdir('/content/drive/MyDrive/Colab Notebooks/TFRP_project/github_publish/')
import TFRP_v5 as TFRP

################################################################################
# XAI methods
################################################################################
def get_map_tfrp(model, image, class_index):

    model_type = TFRP.check_model_type(model)
    if model_type == "Functional":
        res = TFRP.get_relevance_map_graph(model, image, target_class_idx=class_index)
    else:
        res = TFRP.get_relevance_map_sequential(model, image, target_class_idx=class_index)
    return res

####################################################################
from xplique.attributions import GradCAM

def get_map_gradcam_tf(explainer, model, image_arr, class_index, layer_name=None):
    # shape: (1, 224, 224, 3)
    img_input = np.expand_dims(image_arr, axis=0).astype(np.float32)

    # one-hot encoding
    num_classes = model.output_shape[-1]
    target_one_hot = tf.one_hot([class_index], num_classes) # Shape: (1, 1000)

    # (1, 1000) - (1, 224, 224, 3)
    explanation = explainer(img_input, target_one_hot)

    # explanation : (1, H, W, 1) 
    result = explanation[0]
    result = np.squeeze(result)

    # 0~1 Normalization
    res_min, res_max = np.min(result), np.max(result)
    if res_max - res_min > 0:
        result = (result - res_min) / (res_max - res_min)
    else:
        result = np.zeros_like(result) # Edge case: No gradient detected

    return result # (H,W)

#################################################################

from xplique.attributions import IntegratedGradients, GuidedBackprop

def get_map_integrated_gradients_tf(explainer, model, images, class_index, n_steps=50):
    """
    Args:
        images: (N, H, W, 3) or (H, W, 3) numpy array
        class_index: target class index for each image (N,) or a single integer
    """
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    if isinstance(class_index, int) or isinstance(class_index, np.integer):
        class_index = [class_index]

    # one-hot encoding
    num_classes = model.output_shape[-1]
    # (1, 1000) one-hot vector 
    target_one_hot = tf.one_hot([class_index], num_classes)

    explanations = explainer(images, target_one_hot)

    # Post-processing: Channel aggregation
    heatmap = np.sum(explanations, axis=-1)

    # 0~1 normalization
    for i in range(len(heatmap)):
        max_val = np.max(np.abs(heatmap[i]))
        if max_val > 0:
            heatmap[i] = heatmap[i] / max_val

    return heatmap if len(heatmap) > 1 else heatmap[0]

#################################################################

def get_map_guided_backprop_tf(explainer, model, image, class_index):

    img_input = np.expand_dims(image, axis=0).astype(np.float32)

    if np.max(img_input) > 1.0:
        img_input = img_input / np.float32(255.0)

    img_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    num_classes = model.output_shape[-1]
    target_one_hot = tf.one_hot([int(class_index)], num_classes, dtype=tf.float32)

    explanation = explainer(img_tensor, target_one_hot)

    # Extract result and convert to NumPy array
    result = explanation[0]
    if hasattr(result, "numpy"):
        result = result.numpy()

    return result

################################################################
from xplique.attributions import Lime

def get_map_lime_tf(explainer, model,img_array, class_index):

    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    if np.max(img_input) > 1.0:
        img_input /= 255.0

    num_classes = model.output_shape[-1]
    target_one_hot = tf.one_hot([class_index], num_classes)

    explanation = explainer(img_input, target_one_hot)

    result = np.sum(np.abs(explanation[0]), axis=-1)

    if np.max(result) > 0:
        result = result / np.max(result)

    return result

##################################################################
from xplique.attributions import SmoothGrad

def get_map_smoothgrad_tf(explainer, model, img_array,class_index, nb_samples=50, noise=0.15):

    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    if np.max(img_input) > 1.0:
        img_input /= 255.0

    num_classes = model.output_shape[-1]
    target_one_hot = tf.one_hot([class_index], num_classes) # result Shape: (1, 1000)

    explanation = explainer(img_input, target_one_hot)

    result = np.sum(np.abs(explanation[0]), axis=-1)

    # 0~1 normalization
    if np.max(result) > 0:
        result = result / np.max(result)

    return result

#####################################################################
from xplique.attributions import Rise

def get_map_rise_tf(explainer, model, img_array, class_index, nb_samples=1000, grid_size=7):

    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)
    if np.max(img_input) > 1.0:
        img_input /= 255.0

    num_classes = model.output_shape[-1]

    target_idx = tf.cast([class_index], tf.int32)
    target_one_hot = tf.one_hot(target_idx, num_classes)

    explanation = explainer(img_input, target_one_hot)

    result = explanation[0]

    # 0~1 normalization
    if np.max(result) > 0:
        result = result / np.max(result)

    return result

#############################################################################
# Evaluation metrics 
#############################################################################

from xplique.metrics import Deletion, Insertion
import cv2

#######################################################################
def apply_blur(images, ksize=11, sigma=5.0):
    """
    images: (N, H, W, 3), float32
    """
    blurred = []
    for img in images:
        b = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        blurred.append(b)
    return np.stack(blurred, axis=0)

#######################################################################
def measure_fidelity_auc(model, images, heatmaps, class_indices, steps=18,  baseline_type="mean"):
    """
    Deletion (Lower is better),  Insertion (Higher is better) 
    images: (N, H, W, 3), heatmaps: (N, H, W), targets: (N, Num_Classes) 
    """
    img_input = images.astype(np.float32)

    if len(heatmaps.shape) == 4:
        heatmaps = np.sum(np.abs(heatmaps), axis=-1)
    heatmap_input = heatmaps.astype(np.float32)

    num_classes = model.output_shape[-1]
    targets_one_hot = tf.one_hot(class_indices, num_classes)

    if baseline_type == "min":
        baseline_val = float(np.min(img_input))
        img_for_metric = img_input
    elif baseline_type == "mean":
        baseline_val = float(np.mean(img_input))
        img_for_metric = img_input
    elif baseline_type == "blur":
        img_for_metric = apply_blur(
            img_input,
            ksize=11,
            sigma=5.0
        )
        baseline_val = float(np.min(img_for_metric))
        img_for_metric = img_input

    # Higher steps = finer curves, but slower speed. (Recommended: 18–30)
    deletion_metric = Deletion(
        model,
        img_for_metric,
        targets_one_hot,
        steps=steps,
        baseline_mode=baseline_val
    )
    
    deletion_auc = deletion_metric.evaluate(heatmap_input)

    insertion_metric = Insertion(
        model,
        img_for_metric,
        targets_one_hot,
        steps=steps,
        baseline_mode=baseline_val
    )
    insertion_auc = insertion_metric.evaluate(heatmap_input)

    return {
        "deletion_auc": np.mean(deletion_auc),
        "insertion_auc": np.mean(insertion_auc)
    }


#####################################
from xplique.metrics import AverageStability

def measure_stability(model, images, targets, explainer, nb_samples=10, radius=0.1):
    """
    AverageStability (Higher is better):
    Measures heatmap stability against input noise.
    """
    targets_arr = np.array(targets)
    if len(targets_arr.shape) == 1:
        targets_arr = np.expand_dims(targets_arr, axis=-1)

    original_explanations = explainer(images, targets_arr)

    stability_metric = AverageStability(
        model,
        images,
        targets_arr, 
        nb_samples=nb_samples,
        radius=radius
    )

    score = stability_metric.evaluate(explainer, base_explanations=original_explanations)

    return score

###############################################################
def measure_sparsity(heatmaps):
    """
    Measures how well the heatmap focuses on the core regions. (Higher is better)
    Same implementation of xplique's Sparsity metric.
    heatmaps: (N, H, W) or (H, W) numpy array
    """
    if heatmaps.ndim == 2:
        heatmaps = np.expand_dims(heatmaps, axis=0)

    N, H, W = heatmaps.shape
    flatten_maps = np.abs(heatmaps.reshape(N, -1))


    max_vals = np.max(flatten_maps, axis=1, keepdims=True)
    max_vals = np.where(max_vals == 0, 1, max_vals)
    norm_maps = flatten_maps / max_vals

    n = H * W
    l1_norm = np.sum(norm_maps, axis=1)
    l2_norm = np.sqrt(np.sum(norm_maps**2, axis=1))

    l2_norm = np.where(l2_norm == 0, 1, l2_norm)

    sparsity_scores = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)

    return np.mean(sparsity_scores)


# 4. Efficiency (Performance)
# 알고리즘이 히트맵을 생성하는 데 걸리는 시간을 측정합니다.
###############################################################

def measure_efficiency(explainer, m_name, model, images, targets):
    """
    Measures the time required for the algorithm to generate a heatmap (Lower is better)
    """
    if m_name == "TFRP":
        r_method = get_map_tfrp    # get_relevance_map_graph_patched # get_map_tfrp
    elif m_name == "GC":
        r_method = get_map_gradcam_tf
    elif m_name == "IG":
        r_method = get_map_integrated_gradients_tf
    elif m_name == "GB":
        r_method = get_map_guided_backprop_tf
    elif m_name == "LIME":
        r_method = get_map_lime_tf
    elif m_name == "SG":
        r_method = get_map_smoothgrad_tf
    elif m_name == "RISE":
        r_method = get_map_rise_tf

    process_time = []
    for i in range(len(images)):
      if m_name == "TFRP":
          start_time = time.time()
          _ = r_method(model, images[i], targets[i])
          end_time = time.time()
          process_time.append(end_time - start_time)
      else:
          start_time = time.time()
          _ = r_method(explainer, model, images[i].astype(np.float32), int(targets[i]))
          end_time = time.time()
          process_time.append(end_time - start_time)

    return np.mean(process_time)