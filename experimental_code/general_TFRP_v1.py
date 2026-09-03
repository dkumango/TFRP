###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Generalized Token/Feature Relevance Propagation (TFRP) Framework.

This module implements the core Generalized Relevance Propagation algorithms for deep neural networks.
It supports standard LRP attribution rules (z+, epsilon, alpha-beta) as well as composite presets
across modern CNN architectures (e.g. ResNet, MobileNet, Inception, VGG).

Key Features:
- Layer-aware contribution distribution across Dense, Conv2D, and DepthwiseConv2D layers.
- Defensive spatial shape alignment for cross-layer feature maps.
- Strict conservation tracking to monitor attribution leakage across network layers.
- Post-processing and visualization utilities (Grad-CAM gating, superpixel pooling, and heatmap refinement).
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2


##############################################################################
# 1. Generalized Contribution Distribution Function
##############################################################################
def distribute_relevance_gen(layer, p_i, R_j, rule='z_plus', alpha=2.0, beta=1.0, epsilon=1e-1):
    """Distribute upper-layer relevance R_j to lower-layer activations p_i.

    Supports epsilon, alpha-beta, and z_plus attribution rules with defensive spatial
    shape alignment between layers.

    Args:
        layer (tf.keras.layers.Layer): The neural network layer undergoing relevance backpropagation.
        p_i (tf.Tensor): Lower-layer forward activation tensor of shape (batch, ..., channels).
        R_j (tf.Tensor): Upper-layer relevance tensor to distribute.
        rule (str, optional): Attribution rule ('z_plus', 'epsilon', 'alpha_beta'). Defaults to 'z_plus'.
        alpha (float, optional): Positive relevance weighting factor for alpha-beta rule. Defaults to 2.0.
        beta (float, optional): Negative relevance weighting factor for alpha-beta rule. Defaults to 1.0.
        epsilon (float, optional): Small numerical stabilizer for epsilon rule. Defaults to 1e-1.

    Returns:
        tf.Tensor: Distributed relevance tensor allocated to the lower-layer activation p_i.
    """
    # 1. Weight extraction based on layer type
    if hasattr(layer, 'depthwise_kernel'):
        w = layer.depthwise_kernel
    else:
        w = layer.kernel

    # Define forward operation according to layer type
    def forward_op(input_tensor, weights):
        layer_name = layer.__class__.__name__
        if 'Dense' in layer_name:
            return tf.matmul(input_tensor, weights)
        elif 'Conv2D' in layer_name:
            return tf.nn.conv2d(input_tensor, weights, strides=layer.strides, padding=layer.padding.upper())
        elif 'DepthwiseConv2D' in layer_name:
            strides = [1, layer.strides[0], layer.strides[1], 1]
            return tf.nn.depthwise_conv2d(
                input_tensor, weights,
                strides=strides,
                padding=layer.padding.upper()
            )
        return input_tensor

    # 2. Forward response computation for shape alignment
    if rule == 'epsilon':
        z_sample = forward_op(p_i, w)
    else:
        z_sample = forward_op(p_i, tf.maximum(w, 0.0))

    # Defensive Shape Alignment:
    # Automatically resize R_j if its spatial resolution differs from the current layer's z_sample.
    if len(R_j.shape) == 4 and len(z_sample.shape) == 4:
        if R_j.shape[1:3] != z_sample.shape[1:3]:
            R_j = tf.image.resize(R_j, z_sample.shape[1:3], method='nearest')

    # 3. Rule-specific relevance redistribution logic

    # [Case A] LRP-Epsilon: Full weights with stabilization factor
    if rule == 'epsilon':
        z_raw = forward_op(p_i, w)
        # Ensure denominator does not evaluate to zero (1e-9 prevents underflow)
        z = z_raw + tf.sign(z_raw + 1e-9) * epsilon

        s = R_j / z
        with tf.GradientTape() as tape:
            tape.watch(p_i)
            z_prime = forward_op(p_i, w)
            tmp = tf.reduce_sum(z_prime * s)
        return p_i * tape.gradient(tmp, p_i)

    # [Case B] LRP-AlphaBeta: Separate control over positive and negative contributions
    elif rule == 'alpha_beta':
        w_plus = tf.maximum(w, 0.0)
        w_minus = tf.minimum(w, 0.0)

        # Alpha pass (positive contribution)
        z_p = forward_op(p_i, w_plus) + 1e-9
        s_p = R_j / z_p
        with tf.GradientTape() as tape_p:
            tape_p.watch(p_i)
            z_p_prime = forward_op(p_i, w_plus)
            tmp_p = tf.reduce_sum(z_p_prime * s_p)
        R_alpha = p_i * tape_p.gradient(tmp_p, p_i)

        # Beta pass (negative contribution)
        z_m = forward_op(p_i, w_minus) - 1e-9
        s_m = R_j / z_m
        with tf.GradientTape() as tape_m:
            tape_m.watch(p_i)
            z_m_prime = forward_op(p_i, w_minus)
            tmp_m = tf.reduce_sum(z_m_prime * s_m)
        R_beta = p_i * tape_m.gradient(tmp_m, p_i)

        return alpha * R_alpha - beta * R_beta

    # [Case C] LRP-z+ (Default): Stable distribution over excitatory weights
    else:  # z_plus
        w_plus = tf.maximum(w, 1e-9)
        z = forward_op(p_i, w_plus) + 1e-9
        s = R_j / z
        with tf.GradientTape() as tape:
            tape.watch(p_i)
            z_prime = forward_op(p_i, w_plus)
            tmp = tf.reduce_sum(z_prime * s)
        return p_i * tape.gradient(tmp, p_i)


##############################################################################
# 2. Functional Graph Traversal with Composite Rule Support
##############################################################################
def get_relevance_map_generalized(model, input_image, target_class_idx=None, use_logit=True,
                                  global_rule='z_plus', composite_preset=None,
                                  alpha=2.0, beta=1.0, epsilon=1e-1):
    """Compute generalized relevance attribution maps and track network-wide conservation.

    Performs reverse layer-by-layer traversal through Keras functional graphs,
    handling modern architectural blocks including Conv2D, Dense, Add (residuals),
    Concatenate, ZeroPadding2D, Flatten, and Pooling layers.

    Args:
        model (tf.keras.Model): Target Keras neural network model.
        input_image (tf.Tensor or np.ndarray): Input image tensor of shape (H, W, C) or (1, H, W, C).
        target_class_idx (int, optional): Target class index for explanation. If None, uses top-1 predicted class.
        use_logit (bool, optional): If True, starts propagation from pre-softmax un-biased logit scores. Defaults to True.
        global_rule (str, optional): Default propagation rule ('z_plus', 'epsilon', 'alpha_beta'). Defaults to 'z_plus'.
        composite_preset (dict, optional): Layer-type specific rule overrides, e.g. {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}.
        alpha (float, optional): Alpha weight for alpha-beta rule. Defaults to 2.0.
        beta (float, optional): Beta weight for alpha-beta rule. Defaults to 1.0.
        epsilon (float, optional): Epsilon stabilizer for epsilon rule. Defaults to 1e-1.

    Returns:
        tuple: (final_input_R, initial_sum, final_sum, total_leakage_pct)
            - final_input_R (np.ndarray): Pixel-level relevance attribution map at input resolution.
            - initial_sum (float): Target class output logit/score initiating relevance propagation.
            - final_sum (float): Sum of total relevance reaching the input layer before correction.
            - total_leakage_pct (float): Total percentage conservation leakage through the network.
    """
    # 1. Preprocessing input data into a 4D float32 tensor
    if not tf.is_tensor(input_image):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    if len(input_image.shape) == 3:
        input_image = tf.expand_dims(input_image, axis=0)

    # 2. Extract forward activations across all layers (tensor ID mapping)
    layer_outputs = [l.output for l in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model(input_image)
    act_dict = {id(l.output): act for l, act in zip(model.layers, activations)}

    # 3. Initialize relevance (R) from target class score
    if use_logit:
        # Locate the final Dense layer for logit extraction without bias
        target_layer_idx = -1
        for idx in range(len(model.layers) - 1, -1, -1):
            if 'Dense' in model.layers[idx].__class__.__name__:
                target_layer_idx = idx
                break

        if target_layer_idx == -1:
            raise ValueError("No Dense layer found in model for logit extraction.")

        target_layer = model.layers[target_layer_idx]
        input_to_dense_model = tf.keras.Model(inputs=model.input, outputs=target_layer.input)
        input_to_dense = input_to_dense_model(input_image)

        # Exclude bias and calculate logit via weight dot-product to satisfy exact conservation
        weights = target_layer.get_weights()[0]
        preds = tf.matmul(input_to_dense, weights)

        start_idx = target_layer_idx
        target_id = id(target_layer.output)
    else:
        preds = model(input_image)
        start_idx = len(model.layers) - 1
        target_id = id(model.layers[-1].output)

    if target_class_idx is None:
        target_class_idx = tf.argmax(preds[0])

    # Initialize relevance with target class score
    R_start = tf.one_hot([target_class_idx], preds.shape[-1]) * preds
    initial_sum = tf.reduce_sum(R_start).numpy()
    rel_dict = {target_id: R_start}

    # 4. Reverse graph traversal for relevance backpropagation
    for i in range(start_idx, -1, -1):
        layer = model.layers[i]
        layer_output_id = id(layer.output)

        if layer_output_id not in rel_dict:
            continue

        current_R = rel_dict[layer_output_id]
        sum_upper = tf.reduce_sum(current_R).numpy()

        inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        layer_type = layer.__class__.__name__

        # Select rule for current layer
        active_rule = global_rule
        if composite_preset and layer_type in composite_preset:
            active_rule = composite_preset[layer_type]

        # Execute layer-specific relevance distribution
        distributed_R_list = []

        if 'Dense' in layer_type or 'Conv2D' in layer_type or 'DepthwiseConv2D' in layer_type:
            p_i = act_dict[id(inputs[0])]
            dist_R = distribute_relevance_gen(layer, p_i, current_R,
                                              rule=active_rule, alpha=alpha, beta=beta, epsilon=epsilon)
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        elif 'Add' in layer_type:
            # Conserved uniform split across residual branches
            split_R = current_R / len(inputs)
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R
                distributed_R_list.append(split_R)

        elif 'Concatenate' in layer_type:
            # Axis-aligned slicing for concatenated branches
            axis = layer.axis if hasattr(layer, 'axis') else -1
            start_idx_dim = 0
            for inp in inputs:
                dim_size = inp.shape[axis]
                slice_spec = [slice(None)] * len(current_R.shape)
                slice_spec[axis] = slice(start_idx_dim, start_idx_dim + dim_size)
                split_R = current_R[tuple(slice_spec)]
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R
                distributed_R_list.append(split_R)
                start_idx_dim += dim_size

        elif 'ZeroPadding2D' in layer_type:
            # Invert padding by cropping relevance map to original input dimensions
            pad = layer.padding
            (top, bottom), (left, right) = pad
            h, w = current_R.shape[1], current_R.shape[2]
            dist_R = current_R[:, top:h - bottom, left:w - right, :]
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        elif 'Flatten' in layer_type:
            # Reshape 2D relevance back into original 4D feature map geometry
            input_shape = inputs[0].shape
            dist_R = tf.reshape(current_R, [-1] + list(input_shape[1:]))
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        elif 'MaxPooling2D' in layer_type or 'GlobalAveragePooling2D' in layer_type:
            p_i = act_dict[id(inputs[0])]
            with tf.GradientTape() as tape:
                tape.watch(p_i)
                z = layer(p_i)
                tmp = tf.reduce_sum(z * current_R)
            dist_R = tape.gradient(tmp, p_i)
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        else:
            # Identity passthrough (e.g. BatchNormalization, Activation, Dropout)
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + current_R
                distributed_R_list.append(current_R)

        # Track layer conservation leakage
        sum_lower = sum([tf.reduce_sum(r).numpy() for r in distributed_R_list])
        leakage_pc = (abs(sum_upper - sum_lower) / (abs(sum_upper) + 1e-10)) * 100

    # 5. Extract input-level relevance and normalize
    final_input_R = rel_dict[id(model.input)]
    final_sum = tf.reduce_sum(final_input_R).numpy()

    # Global normalization factor to guarantee conservation equivalence
    correction_factor = initial_sum / (final_sum + 1e-10)
    final_input_R = rel_dict[id(model.input)] * correction_factor

    total_leakage = ((initial_sum - final_sum) / (initial_sum + 1e-10)) * 100

    print("-" * 75)
    print(f"Initial Prediction Sum: {initial_sum:.6f}")
    print(f"Final Relevance Sum:    {final_sum:.6f}")
    print(f"Total Network Leakage:  {total_leakage:.6f}%")

    return final_input_R.numpy(), initial_sum, final_sum, total_leakage


##############################################################################
# 3. Post-Processing & Visualization Utilities
##############################################################################

def apply_superpixel_pooling(original_image, relevance_map, n_segments=200, compactness=10):
    """Aggregate relevance within SLIC superpixel segments.

    Averages pixel-wise attribution scores across structurally coherent image segments.

    Args:
        original_image (np.ndarray or tf.Tensor): RGB input image array of shape (H, W, 3).
        relevance_map (np.ndarray or tf.Tensor): Attribution map of shape (1, H, W, 1) or (H, W).
        n_segments (int, optional): Number of SLIC superpixel clusters. Defaults to 200.
        compactness (float, optional): SLIC compactness parameter balancing color proximity and space. Defaults to 10.

    Returns:
        np.ndarray: Superpixel-pooled relevance map of shape (1, H, W, 1).
    """
    if tf.is_tensor(relevance_map):
        relevance_map = relevance_map.numpy()

    heatmap_2d = np.squeeze(relevance_map)
    if len(heatmap_2d.shape) == 3:
        heatmap_2d = np.mean(heatmap_2d, axis=-1)

    if tf.is_tensor(original_image):
        original_image = original_image.numpy()
    original_image = np.squeeze(original_image)

    segments = slic(img_as_float(original_image), n_segments=n_segments, compactness=compactness, start_label=1)
    refined_map = np.zeros_like(heatmap_2d)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        refined_map[mask] = np.mean(heatmap_2d[mask])

    if refined_map.max() > 0:
        refined_map = refined_map / refined_map.max()

    h, w = refined_map.shape
    return refined_map.reshape(1, h, w, 1)


def visualize_refined_heatmap2(original_image, relevance_map, alpha=0.5, sigma=5):
    """Render a region-smoothed heatmap overlay onto the original image.

    Applies positive-clipping, log-scaling, morphological dilation, and Gaussian smoothing
    to transform high-frequency attribution points into smooth, visually interpretable regions.

    Args:
        original_image (np.ndarray): Original image array of shape (H, W, 3).
        relevance_map (np.ndarray): Raw relevance heatmap of shape (H, W) or (1, H, W, C).
        alpha (float, optional): Overlay transparency factor in [0, 1]. Defaults to 0.5.
        sigma (float, optional): Gaussian filter standard deviation. Defaults to 5.

    Returns:
        tuple: (res_norm, overlay)
            - res_norm (np.ndarray): Normalized 2D attribution heatmap in [0, 1].
            - overlay (np.ndarray): Blended RGB overlay image array.
    """
    relevance_map = np.array(relevance_map)

    if len(relevance_map.shape) == 4:
        relevance_map = np.sum(relevance_map[0], axis=-1)
    elif len(relevance_map.shape) == 3:
        relevance_map = np.sum(relevance_map, axis=-1)

    if len(original_image.shape) == 4:
        original_image = np.squeeze(original_image, axis=0)

    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)

    # 1. R+ extraction and outlier suppression
    res_plus = np.maximum(relevance_map, 0)
    v_max = np.percentile(res_plus, 99)
    res_clipped = np.clip(res_plus, 0, v_max)

    # 2. Log scaling for dynamic range compression
    res_log = np.log1p(res_clipped)

    # 3. Morphological dilation
    kernel = np.ones((3, 3), np.uint8)
    res_dilated = cv2.dilate(res_log, kernel, iterations=1)

    # 4. Gaussian smoothing
    res_smoothed = cv2.GaussianBlur(res_dilated, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 5. Background noise suppression
    threshold = np.percentile(res_smoothed, 80)
    res_refined = np.where(res_smoothed >= threshold, res_smoothed, 0)

    # 6. Normalization and colormap application
    if res_refined.max() > 0:
        res_norm = (res_refined - res_refined.min()) / (res_refined.max() - res_refined.min() + 1e-10)
    else:
        res_norm = res_refined

    res_uint8 = (res_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(res_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 7. Blended overlay creation
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)

    return res_norm, overlay


def get_gradcam_pure_tf_fixed(model, img_array, layer_name, class_idx):
    """Compute Grad-CAM heatmap using pure TensorFlow GradientTape.

    Calculates feature map gradients with respect to the target class score to produce
    a class activation map.

    Args:
        model (tf.keras.Model): Target Keras functional model.
        img_array (np.ndarray): Input image array of shape (1, H, W, C).
        layer_name (str): Name of the target convolutional layer.
        class_idx (int): Target class index for Grad-CAM explanation.

    Returns:
        np.ndarray: Normalized 2D Grad-CAM heatmap array of shape (H, W).
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(f"Gradient computation failed. Verify layer name '{layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def visualize_TFRP_new(model, original_image, model_input, relevance_map, top_class, grad_th=0.2,
                       alpha=0.5, sigma=1, segment_n=400, step=3):
    """Comprehensive TFRP visualization pipeline with multi-stage refinement.

    Combines raw relevance propagation with optional Grad-CAM coarse gating (step >= 3)
    and SLIC superpixel clustering (step >= 4) for refined saliency overlays.

    Args:
        model (tf.keras.Model): Neural network classification model.
        original_image (np.ndarray): Original image array for visualization.
        model_input (np.ndarray): Preprocessed input tensor fed to the network.
        relevance_map (np.ndarray): Input relevance map from backpropagation.
        top_class (int): Predicted target class index.
        grad_th (float, optional): Threshold for Grad-CAM gating mask. Defaults to 0.2.
        alpha (float, optional): Heatmap transparency factor in blending. Defaults to 0.5.
        sigma (float, optional): Gaussian smoothing sigma parameter. Defaults to 1.
        segment_n (int, optional): Superpixel cluster count for SLIC. Defaults to 400.
        step (int, optional): Pipeline progression level (1: raw, 3: Grad-CAM gated, 4: superpixel pooled). Defaults to 3.

    Returns:
        tuple: (res_norm, overlay)
            - res_norm (np.ndarray): Refined attribution heatmap array.
            - overlay (np.ndarray): Final blended RGB overlay visualization.
    """
    if len(original_image.shape) == 4:
        img_h = original_image.shape[1]
        img_w = original_image.shape[2]
    else:
        img_h = original_image.shape[0]
        img_w = original_image.shape[1]

    if step >= 3:
        new_model = model
        last_conv_name = None

        # Find the last convolutional layer
        for layer in new_model.layers:
            if 'convolution' in layer.name.lower() or 'conv' in layer.name.lower():
                last_conv_name = layer.name

        if last_conv_name is None:
            raise ValueError("No Convolutional layer found in the model architecture.")

        # Find the subsequent ReLU activation layer if present
        target_layer_name = None
        found_last_conv = False
        for layer in new_model.layers:
            if layer.name == last_conv_name:
                found_last_conv = True
                continue
            if found_last_conv and 'relu' in layer.name.lower():
                target_layer_name = layer.name
                break

        if target_layer_name is None:
            target_layer_name = last_conv_name

        safe_model_input = model_input.copy()
        if safe_model_input.ndim == 3:
            safe_model_input = safe_model_input[np.newaxis, ...]
        elif safe_model_input.ndim == 2:
            safe_model_input = safe_model_input[np.newaxis, ..., np.newaxis]

        cam = get_gradcam_pure_tf_fixed(new_model, safe_model_input, target_layer_name, top_class)

        gradcam_map = cv2.resize(cam, (img_h, img_w))
        gradcam_map = gradcam_map[:, :, np.newaxis]

        srm_pos = relevance_map.copy()
        srm_pos = np.maximum(srm_pos, 0)

        # Min-max normalization
        pixel_map = (srm_pos - srm_pos.min()) / (srm_pos.max() - srm_pos.min() + 1e-10)
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-10)
        gradcam_map = gradcam_map[np.newaxis, :, :, :]

        # Grad-CAM threshold mask
        gradcam_map[gradcam_map < grad_th] = 0

        # Element-wise gating
        relevance_map = pixel_map * gradcam_map

    if step >= 4:
        relevance_map = apply_superpixel_pooling(original_image, relevance_map, n_segments=segment_n)

    res_norm, overlay = visualize_refined_heatmap2(original_image, relevance_map, alpha=alpha, sigma=sigma)
    return res_norm, overlay