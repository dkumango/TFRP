###############################################################################
# TFRP library (Verion 5)
# TFRP : Tensor-Flow Relevance Propagation
# Keras tensorflow
###############################################################################

import tensorflow as tf
import numpy as np

###############################################################################
# Check CNN model type
# model type : Functional or Sequential
##############################################################################
from tensorflow.keras.layers import Add, Concatenate, Multiply, Average, Maximum, Subtract, Dot

def check_model_type(model):
    merge_layers = (Add, Concatenate, Multiply, Average, Maximum, Subtract, Dot)
    
    for layer in model.layers:
        # if the layer is in Merge_layers, it is a Functional model
        if isinstance(layer, merge_layers):
            return "Functional" 
            
    return "Sequential" 


##############################################################################
# Contribution distribution for linear layers via p_i * x_i 
# (resolves padding/stride issues)
##############################################################################
def distribute_relevance(layer, p_i, R_j):

    # w = layer.kernel
    if hasattr(layer, 'depthwise_kernel'):
        w = layer.depthwise_kernel

    else:
        w = layer.kernel
    
    # Use a small epsilon for numerical stability
    w_plus = tf.maximum(w, 1e-9)

    if 'Dense' in layer.__class__.__name__:
        # Defensive shape handling for Dense layers
        z = tf.matmul(p_i, w_plus) + 1e-9
        s = R_j / z
        c = tf.matmul(s, tf.transpose(w_plus))
        return p_i * c

    elif 'Conv2D' in layer.__class__.__name__:
        # 1. Reproducing forward pass using w_plus
        z = tf.nn.conv2d(p_i, w_plus,
                         strides=layer.strides,
                         padding=layer.padding.upper()) + 1e-9

        # 2. Shape mismatch resolution: Adjust $R_j$ to match $z$ if they differ.
        # (Normally, z and R_j should have identical shapes)
        s = R_j / z

        # 3. Backward pass: Extracting c with the same shape as p_i using GradientTape
        with tf.GradientTape() as tape:
            tape.watch(p_i)
            
            z_prime = tf.nn.conv2d(p_i, w_plus,
                                   strides=layer.strides,
                                   padding=layer.padding.upper())
            # Calculate weighted gradient for p_i using s(R_j/z). 
            tmp = tf.reduce_sum(z_prime * s)

        c = tape.gradient(tmp, p_i)

        # Handle disconnected gradients: Set c to 0 if it is None
        if c is None:
            return tf.zeros_like(p_i)

        return p_i * c


##############################################################################
# Get a relavance map (Sequential model) 
##############################################################################
def get_relevance_map_sequential(model, input_image, target_class_idx=None):
    # Aligning input data format
    if isinstance(input_image, list):
        input_image = input_image[0]

    if not tf.is_tensor(input_image):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    if len(input_image.shape) == 3:
        input_image = tf.expand_dims(input_image, axis=0)

    # Initial model run to trigger inbound node creation
    _ = model(input_image)

    # Layer activation extraction; uses model.input for robustness.
    #layer_outputs = [layer.output for layer in model.layers]   #@@@@

    act_dict = {}
    x = input_image
    for i, layer in enumerate(model.layers):
        x = layer(x)
        act_dict[i] = x


    # Setting the initial relevance score (R)
    preds = model(input_image)
    if target_class_idx is None:
        target_class_idx = tf.argmax(preds[0])
    
    R = tf.one_hot([target_class_idx], preds.shape[-1]) * preds

    # Iterating through layers in reverse order
    for i in range(len(model.layers)-1, -1, -1):
        layer = model.layers[i]
        layer_type = layer.__class__.__name__

        # Current input (Original image if layer index is 0, else previous layer's output)
        p_i = input_image if i == 0 else act_dict[i-1]

        if 'Dense' in layer_type:
            w = layer.kernel
            w_plus = tf.maximum(w, 1e-9)
            z = tf.matmul(p_i, w_plus) + 1e-9
            s = R / z
            c = tf.matmul(s, tf.transpose(w_plus))
            R = p_i * c

        elif 'Conv2D' in layer_type:
            w = getattr(layer, 'depthwise_kernel', getattr(layer, 'kernel', None))
            w_plus = tf.maximum(w, 1e-9)
            
            strides = layer.strides
            padding = layer.padding.upper()
            
            z = tf.nn.conv2d(p_i, w_plus, strides=strides, padding=padding) + 1e-9
            s = R / z

            with tf.GradientTape() as tape:
                tape.watch(p_i)
                z_prime = tf.nn.conv2d(p_i, w_plus, strides=strides, padding=padding)
                
                tmp = tf.reduce_sum(z_prime * s)           # Calculating the contribution distribution ratio 
            R = p_i * tape.gradient(tmp, p_i)

        elif 'MaxPooling2D' in layer_type:
            # Aligning R with pooling output shape for error prevention.
            with tf.GradientTape() as tape:
                tape.watch(p_i)
                z = tf.nn.max_pool(p_i, ksize=layer.pool_size, 
                                   strides=layer.strides, padding=layer.padding.upper())
                # Attempting to align R (from upper layer) with z's shape
                R_reshaped = tf.reshape(R, z.shape)
                tmp = tf.reduce_sum(z * R_reshaped)
            R = tape.gradient(tmp, p_i)

        elif 'Flatten' in layer_type:
            R = tf.reshape(R, p_i.shape)

        elif 'Dropout' in layer_type or 'BatchNormalization' in layer_type:
            # Identity mapping (1:1 transfer).
            continue
            
    return R.numpy()


##############################################################################
# Get a relavance map (Functional model) 
##############################################################################
from tensorflow.keras import layers, Input

def get_relevance_map_graph(model, input_image, target_class_idx=None):

    # Cleaning and preprocessing input data
    if isinstance(input_image, list): input_image = input_image[0]
    if not tf.is_tensor(input_image): input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    if len(input_image.shape) == 3: input_image = tf.expand_dims(input_image, axis=0)

    # Extracting activation values via Tensor ID mapping
    layer_outputs = [l.output for l in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model(input_image)          #########################

    act_dict = {id(l.output): act for l, act in zip(model.layers, activations)}

    # Initializing starting relevance (R)
    preds = model(input_image)
    if target_class_idx is None:
        target_class_idx = tf.argmax(preds[0])
    R = tf.one_hot([target_class_idx], preds.shape[-1]) * preds

    # Define relevance tracking dictionary
    rel_dict = {id(model.layers[-1].output): R}

    # Reverse graph traversal
    for i in range(len(model.layers)-1, -1, -1):

        layer = model.layers[i]
        layer_output_id = id(layer.output)

        if layer_output_id not in rel_dict:
            continue

        current_R = rel_dict[layer_output_id]
        inputs = layer.input
        if not isinstance(inputs, list):
            inputs = [inputs]

        layer_type = layer.__class__.__name__

        # --- Layer-specific processing ---

        if 'Dense' in layer_type or 'Conv2D' in layer_type:
            p_i = act_dict[id(inputs[0])]
            distributed_R = distribute_relevance(layer, p_i, current_R)
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + distributed_R

        elif 'Add' in layer_type:
            # Relevance distribution: Uniform or proportional to activations
            split_R = current_R / len(inputs)
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R

        elif 'Concatenate' in layer_type:
            # Decomposing $R$ along the merged axis
            axis = layer.axis if hasattr(layer, 'axis') else -1
            start_idx = 0
            for inp in inputs:
                # Slicing based on original input dimensions along the target axis
                dim_size = inp.shape[axis]
                # Replacing tf.split with slice for handling diverse dimensions
                slice_spec = [slice(None)] * len(current_R.shape)
                slice_spec[axis] = slice(start_idx, start_idx + dim_size)

                split_R = current_R[tuple(slice_spec)]
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R
                start_idx += dim_size

        elif 'MaxPooling2D' in layer_type or 'GlobalAveragePooling2D' in layer_type:
            p_i = act_dict[id(inputs[0])]
            with tf.GradientTape() as tape:
                tape.watch(p_i)
                z = layer(p_i)
                tmp = tf.reduce_sum(z * current_R)
            distributed_R = tape.gradient(tmp, p_i)
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + distributed_R

        elif 'Flatten' in layer_type:
            p_i = act_dict[id(inputs[0])]
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + tf.reshape(current_R, p_i.shape)

        elif 'ZeroPadding2D' in layer_type:
            # Cropping $R$ to match the original input size
            # Padding format: ((top, bottom), (left, right))
            pad = layer.padding
            top, bottom = pad[0]
            left, right = pad[1]

            # Slicing the valid interior (excluding padded boundaries)
            # ex) [:, 1:-1, 1:-1, :]
            # bottom이나 right가 0일 수도 있으므로 슬라이싱 범위를 동적으로 계산합니다.
            h_end = current_R.shape[1] - bottom if bottom > 0 else None
            w_end = current_R.shape[2] - right if right > 0 else None

            distributed_R = current_R[:, top:h_end, left:w_end, :]
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + distributed_R

        elif 'BatchNormalization' in layer_type:
            # Direct 1:1 relevance transfer to the preceding layer.
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + current_R

        else:
            # Activation, BatchNormalization, Dropout, ...
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + current_R

    return rel_dict[id(model.input)].numpy()


#############################################################################
#############################################################################
# VISUALIZATION
#############################################################################
#############################################################################
# pip install tf_keras_vis       # for gradCAM

import cv2
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.util import img_as_float


#############################################################################
# Superpixel pooling
#############################################################################
def apply_superpixel_pooling(original_image, relevance_map, n_segments=200, compactness=10):
    if tf.is_tensor(relevance_map):
        relevance_map = relevance_map.numpy()

    heatmap_2d = np.squeeze(relevance_map)

    if len(heatmap_2d.shape) == 3:
        heatmap_2d = np.mean(heatmap_2d, axis=-1)

    print(f"[Superpixel] Input Range: Min={heatmap_2d.min():.4f}, Max={heatmap_2d.max():.4f}")

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


#############################################################################
# Visualize relevance map
#############################################################################
def visualize_refined_heatmap2(original_image, relevance_map, alpha=0.5, sigma=5):

    relevance_map = np.array(relevance_map)

    # Preprocessing: Aligning data formats
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

    # Area expansion algorithm applied

    # (1) Extracting R+ and suppressing outliers to mitigate extreme values.
    res_plus = np.maximum(relevance_map, 0)
    # Clipping top 1% values to the 99th percentile level for stability
    v_max = np.percentile(res_plus, 99)    # 99 -> 95
    res_clipped = np.clip(res_plus, 0, v_max)

    # (2) Log Scaling
    # Dynamic range compression for better visualization of low-relevance areas
    res_log = np.log1p(res_clipped)

    # (3) Morphological dilation for area expansion (Dilation)
    # Regionalizing small points through spatial expansion
    kernel = np.ones((3, 3), np.uint8)           
    res_dilated = cv2.dilate(res_log, kernel, iterations=1)

    # (4) Heavy Gaussian blur for noise reduction and smoothing (Large Sigma)
    # Enhancing sigma and kernel dimensions for broader smoothing effects
    # Higher sigma values allow isolated points to merge, forming cohesive regions
    res_smoothed = cv2.GaussianBlur(res_dilated, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # (5) Eliminating background noise and artifacts (Thresholding)
    # Filtering the lower 30–40% to clean the background without losing structural details.
    threshold = np.percentile(res_smoothed, 80)   # 40
    res_refined = np.where(res_smoothed >= threshold, res_smoothed, 0)

    # 0–1 Normalization and colormap assignment
    if res_refined.max() > 0:
        res_norm = (res_refined - res_refined.min()) / (res_refined.max() - res_refined.min() + 1e-10)
    else:
        res_norm = res_refined

    res_uint8 = (res_norm * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(res_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlaying the normalized heatmap on the input image
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)

    # Draw overay graph
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Region-based Heatmap (sigma={sigma})")
    plt.imshow(res_norm, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay Result")
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return res_norm, overlay

#############################################################################
# Implement GradCam
#############################################################################
def get_gradcam_pure_tf_fixed(model, img_array, layer_name, class_idx):
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
        raise ValueError("Error: Gradient flow is disconnected. \n Please verify that the target layer name is correct")
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


#############################################################################
# Visualize relevance map (Main function)
#############################################################################
def visualize_TFRP_new(model, original_image, model_input, relevance_map, top_class, grad_th=0.2,
                       alpha=0.5, sigma=1, segment_n=400, step=2):


    if len(original_image.shape) ==4:
        img_h = original_image.shape[1]
        img_w = original_image.shape[2]
    else:
        img_h = original_image.shape[0]
        img_w = original_image.shape[1]

    print(f"Original Image Size: {img_h}x{img_w}")

    if step >= 2:
        # apply grad cam
        # gradCAM mask

        model_type = check_model_type(model)
        if model_type == "Sequential" :   
            print("Detected Sequential model. Wrapping with Functional API for GradCAM compatibility.") 

            # Specifying the input layer definition
            inputs = tf.keras.Input(shape=original_image.shape)
            x = inputs
            
            for layer in model.layers:
                x = layer(x)
                
            new_model = tf.keras.Model(inputs=inputs, outputs=x)
          
        else:
            new_model = model  

        ############################
        target_layer_name = None
        last_conv_name = None 

        # Locating the last 'convolution' operation name.
        for layer in new_model.layers:
            if 'convolution' in layer.name.lower() or 'conv' in layer.name.lower():
                last_conv_name = layer.name

        if last_conv_name is None:
            raise ValueError("Error: Failed to locate any 'Conv' layers in the current model.")

        # Finding the ReLU layer subsequent to the final convolution.
        found_last_conv = False
        for layer in new_model.layers:
            if layer.name == last_conv_name:
                found_last_conv = True
                continue                    # Checking layers following the identified convolution
            if found_last_conv and 'relu' in layer.name.lower():
                target_layer_name = layer.name
                break

        # Falling back to the convolution layer if ReLU is unavailable.
        if target_layer_name is None:
            target_layer_name = last_conv_name

        safe_model_input = model_input.copy()
        if safe_model_input.ndim == 3: 
            safe_model_input = safe_model_input[np.newaxis, ...] 
        elif safe_model_input.ndim == 2: 
            safe_model_input = safe_model_input[np.newaxis, ..., np.newaxis]

        cam = get_gradcam_pure_tf_fixed(new_model, safe_model_input, target_layer_name, top_class)

        ############################
        gradcam_map = cv2.resize(cam, (img_h, img_w)) # Matching heatmap scale to input image size
        gradcam_map = gradcam_map[:, :, np.newaxis]   # (h,w,1)

        srm_pos = relevance_map.copy()
        srm_pos = np.maximum(srm_pos, 0)

        # Normalization
        pixel_map = (srm_pos  - srm_pos .min()) / (srm_pos.max() - srm_pos.min())
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min())

        gradcam_map = gradcam_map[np.newaxis, :, :,:]

        threshold = grad_th
        gradcam_map[gradcam_map < threshold] = 0

        # Merge pixel_map and gradcam_map (Element-wise Multiplication)
        relevance_map = pixel_map * gradcam_map

    if step >= 3:
        # apply super pixel pooling
        relevance_map = apply_superpixel_pooling(original_image, relevance_map, n_segments=segment_n)

    res_norm, overlay = visualize_refined_heatmap2(original_image, relevance_map, alpha=alpha, sigma=sigma)

    return res_norm, overlay

