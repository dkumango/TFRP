###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""Evaluation Library for Relevance Heatmaps and Structural Tolerance.

This module provides quantitative evaluation metrics for explainability heatmaps,
including relevance sparsity, batch conservation error, and heatmap entropy.
Additionally, it provides synthetic neural network architectures (residual additions,
zero-padding, and global average pooling) designed to benchmark model structure tolerance
and relevance conservation properties.
"""

import numpy as np
from scipy.stats import entropy
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Add, Flatten, Dense


# 1. Relevance Sparsity #####################################################
# Measures the fraction of near-zero relevance values in the attribution map.
# If relevance is excessively sparse or contains large dead regions, distortion may occur.
# Low: Well-distributed relevance across informative regions.
# High: Many inactive/dead relevance pixels.

def relevance_sparsity(R, threshold=1e-6):
    """Compute the relevance sparsity of an attribution heatmap.

    Calculates the proportion of relevance values whose absolute magnitude
    falls below a predefined numerical threshold.

    Args:
        R (np.ndarray): Relevance heatmap array of shape (H, W, C) or (N, H, W, C).
        threshold (float, optional): Magnitude threshold below which relevance
            is considered inactive/zero. Defaults to 1e-6.

    Returns:
        float: Sparsity score representing the fraction of near-zero relevance elements.
    """
    return np.mean(np.abs(R) < threshold)


# 2. Conservation Error #####################################################

def batch_conservation_error(logit, R):
    """Compute the percentage conservation leakage error for a batch of explanations.

    Evaluates the degree to which the sum of the input attributions matches
    the model's target logit output score.

    Args:
        logit (float or np.ndarray): Target class output logit score(s) of shape (N,) or scalar.
        R (np.ndarray): Input relevance attribution tensor of shape (N, H, W, C).

    Returns:
        np.ndarray: Percentage conservation error between total relevance and logit score.
    """
    relevance_sum = np.sum(R, axis=(1, 2, 3))
    leakage_percent = np.abs((np.sum(relevance_sum) - logit) / (logit + 1e-10) * 100)
    return leakage_percent


# 3. Heatmap Entropy #####################################################
# Quantifies fragmentation and noise in relevance heatmaps.
# Low: Highly focused, localized attribution.
# High: Fragmented, dispersed, or noisy attribution.

def heatmap_entropy(R):
    """Calculate the Shannon entropy of a normalized absolute relevance distribution.

    Args:
        R (np.ndarray): Relevance heatmap array of arbitrary shape.

    Returns:
        float: Shannon entropy representing the spatial dispersion/concentration of relevance.
    """
    p = np.abs(R).flatten()
    p = p / (np.sum(p) + 1e-12)
    return entropy(p)


# 4. Structure Tolerance Test Models ########################################
# Evaluates resilience to specific architectural motifs (Residual/Add, Padding, GAP).

def create_add_test_model(shape=(224, 224, 3)):
    """Create a minimal synthetic model with a residual addition branch.

    Constructs a dual-branch network containing a Conv2D branch and an identity branch
    merged via an Add layer, used to evaluate relevance conservation through residual junctions.

    Args:
        shape (tuple, optional): Input tensor shape (H, W, C). Defaults to (224, 224, 3).

    Returns:
        keras.Model: Compiled Keras model instance for residual addition tolerance testing.
    """
    inputs = Input(shape=shape)

    # Convolutional branch
    path1 = Conv2D(
        3,
        (1, 1),
        padding='same',
        use_bias=False,
        name='conv_branch'
    )(inputs)

    # Identity branch
    path2 = inputs

    # Residual addition
    merged = Add(name='add_layer')([path1, path2])

    x = Flatten()(merged)
    outputs = Dense(
        1,
        activation='linear',
        use_bias=False,
        name='output_dense'
    )(x)

    return Model(inputs, outputs, name="Add_Test")


def create_padding_test_model(shape=(224, 224, 3)):
    """Create a synthetic model containing zero-padding followed by valid convolution.

    Tests whether relevance propagation correctly accounts for zero-padding boundary
    regions without spurious leakage.

    Args:
        shape (tuple, optional): Input tensor shape (H, W, C). Defaults to (224, 224, 3).

    Returns:
        keras.Model: Keras model instance for spatial padding tolerance testing.
    """
    inputs = keras.Input(shape=shape)
    # Expand spatial dimensions (e.g. (224, 224) -> (230, 230))
    padded = keras.layers.ZeroPadding2D(padding=(3, 3))(inputs)
    # Recover original dimensions using valid convolution (e.g. (230, 230) -> (224, 224))
    x = keras.layers.Conv2D(3, (7, 7), padding='valid')(padded)

    outputs = keras.layers.Dense(1, activation='linear')(keras.layers.Flatten()(x))
    return keras.Model(inputs=inputs, outputs=outputs, name="Padding_Test")


def create_gap_test_model(shape=(224, 224, 3)):
    """Create a synthetic model with Global Average Pooling (GAP).

    Compresses spatial feature maps into channel-wise averages to test relevance
    backpropagation fidelity across pooling operations.

    Args:
        shape (tuple, optional): Input tensor shape (H, W, C). Defaults to (224, 224, 3).

    Returns:
        keras.Model: Keras model instance for Global Average Pooling tolerance testing.
    """
    inputs = keras.Input(shape=shape)
    # Compress spatial pixels into channel-wise means
    gap = keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = keras.layers.Dense(1, activation='linear')(gap)

    return keras.Model(inputs=inputs, outputs=outputs, name="GAP_Test")
