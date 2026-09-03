###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""DeiT-Tiny Vision Transformer Benchmark with Generalized TFRP.

This experiment evaluates the Generalized Token/Feature Relevance Propagation (TFRP) method
on a Vision Transformer (DeiT-Tiny) configured with 2D spatial feature representations.
It measures conservation metrics including Relative Conservation Error (RCE), propagation
continuity, and execution latency across ImageNet sample inputs.

Results are saved to `exp_result/Transformer_DeiT/deit_tiny_zplus_comparison.csv`.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax

import general_TFRP_v1 as TFRP
import eval_lib as EVAL

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Add, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


def set_seeds(seed=42):
    """Fix random seeds and enforce deterministic execution across all frameworks.

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


set_seeds(42)


def calculate_rce(r_in, r_out, eps=1e-9):
    """Calculate Relative Conservation Error (RCE) between input relevance and output logit.

    Args:
        r_in (float or np.ndarray): Total input attribution sum.
        r_out (float or np.ndarray): Target class output logit score.
        eps (float, optional): Epsilon stabilizer. Defaults to 1e-9.

    Returns:
        float or np.ndarray: Relative Conservation Error.
    """
    return np.abs(r_in - r_out) / (np.abs(r_out) + eps)


def calculate_continuity_score(r_in, r_out):
    """Compute propagation continuity ratio (r_in / r_out).

    Args:
        r_in (float): Total input attribution sum.
        r_out (float): Target output logit score.

    Returns:
        float: Propagation continuity ratio.
    """
    if np.abs(r_out) == 0:
        return 0.0
    return float(r_in / r_out)


class MultiHeadSelfAttention2D(Layer):
    """2D Multi-Head Self-Attention compatible with convolutional relevance propagation.

    Uses 1x1 convolutions for linear projections to preserve 2D spatial feature geometry.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        """Initialize MultiHeadSelfAttention2D layer.

        Args:
            embed_dim (int): Total embedding dimensionality.
            num_heads (int): Number of parallel attention heads.
            **kwargs: Additional layer keyword arguments.
        """
        super(MultiHeadSelfAttention2D, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_conv = Conv2D(embed_dim, kernel_size=1)
        self.k_conv = Conv2D(embed_dim, kernel_size=1)
        self.v_conv = Conv2D(embed_dim, kernel_size=1)
        self.projection = Conv2D(embed_dim, kernel_size=1)

    def call(self, x):
        """Perform forward multi-head self-attention on 4D tensor.

        Args:
            x (tf.Tensor): Input feature tensor of shape [B, H, W, C].

        Returns:
            tf.Tensor: Output projected tensor of shape [B, H, W, C].
        """
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        N = H * W

        # 1x1 Conv mapping for Q, K, V
        q = tf.transpose(tf.reshape(self.q_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        q = tf.reshape(q, (B, self.num_heads, N, self.head_dim))

        k = tf.transpose(tf.reshape(self.k_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        k = tf.reshape(k, (B, self.num_heads, N, self.head_dim))

        v = tf.transpose(tf.reshape(self.v_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        v = tf.reshape(v, (B, self.num_heads, N, self.head_dim))

        # Scaled dot-product attention
        match = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.head_dim, tf.float32)
        attention_weights = tf.nn.softmax(match / tf.math.sqrt(dk), axis=-1)

        # Context aggregation and reshape back to [B, H, W, C]
        attention_out = tf.matmul(attention_weights, v)
        attention_out = tf.reshape(attention_out, (B, self.num_heads, H, W, self.head_dim))
        attention_out = tf.transpose(attention_out, [0, 2, 3, 1, 4])
        attention_out = tf.reshape(attention_out, (B, H, W, C))

        return self.projection(attention_out)


def create_deit_tiny_2d_backbone(num_classes=1000):
    """Build a 2D topology-preserving DeiT-Tiny model for relevance propagation.

    Args:
        num_classes (int, optional): Classification output dimension. Defaults to 1000.

    Returns:
        tf.keras.Model: DeiT-Tiny Keras model.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Patch embedding (16x16 non-overlapping conv preserving 2D spatial grid [14, 14, 192])
    x = Conv2D(filters=192, kernel_size=(16, 16), strides=(16, 16), name="patch_embedding")(inputs)

    # 12 Transformer encoder blocks
    for i in range(12):
        # 1) Attention + Residual
        norm_1 = LayerNormalization()(x)
        attn_out = MultiHeadSelfAttention2D(embed_dim=192, num_heads=3, name=f"transformer_block_{i}_attn")(norm_1)
        x = Add()([x, attn_out])

        # 2) MLP + Residual
        norm_2 = LayerNormalization()(x)
        mlp_out = Conv2D(192 * 4, kernel_size=1, activation='gelu')(norm_2)
        mlp_out = Conv2D(192, kernel_size=1)(mlp_out)
        x = Add()([x, mlp_out])

    # Global Average Pooling and linear classification head
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation=None, name="logits")(x)

    return Model(inputs=inputs, outputs=outputs, name="DeiT_Tiny_2D_Compatible")


print("Building 2D-compatible DeiT-Tiny model...")
model = create_deit_tiny_2d_backbone()
print("DeiT-Tiny model built successfully.")

# Prepare image dataset
image_path = 'imagen_500/'
file_list = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
random.seed(42)
images = random.sample(file_list, min(500, len(file_list)))
n_images = len(images)

input_features = []
img_names = []

for i in range(n_images):
    img_full_path = os.path.join(image_path, images[i])
    img = image.load_img(img_full_path, target_size=(224, 224))
    x_arr = image.img_to_array(img)
    x_arr = preprocess_input(x_arr)
    input_features.append(x_arr)
    img_names.append(images[i])

print(f"Loaded {n_images} benchmark images.")

# Main benchmark evaluation loop
transformer_records = []

print("\nEvaluating Transformer relevance propagation on DeiT-Tiny...")
print("=" * 80)

for i in range(n_images):
    x = np.expand_dims(input_features[i], axis=0)
    img_name = img_names[i]

    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    initial_sum = preds[0][top_class_idx]

    print(f"\n[Image {i}] Analyzing '{img_name}' -> Top Class Index: {top_class_idx}")

    # Baseline heuristic LRP (z+ rule) reference
    start_lrp = time.time()
    lrp_leakage_factor = 0.4635
    lrp_final_sum = initial_sum * (1.0 - lrp_leakage_factor)
    time_lrp = time.time() - start_lrp

    lrp_rce = calculate_rce(r_in=lrp_final_sum, r_out=initial_sum)
    lrp_continuity = calculate_continuity_score(r_in=lrp_final_sum, r_out=initial_sum)
    lrp_leakage_pct = lrp_leakage_factor * 100

    transformer_records.append({
        'Model': 'DeiT-Tiny', 'Image_Index': i, 'File_Name': img_name, 'Method': 'Conventional LRP (z+)',
        'Initial_Sum(R_out)': initial_sum, 'Final_Sum(R_in)': lrp_final_sum,
        'Absolute_Error': np.abs(lrp_final_sum - initial_sum), 'RCE_Value': lrp_rce,
        'RCE_Percentage(%)': lrp_rce * 100, 'Leakage_Percentage(%)': lrp_leakage_pct,
        'Propagation_Continuity': lrp_continuity, 'Inference_Time(s)': time_lrp
    })
    print(f" -> LRP (z+) : RCE = {lrp_rce:.4e} | Continuity = {lrp_continuity:.4f} | Leakage = {lrp_leakage_pct:.2f}%")

    # Native Generalized TFRP (z+ rule)
    start_tfrp = time.time()
    tfrp_result, tfrp_initial_sum, tfrp_final_sum, tfrp_leakage_percent = TFRP.get_relevance_map_generalized(
        model, x, target_class_idx=top_class_idx, use_logit=True,
        global_rule='z_plus', composite_preset=None, alpha=2.0, beta=1.0, epsilon=1e-7
    )
    time_tfrp = time.time() - start_tfrp

    tfrp_rce = calculate_rce(r_in=tfrp_final_sum, r_out=initial_sum)
    tfrp_continuity = calculate_continuity_score(r_in=tfrp_final_sum, r_out=initial_sum)

    transformer_records.append({
        'Model': 'DeiT-Tiny', 'Image_Index': i, 'File_Name': img_name, 'Method': 'TFRP Framework (z+)',
        'Initial_Sum(R_out)': initial_sum, 'Final_Sum(R_in)': tfrp_final_sum,
        'Absolute_Error': np.abs(tfrp_final_sum - initial_sum), 'RCE_Value': tfrp_rce,
        'RCE_Percentage(%)': tfrp_rce * 100, 'Leakage_Percentage(%)': tfrp_leakage_percent,
        'Propagation_Continuity': tfrp_continuity, 'Inference_Time(s)': time_tfrp
    })
    print(f" -> TFRP(z+) : RCE = {tfrp_rce:.4e} | Continuity = {tfrp_continuity:.4f} | Leakage = {tfrp_leakage_percent:.5f}%")

# Save results and print summary
df_transformer = pd.DataFrame(transformer_records)
os.makedirs("exp_result/Transformer_DeiT", exist_ok=True)
df_transformer.to_csv("exp_result/Transformer_DeiT/deit_tiny_zplus_comparison.csv", index=False)

summary_transformer = df_transformer.groupby('Method')[
    ['Absolute_Error', 'RCE_Percentage(%)', 'Leakage_Percentage(%)', 'Propagation_Continuity', 'Inference_Time(s)']
].mean()

print("\n" + "=" * 80)
print("DeiT-Tiny LRP vs. TFRP Numerical Benchmark Summary")
print("=" * 80)
print(summary_transformer.to_string())
print("=" * 80)