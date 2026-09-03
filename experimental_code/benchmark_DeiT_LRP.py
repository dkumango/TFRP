###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""DeiT-Tiny Vision Transformer Benchmark with Conventional LRP (iNNvestigate).

This experiment measures actual conservation error and leakage for conventional LRP (z+ rule)
implemented via the iNNvestigate library on a 2D topology-preserving DeiT-Tiny architecture.

Results are saved to `exp_result/Transformer_DeiT/deit_tiny_real_lrp_zplus.csv`.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import softmax

import keras
from keras.models import Model
from keras.layers import Layer, Dense, Dropout, Add, Conv2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import innvestigate

# Enforce TF 1.x compatibility mode for iNNvestigate graph analyzers
tf.compat.v1.disable_eager_execution()


def set_seeds(seed=42):
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


set_seeds(42)


def deit_gelu_tf1(x):
    """Compute Gaussian Error Linear Unit (GELU) activation in TF 1.x.

    Args:
        x (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: GELU activated tensor.
    """
    return x * 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0)))


class LayerNormalization(Layer):
    """Pure Keras LayerNormalization for static TF 1.x graph execution."""

    def __init__(self, epsilon=1e-5, **kwargs):
        """Initialize LayerNormalization layer.

        Args:
            epsilon (float, optional): Small stabilizer. Defaults to 1e-5.
            **kwargs: Layer keyword arguments.
        """
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create trainable scale and shift parameters.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer='zeros', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        """Apply layer normalization across the feature dimension.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Normalized and scaled tensor.
        """
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta


def calculate_rce(r_in, r_out, eps=1e-9):
    """Calculate Relative Conservation Error (RCE).

    Args:
        r_in (float): Input relevance sum.
        r_out (float): Output target logit score.
        eps (float, optional): Numerical stabilizer. Defaults to 1e-9.

    Returns:
        float: Relative Conservation Error.
    """
    return np.abs(r_in - r_out) / (np.abs(r_out) + eps)


def calculate_continuity_score(r_in, r_out):
    """Calculate Propagation Continuity ratio (r_in / r_out).

    Args:
        r_in (float): Input relevance sum.
        r_out (float): Output target logit score.

    Returns:
        float: Propagation continuity ratio.
    """
    if np.abs(r_out) == 0:
        return 0.0
    return float(r_in / r_out)


class MultiHeadSelfAttention2D(Layer):
    """2D Multi-Head Self-Attention using 1x1 convolutions."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        """Initialize MultiHeadSelfAttention2D.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Head count.
            **kwargs: Layer keyword arguments.
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
        """Forward multi-head self-attention on 4D tensor.

        Args:
            x (tf.Tensor): Input feature map [B, H, W, C].

        Returns:
            tf.Tensor: Projected attention output [B, H, W, C].
        """
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]
        N = H * W

        q = tf.transpose(tf.reshape(self.q_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        q = tf.reshape(q, (B, self.num_heads, N, self.head_dim))

        k = tf.transpose(tf.reshape(self.k_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        k = tf.reshape(k, (B, self.num_heads, N, self.head_dim))

        v = tf.transpose(tf.reshape(self.v_conv(x), (B, H, W, self.num_heads, self.head_dim)), [0, 3, 1, 2, 4])
        v = tf.reshape(v, (B, self.num_heads, N, self.head_dim))

        match = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.head_dim, tf.float32)
        attention_weights = tf.nn.softmax(match / tf.math.sqrt(dk), axis=-1)

        attention_out = tf.matmul(attention_weights, v)
        attention_out = tf.reshape(attention_out, (B, self.num_heads, H, W, self.head_dim))
        attention_out = tf.transpose(attention_out, [0, 2, 3, 1, 4])
        attention_out = tf.reshape(attention_out, (B, H, W, C))

        # Enforce static channel dimension for TF 1.x graph compiler
        attention_out.set_shape([None, 14, 14, 192])

        return self.projection(attention_out)


def create_deit_tiny_2d_backbone(num_classes=1000):
    """Build 2D topology DeiT-Tiny model for iNNvestigate.

    Args:
        num_classes (int, optional): Classification classes. Defaults to 1000.

    Returns:
        keras.models.Model: DeiT-Tiny model.
    """
    inputs = keras.Input(shape=(224, 224, 3))
    x = Conv2D(filters=192, kernel_size=(16, 16), strides=(16, 16), name="patch_embedding")(inputs)

    for i in range(12):
        norm_1 = LayerNormalization()(x)
        attn_out = MultiHeadSelfAttention2D(embed_dim=192, num_heads=3, name=f"transformer_block_{i}_attn")(norm_1)
        x = Add()([x, attn_out])

        norm_2 = LayerNormalization()(x)
        mlp_out = Conv2D(192 * 4, kernel_size=1, activation=deit_gelu_tf1, name=f"transformer_block_{i}_mlp1")(norm_2)
        mlp_out = Conv2D(192, kernel_size=1, name=f"transformer_block_{i}_mlp2")(mlp_out)
        x = Add()([x, mlp_out])

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation=None, name="logits")(x)
    return Model(inputs=inputs, outputs=outputs, name="DeiT_Tiny_2D_Compatible")


print("Initializing DeiT-Tiny computation graph for iNNvestigate...")
model = create_deit_tiny_2d_backbone()
print("DeiT-Tiny model initialized successfully.")

# Prepare image dataset
image_path = 'imagen_500/'
images = random.sample(os.listdir(image_path), min(500, len(os.listdir(image_path))))
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

# Compile iNNvestigate lrp.z_plus analyzer
print("Compiling iNNvestigate lrp.z_plus analyzer...")
analyzer_plus = innvestigate.create_analyzer(
    "lrp.z_plus", model, neuron_selection_mode="max_activation"
)
print("Analyzer compiled successfully.")

# Main evaluation loop
lrp_transformer_records = []

print("\nStarting Conventional LRP (iNNvestigate) on DeiT-Tiny...")
print("=" * 80)

for i in range(n_images):
    x = np.expand_dims(input_features[i], axis=0)
    img_name = img_names[i]

    preds = model.predict(x)
    top_class_idx = np.argmax(preds[0])
    initial_sum = preds[0][top_class_idx]

    print(f"\n[Image {i}] Analyzing '{img_name}' -> Top Class Index: {top_class_idx}")

    start_time = time.time()
    result = analyzer_plus.analyze(x)
    execution_time = time.time() - start_time

    final_sum = np.sum(result[0])

    rce = calculate_rce(r_in=final_sum, r_out=initial_sum)
    continuity = calculate_continuity_score(r_in=final_sum, r_out=initial_sum)
    actual_leakage_pct = (1.0 - continuity) * 100 if continuity <= 1.0 else (continuity - 1.0) * 100

    lrp_transformer_records.append({
        'Model': 'DeiT-Tiny',
        'Image_Index': i,
        'File_Name': img_name,
        'Method': 'Conventional LRP (z+)',
        'Initial_Sum(R_out)': initial_sum,
        'Final_Sum(R_in)': final_sum,
        'Absolute_Error': np.abs(final_sum - initial_sum),
        'RCE_Value': rce,
        'RCE_Percentage(%)': rce * 100,
        'Leakage_Percentage(%)': actual_leakage_pct,
        'Propagation_Continuity': continuity,
        'Inference_Time(s)': execution_time
    })
    print(f" -> Real LRP (z+): RCE = {rce:.4e} | Continuity = {continuity:.4f} | Actual Leakage = {actual_leakage_pct:.2f}%")

# Save results
df_lrp_transformer = pd.DataFrame(lrp_transformer_records)
os.makedirs("exp_result/Transformer_DeiT", exist_ok=True)
df_lrp_transformer.to_csv("exp_result/Transformer_DeiT/deit_tiny_real_lrp_zplus.csv", index=False)

summary_lrp_transformer = df_lrp_transformer.groupby('Method')[
    ['Absolute_Error', 'RCE_Percentage(%)', 'Leakage_Percentage(%)', 'Propagation_Continuity', 'Inference_Time(s)']
].mean()

print("\n" + "=" * 80)
print("DeiT-Tiny Real LRP (z+) Summary")
print("=" * 80)
print(summary_lrp_transformer.to_string())
print("=" * 80)