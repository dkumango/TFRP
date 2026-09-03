###############################################################################
# Execution Environment: Python 3.12 (.venv_312)
###############################################################################
"""DeiT-Tiny Vision Transformer Benchmark with Zennit LRP (PyTorch).

This benchmark evaluates actual conservation error and leakage for LRP (z+ rule)
implemented via the PyTorch Zennit library on a 2D topology DeiT-Tiny model.

Results are saved to `exp_result/Transformer_DeiT/deit_tiny_real_lrp_zplus_zennit.csv`.
"""

import os
import sys
import random
import time
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
from zennit.composites import LayerMapComposite
from zennit.rules import ZPlus
from zennit.attribution import Gradient


def set_seeds(seed=42):
    """Enforce deterministic operations and fix random seeds for PyTorch.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seeds(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_rce(r_in, r_out, eps=1e-9):
    """Calculate Relative Conservation Error (RCE).

    Args:
        r_in (float): Total input attribution sum.
        r_out (float): Target class output logit score.
        eps (float, optional): Epsilon stabilizer. Defaults to 1e-9.

    Returns:
        float: Relative Conservation Error.
    """
    return np.abs(r_in - r_out) / (np.abs(r_out) + eps)


def calculate_continuity_score(r_in, r_out):
    """Calculate Propagation Continuity ratio (r_in / r_out).

    Args:
        r_in (float): Total input attribution sum.
        r_out (float): Target output logit score.

    Returns:
        float: Propagation continuity ratio.
    """
    if np.abs(r_out) == 0:
        return 0.0
    return float(r_in / r_out)


class LayerNormalization2d(nn.Module):
    """Channel-wise Layer Normalization operating on 2D feature maps [B, C, H, W]."""

    def __init__(self, num_channels, eps=1e-5):
        """Initialize LayerNormalization2d module.

        Args:
            num_channels (int): Channel count.
            eps (float, optional): Stabilizer. Defaults to 1e-5.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        """Apply channel normalization.

        Args:
            x (torch.Tensor): Feature map of shape [B, C, H, W].

        Returns:
            torch.Tensor: Normalized feature map.
        """
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class MultiHeadSelfAttention2D(nn.Module):
    """Multi-Head Self-Attention for 2D spatial feature maps using Conv2d 1x1."""

    def __init__(self, embed_dim=192, num_heads=3):
        """Initialize MultiHeadSelfAttention2D.

        Args:
            embed_dim (int, optional): Embedding dimension. Defaults to 192.
            num_heads (int, optional): Number of heads. Defaults to 3.
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        """Forward pass for 2D attention.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].

        Returns:
            torch.Tensor: Attended output tensor [B, C, H, W].
        """
        B, C, H, W = x.shape
        N = H * W

        q = self.q_conv(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.k_conv(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = self.v_conv(x).reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        match = torch.matmul(q, k.transpose(-2, -1))
        dk = float(self.head_dim)
        attention_weights = torch.softmax(match / (dk ** 0.5), dim=-1)

        attention_out = torch.matmul(attention_weights, v)
        attention_out = attention_out.permute(0, 1, 3, 2).reshape(B, self.num_heads * self.head_dim, H, W)
        return self.projection(attention_out)


class DeiTTiny2D(nn.Module):
    """PyTorch 2D topology DeiT-Tiny model architecture."""

    def __init__(self, num_classes=1000):
        """Initialize DeiTTiny2D model.

        Args:
            num_classes (int, optional): Output class count. Defaults to 1000.
        """
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, 192, kernel_size=16, stride=16)

        self.blocks = nn.ModuleList()
        for i in range(12):
            block = nn.ModuleDict({
                'norm_1': LayerNormalization2d(192),
                'attn': MultiHeadSelfAttention2D(embed_dim=192, num_heads=3),
                'norm_2': LayerNormalization2d(192),
                'mlp1': nn.Conv2d(192, 192 * 4, kernel_size=1),
                'gelu': nn.GELU(),
                'mlp2': nn.Conv2d(192 * 4, 192, kernel_size=1)
            })
            self.blocks.append(block)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.logits = nn.Linear(192, num_classes)

    def forward(self, x):
        """Forward inference pass.

        Args:
            x (torch.Tensor): Input image tensor [B, 3, 224, 224].

        Returns:
            torch.Tensor: Output logit tensor [B, num_classes].
        """
        x = self.patch_embedding(x)
        for block in self.blocks:
            norm_1 = block['norm_1'](x)
            attn_out = block['attn'](norm_1)
            x = x + attn_out

            norm_2 = block['norm_2'](x)
            mlp_out = block['mlp1'](norm_2)
            mlp_out = block['gelu'](mlp_out)
            mlp_out = block['mlp2'](mlp_out)
            x = x + mlp_out

        x = self.gap(x).flatten(1)
        return self.logits(x)


print("Initializing PyTorch DeiT-Tiny model...")
model = DeiTTiny2D().to(device)
model.eval()
print("PyTorch DeiT-Tiny model ready.")

# Zennit LRP composite mapping (ZPlus applied to Conv2d and Linear layers)
composite = LayerMapComposite(layer_map=[
    (nn.Conv2d, ZPlus()),
    (nn.Linear, ZPlus()),
])

# Prepare image dataset
image_path = 'imagen_500/'
file_list = sorted([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
random.seed(42)
images = random.sample(file_list, min(500, len(file_list)))
n_images = len(images)

input_tensors = []
img_names = []

for i in range(n_images):
    img_full_path = os.path.join(image_path, images[i])
    img = Image.open(img_full_path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)

    # Preprocessing (RGB -> BGR and Mean Subtraction)
    x_arr = np.array(img, dtype=np.float32)[..., ::-1].copy()
    x_arr[..., 0] -= 103.939
    x_arr[..., 1] -= 116.779
    x_arr[..., 2] -= 123.68

    # NCHW tensor conversion
    x_tensor = torch.from_numpy(x_arr).permute(2, 0, 1).unsqueeze(0).float()
    input_tensors.append(x_tensor)
    img_names.append(images[i])

print(f"Loaded {n_images} benchmark images.")

# Main Zennit evaluation loop
lrp_zennit_records = []

print("\nStarting Zennit LRP (z+) on DeiT-Tiny...")
print("=" * 80)

for i in range(n_images):
    x = input_tensors[i].to(device)
    x.requires_grad_(True)
    img_name = img_names[i]

    preds = model(x)
    top_class_idx = preds.argmax(dim=1).item()
    initial_sum = preds[0, top_class_idx].item()

    print(f"\n[Image {i}] Analyzing '{img_name}' -> Top Class Index: {top_class_idx}")

    target_mask = torch.zeros_like(preds)
    target_mask[0, top_class_idx] = 1.0

    start_time = time.time()
    with Gradient(model=model, composite=composite) as attributor:
        _, relevance = attributor(x, target_mask)
    execution_time = time.time() - start_time

    final_sum = relevance.sum().item()

    rce = calculate_rce(r_in=final_sum, r_out=initial_sum)
    continuity = calculate_continuity_score(r_in=final_sum, r_out=initial_sum)
    actual_leakage_pct = (1.0 - continuity) * 100 if continuity <= 1.0 else (continuity - 1.0) * 100

    lrp_zennit_records.append({
        'Model': 'DeiT-Tiny',
        'Image_Index': i,
        'File_Name': img_name,
        'Method': 'Conventional LRP (z+ / Zennit)',
        'Initial_Sum(R_out)': initial_sum,
        'Final_Sum(R_in)': final_sum,
        'Absolute_Error': np.abs(final_sum - initial_sum),
        'RCE_Value': rce,
        'RCE_Percentage(%)': rce * 100,
        'Leakage_Percentage(%)': actual_leakage_pct,
        'Propagation_Continuity': continuity,
        'Inference_Time(s)': execution_time
    })
    print(f" -> Zennit LRP (z+): RCE = {rce:.4e} | Continuity = {continuity:.4f} | Actual Leakage = {actual_leakage_pct:.2f}%")

# Save results
df_lrp_zennit = pd.DataFrame(lrp_zennit_records)
output_dir = "exp_result/Transformer_DeiT"
os.makedirs(output_dir, exist_ok=True)
csv_output_path = os.path.join(output_dir, "deit_tiny_real_lrp_zplus_zennit.csv")
df_lrp_zennit.to_csv(csv_output_path, index=False)
print(f"\nResults saved to: '{csv_output_path}'")

summary_lrp_zennit = df_lrp_zennit.groupby('Method')[
    ['Absolute_Error', 'RCE_Percentage(%)', 'Leakage_Percentage(%)', 'Propagation_Continuity', 'Inference_Time(s)']
].mean()

print("\n" + "=" * 80)
print("DeiT-Tiny Zennit LRP (z+) Summary")
print("=" * 80)
print(summary_lrp_zennit.to_string())
print("=" * 80)
