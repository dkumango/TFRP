###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- matplotlib, pandas
###############################################################################
"""Figures added in the revision.

Figure 1  fig_overview      TFRP execution model: forward trace over the Keras node graph, backward
                            traversal with one generic rule per operation type, outputs.
Figure 3  fig_randomisation Cascading model-parameter randomisation on ResNet50 (E7): median Spearman
                            correlation of |phi| with the map of the trained model.

Output: exp_result/E12/fig_overview.{pdf,png}, fig_randomisation.{pdf,png}
Usage : python exp_code/e12_figures_revision.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'exp_result', 'E12')
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#d9d8d2'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                                           '#e87ba4', '#008300', '#6250d6', '#e34948')
NL = '\n'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc='#ffffff', ec=MUTED, fs=8, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, style='-|>'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=1))


def fig_overview():
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 6.8)
    ax.axis('off')

    for x, t in ((0.2, '(a) Forward trace'), (3.7, '(b) Backward traversal'), (10.3, '(c) Outputs')):
        ax.text(x, 6.55, t, fontsize=9, weight='bold', va='center')

    # (a) node graph in Keras execution order; blue = nodes that module hooks do not see
    nodes = [('input x', 5.6, False), ('Conv2D', 4.85, False), ('BatchNorm · ReLU', 4.1, False),
             ('x + y  (keras.ops)', 3.35, True), ('nested sub-model' + NL + '(expanded)', 2.4, True),
             ('GlobalAvgPool', 1.6, False), ('Dense → logit f_c', 0.85, False)]
    heights = [0.75 if NL in t else 0.55 for t, _, _ in nodes]
    for (text, y, special), h in zip(nodes, heights):
        _box(ax, 0.4, y, 2.6, h, text, fs=7.5, fc='#eef4fc' if special else '#ffffff',
             ec=BLUE if special else MUTED)
    for (_, y0, _), (_, y1, _), h1 in zip(nodes[:-1], nodes[1:], heights[1:]):
        _arrow(ax, 1.7, y0, 1.7, y1 + h1)
    _arrow(ax, 0.3, 3.1, 0.3, 4.65, color=BLUE, style='-')       # residual branch
    ax.text(1.7, 0.35, 'blue: raw tensor ops and nested models' + NL + 'are ordinary nodes of the traced graph',
            fontsize=6.5, color=BLUE, ha='center', va='center')

    # (b) generic rules per operation type
    _box(ax, 3.8, 5.55, 6.0, 0.55, 'seed: bias-free target logit f_c',
         fc='#fdf0ea', ec=ORANGE, fs=7.5)
    rules = [('Dense, Conv, Depthwise', 'z⁺ / αβ / ε by local AD'),
             ('Add, Avg / GAP pooling', 'proportional:  Rₖ = R · xₖ / Σx'),
             ('Max pooling', 'gradient routing'),
             ('Concat, Pad, Reshape', 'split / crop / reshape'),
             ('Multiply (gate)', 'signal-take-all'),
             ('BN, activations', 'identity (pass-through)'),
             ('any other operation', 'UnsupportedLayerError')]
    y = 4.95
    for op, rule in rules:
        err = 'Unsupported' in rule
        _box(ax, 3.8, y, 2.55, 0.5, op, fs=7)
        _box(ax, 6.45, y, 3.35, 0.5, rule, fs=7, fc='#fde9e9' if err else '#ffffff', ec=RED if err else MUTED)
        y -= 0.64
    ax.text(6.8, 0.35, 'reverse topological order; R from several consumers is summed;' + NL + 'weighted ops: R = x ⊙ ∇⟨f(x; w̃), s⟩,  s = R_out / z', fontsize=6.5, color=MUTED,
            ha='center', va='center')
    _arrow(ax, 3.05, 1.1, 3.75, 5.8, color=ORANGE)

    # (c) outputs
    _box(ax, 10.4, 4.6, 3.6, 0.9, 'input relevance map R(x)' + NL + '(raw, not renormalised)', fc='#eef4fc',
         ec=BLUE, fs=7.5)
    _box(ax, 10.4, 2.9, 3.6, 1.2, 'conservation log per operation:' + NL + 'ΣR in, ΣR out, loss' + NL
         + '→ locates every loss', fs=7.5)
    _box(ax, 10.4, 1.4, 3.6, 1.0, 'explicit error for unsupported' + NL + 'operations' + NL + '(no silent wrong map)',
         fc='#fde9e9', ec=RED, fs=7.5)
    for y0, y1 in ((5.2, 5.05), (3.5, 3.5), (1.15, 1.9)):
        _arrow(ax, 9.85, y0, 10.35, y1)
    _save(fig, 'fig_overview')


def fig_randomisation():
    d = pd.read_csv(os.path.join(ROOT, 'exp_result', 'E7', 'sanity_per_image.csv'))
    med = d.groupby(['method', 'depth']).spearman_abs.median().unstack(0)
    stages = d.drop_duplicates('depth').sort_values('depth').stage.tolist()
    series = [('TFRP proportional (V3)', 'TFRP z⁺ (default)', BLUE, 'o'),
              ('TFRP z+', 'TFRP z⁺ (uniform, V0)', VIOLET, 's'),
              ('TFRP composite', 'TFRP composite (V0)', AQUA, 'D'),
              ('GuidedBackprop', 'Guided Backprop', YELLOW, '^'),
              ('IntegratedGradients', 'Integrated Gradients', ORANGE, 'v'),
              ('Saliency', 'Saliency', MAGENTA, 'P'),
              ('SmoothGrad', 'SmoothGrad', GREEN, 'X'),
              ('GradCAM', 'Grad-CAM', RED, '*')]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = list(range(1, len(stages) + 1))
    for key, label, color, marker in series:
        ax.plot(x, med[key].values, color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor='white', markeredgewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['output' + NL + 'layer'] + [f'+ {s}' for s in stages[1:]])
    ax.set_xlabel('layers randomised (cumulative, from the top)')
    ax.set_ylabel('median Spearman of |φ|' + NL + 'with the trained model')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.text(1.0, -0.33, 'Higher = less sensitive to the model parameters. ResNet50, 50 images.',
            transform=ax.transAxes, ha='right', fontsize=7, color=MUTED)
    _save(fig, 'fig_randomisation')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_overview()
    fig_randomisation()
    print('saved fig_overview, fig_randomisation')
