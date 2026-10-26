###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E1 preparation: shared image cache and Keras model export.

1. Decodes all images once (PIL, bilinear) into uint8 caches rgb_224.npy / rgb_299.npy.
2. Exports the node graph (op type, config, tensor connectivity, static shapes) and the
   weights of the four keras.applications CNNs, so that the PyTorch environment can rebuild
   the *same* network with the *same* weights (exp_code/common/keras_graph_torch.py).
3. Stores Keras logits of the first 10 images per model for cross-framework verification.

Output: <cache>/images.txt, rgb_224.npy, rgb_299.npy, models/<name>.json|.npz|_logits.npy
Usage : python exp_code/e1_prepare.py [--image_dir <image_dir>]
"""

import argparse
import json
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import keras
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'exp_code', 'common'))
import preprocess as P

A = keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3,
         'mobilenet_v2': A.MobileNetV2}
KERAS_PREP = {'vgg16': A.vgg16.preprocess_input, 'resnet50': A.resnet50.preprocess_input,
              'inception_v3': A.inception_v3.preprocess_input, 'mobilenet_v2': A.mobilenet_v2.preprocess_input}


def build_cache(image_dir):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    os.makedirs(P.CACHE_DIR, exist_ok=True)
    with open(os.path.join(P.CACHE_DIR, 'images.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(files))
    for size in (224, 299):
        arr = np.zeros((len(files), size, size, 3), dtype=np.uint8)
        for i, fname in enumerate(files):
            img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
            arr[i] = np.asarray(img.resize((size, size), Image.Resampling.BILINEAR))
        np.save(os.path.join(P.CACHE_DIR, f'rgb_{size}.npy'), arr)
        print(f'cached {len(files)} images at {size}px')


def export_model(name):
    model = CTORS[name](weights='imagenet')
    nodes, weights = [], {}
    for depth in sorted(model._nodes_by_depth, reverse=True):
        for node in model._nodes_by_depth[depth]:
            op = node.operation
            if node.is_input:
                continue
            nodes.append({
                'name': op.name,
                'type': op.__class__.__name__,
                'config': json.loads(json.dumps(op.get_config(), default=str)),
                'inputs': [t.name for t in node.input_tensors],
                'outputs': [t.name for t in node.outputs],
                'input_shapes': [list(t.shape[1:]) for t in node.input_tensors],
                'output_shape': list(node.outputs[0].shape[1:]),
            })
            for i, w in enumerate(op.get_weights()):
                weights[f'{op.name}__{i}'] = w
    spec = {'name': name, 'input': model.inputs[0].name, 'output': model.outputs[0].name, 'nodes': nodes}
    out = os.path.join(P.CACHE_DIR, 'models')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(spec, f)
    np.savez(os.path.join(out, f'{name}.npz'), **weights)

    # Verification logits (pre-softmax) for the first 10 images
    logit_model = keras.Model(model.inputs, model.get_layer('predictions').output)
    x = np.concatenate([P.model_input(name, i) for i in range(10)])
    size, _ = P.MODEL_INFO[name]
    x_keras = KERAS_PREP[name](np.asarray(P.rgb_cache(size)[:10], dtype=np.float32).copy())
    assert np.allclose(x, x_keras, atol=1e-4), 'numpy preprocessing differs from keras.applications'
    pred_layer = model.get_layer('predictions')
    pre = keras.Model(model.inputs, pred_layer.input)(x).numpy()
    np.save(os.path.join(out, f'{name}_logits.npy'), pre @ pred_layer.kernel.numpy() + pred_layer.bias.numpy())
    print(f'exported {name}: {len(nodes)} nodes, {len(weights)} weight arrays')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', default=P.IMAGE_DIR_DEFAULT)
    args = ap.parse_args()
    build_cache(args.image_dir)
    for name in CTORS:
        export_model(name)


if __name__ == '__main__':
    main()
