###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""Generalized TFRP, revision v2 (core propagation only).

Changes with respect to general_TFRP_v1.py (earlier version, exp_code/tfrp)
--------------------------------------------------------------
1. z+ on signed inputs (mean-subtracted pixels) uses w+x+ + w-x- (SIGNED_INPUT_ZPLUS).
   v1 used w+x, whose denominator can approach zero or turn negative -> extreme outlier pixels.
2. DepthwiseConv2D is dispatched to tf.nn.depthwise_conv2d. v1 tested `'Conv2D' in name`
   first, so depthwise layers were evaluated with a regular conv2d (wrong forward response).
3. Dense on rank>2 inputs uses tensordot; Conv2D/DepthwiseConv2D honour dilation_rate;
   grouped Conv2D is supported (groups == channels maps to a depthwise conv).
4. Operation dispatch is by exact class name. An operation that is neither handled nor a
   shape-preserving unary op raises UnsupportedLayerError instead of being passed through
   (v1 copied the full relevance to every input of an unknown layer, e.g. Multiply).
5. Silent spatial resizing of relevance ("defensive alignment") is removed; a shape
   mismatch now raises, because it always indicates a wrong forward response.
6. The returned map is the raw propagated relevance by default (renormalize=False).
   v1 always rescaled the map to the output score, which hides conservation errors.
7. The traversal follows Keras' own node graph (the same order Functional uses to run the
   model) instead of model.layers. Raw tensor operations (keras.ops arithmetic such as a
   residual `x + y`) are visited like layers, and nested Functional/Sequential sub-models
   are expanded in place.
8. Generic rules added for operation types v1 could not handle:
   - AveragePooling2D / GlobalMaxPooling2D : gradient distribution, as for MaxPooling2D and GAP
   - Multiply                              : signal-take-all -- the input with the output's shape
                                             receives all relevance, the broadcast gate none
                                             (Arras et al., 2017); a constant factor is a unary op
   - keras.ops Add                         : uniform split, identical to the Add layer

9. z+ uses the standard positive part max(w, 0) (v1: max(w, 1e-9), see ZPLUS_WEIGHT_FLOOR), and
   alpha-beta on signed inputs uses the standard sign-split form.

10. Residual additions and average / global-average pooling redistribute relevance in proportion to
    the contributing activations by default (add_rule / avgpool_rule = 'proportional'). v1 split
    uniformly (R/n per branch, R/n per window position), which assigns relevance to inactive branches
    and positions; the v1 behaviour remains available as add_rule='uniform', avgpool_rule='gradient'.

11. Optional bias-aware fallback for z+ (ZPLUS_FALLBACK = 'flat' | 'w2', default None = standard z+):
    relevance of units with z+ = 0, which standard z+ leaves to the bias, is redistributed
    input-independently over the unit's receptive field (E8).

The epsilon rule and the composite presets are unchanged from v1.
"""

import keras
import tensorflow as tf

# z+ on signed inputs (e.g. mean-subtracted pixels at the first layer) uses w+x+ + w-x-.
# Set to False to reproduce the legacy w+x rule (unstable denominator on signed inputs).
SIGNED_INPUT_ZPLUS = True

# Floor applied to negative weights in z+. The standard rule uses max(w, 0) (default 0.0).
# v1 used max(w, 1e-9): for a unit whose positive contributions are all zero (e.g. a depthwise
# kernel with only negative weights) this silently redistributed its relevance in proportion to
# the inputs instead of letting the bias absorb it. Kept as an option for the ablation (E3).
ZPLUS_WEIGHT_FLOOR = 0.0

# Optional bias-aware fallback for z+ (E8). A unit with z+ = 0 but non-zero relevance is active only
# through its bias; standard z+ (None, default) lets the bias absorb that relevance. 'flat' sends it
# uniformly to the unit's receptive field, 'w2' in proportion to w^2 (both input-independent and
# conservative). Only units with z+ = 0 are affected; all other units keep the z+ rule.
ZPLUS_FALLBACK = None

WEIGHTED_LAYERS = ('Dense', 'Conv2D', 'DepthwiseConv2D')
GRADIENT_LAYERS = ('MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D')


class UnsupportedLayerError(NotImplementedError):
    """Raised when TFRP has no propagation rule for an operation type."""


##############################################################################
# 1. Contribution distribution for weighted layers
##############################################################################
def _forward_op(layer):
    """Return f(x, w) that reproduces the layer's linear map without bias."""
    name = layer.__class__.__name__
    if name == 'Dense':
        return lambda x, w: tf.tensordot(x, w, axes=[[x.shape.rank - 1], [0]])
    if name == 'Conv2D':
        conv = lambda x, w: tf.nn.conv2d(x, w, strides=layer.strides, padding=layer.padding.upper(),
                                         dilations=layer.dilation_rate)
        groups = getattr(layer, 'groups', 1)
        if groups == 1:
            return conv
        if groups == layer.input.shape[-1]:
            # Depthwise written as a grouped conv (e.g. ConvNeXt): kernel (kh, kw, 1, C*m) maps to the
            # depthwise kernel (kh, kw, C, m) with identical output-channel order (c*m + k).
            strides = [1, layer.strides[0], layer.strides[1], 1]
            return lambda x, w: tf.nn.depthwise_conv2d(
                x, tf.reshape(w, (w.shape[0], w.shape[1], groups, w.shape[3] // groups)),
                strides=strides, padding=layer.padding.upper(), dilations=layer.dilation_rate)
        # General grouped conv: convolve each channel group separately (CPU has no grouped conv2d)
        return lambda x, w: tf.concat([conv(xg, wg) for xg, wg in
                                       zip(tf.split(x, groups, axis=-1), tf.split(w, groups, axis=-1))], axis=-1)
    if name == 'DepthwiseConv2D':
        strides = [1, layer.strides[0], layer.strides[1], 1]
        return lambda x, w: tf.nn.depthwise_conv2d(x, w, strides=strides, padding=layer.padding.upper(),
                                                   dilations=layer.dilation_rate)
    raise UnsupportedLayerError(f'{layer.name}: no linear map for {name}')


def _kernel(layer):
    return layer.depthwise_kernel if hasattr(layer, 'depthwise_kernel') else layer.kernel


def distribute_relevance_gen(layer, p_i, R_j, rule='z_plus', alpha=2.0, beta=1.0, epsilon=1e-1):
    """Distribute upper-layer relevance R_j to the lower-layer activation p_i."""
    forward_op = _forward_op(layer)
    w = _kernel(layer)

    if rule == 'epsilon':
        z_raw = forward_op(p_i, w)
        z = z_raw + tf.sign(z_raw + 1e-9) * epsilon
        _check_shape(layer, z, R_j)
        s = R_j / z
        with tf.GradientTape() as tape:
            tape.watch(p_i)
            tmp = tf.reduce_sum(forward_op(p_i, w) * s)
        return p_i * tape.gradient(tmp, p_i)

    if rule == 'alpha_beta':
        w_plus = tf.maximum(w, 0.0)
        w_minus = tf.minimum(w, 0.0)
        if SIGNED_INPUT_ZPLUS and tf.reduce_min(p_i) < 0:
            # Standard alpha-beta for signed inputs: activating part w+x+ + w-x-,
            # inhibiting part w-x+ + w+x- (identical to the form below when x >= 0).
            x_plus, x_minus = tf.maximum(p_i, 0.0), tf.minimum(p_i, 0.0)
            return (alpha * _split_share(forward_op, (x_plus, w_plus), (x_minus, w_minus), R_j, layer, 1e-9)
                    - beta * _split_share(forward_op, (x_plus, w_minus), (x_minus, w_plus), R_j, layer, -1e-9))
        z_p = forward_op(p_i, w_plus) + 1e-9
        _check_shape(layer, z_p, R_j)
        s_p = R_j / z_p
        with tf.GradientTape() as tape_p:
            tape_p.watch(p_i)
            tmp_p = tf.reduce_sum(forward_op(p_i, w_plus) * s_p)
        R_alpha = p_i * tape_p.gradient(tmp_p, p_i)

        z_m = forward_op(p_i, w_minus) - 1e-9
        s_m = R_j / z_m
        with tf.GradientTape() as tape_m:
            tape_m.watch(p_i)
            tmp_m = tf.reduce_sum(forward_op(p_i, w_minus) * s_m)
        R_beta = p_i * tape_m.gradient(tmp_m, p_i)
        return alpha * R_alpha - beta * R_beta

    # z_plus
    w_plus = tf.maximum(w, ZPLUS_WEIGHT_FLOOR)

    # Signed inputs (e.g. the mean-subtracted image at the first layer): w+x+ + w-x-
    # keeps every contribution non-negative, so z cannot approach zero or flip sign.
    # For non-negative inputs (post-ReLU layers) the w-x- term vanishes and this
    # reduces to the standard w+x form below.
    if SIGNED_INPUT_ZPLUS and tf.reduce_min(p_i) < 0:
        x_plus, x_minus, w_minus = tf.maximum(p_i, 0.0), tf.minimum(p_i, 0.0), tf.minimum(w, 0.0)
        R_dead = None
        if ZPLUS_FALLBACK:
            z_s = forward_op(x_plus, w_plus) + forward_op(x_minus, w_minus)
            dead = z_s <= 0                    # no positive contribution: unit driven by its bias only
            R_dead = tf.where(dead, R_j, 0.0)
            R_j = tf.where(dead, 0.0, R_j)
        R_i = _split_share(forward_op, (x_plus, w_plus), (x_minus, w_minus), R_j, layer, 1e-9)
        if R_dead is not None:
            R_i = R_i + _fallback_share(forward_op, w, R_dead, tf.ones_like(p_i))
        return R_i

    z = forward_op(p_i, w_plus) + 1e-9
    _check_shape(layer, z, R_j)
    if ZPLUS_FALLBACK:
        dead = z <= 1e-9                       # z+ = 0: unit driven by its bias only
        R_dead = tf.where(dead, R_j, 0.0)
        R_j = tf.where(dead, 0.0, R_j)
    s = R_j / z
    with tf.GradientTape() as tape:
        tape.watch(p_i)
        tmp = tf.reduce_sum(forward_op(p_i, w_plus) * s)
    R_i = p_i * tape.gradient(tmp, p_i)
    if ZPLUS_FALLBACK:
        R_i = R_i + _fallback_share(forward_op, w, R_dead, tf.ones_like(p_i))
    return R_i


def _fallback_share(forward_op, w, R_dead, ones):
    """Input-independent redistribution of R_dead: R_i = sum_j v_ij / sum_i v_ij * R_j, v = 1 or w^2."""
    if ZPLUS_FALLBACK == 'flat':
        v = tf.ones_like(w)
    elif ZPLUS_FALLBACK == 'w2':
        v = tf.square(w)
    else:
        raise ValueError(f'unknown ZPLUS_FALLBACK {ZPLUS_FALLBACK!r}')
    n = forward_op(ones, v)                    # sum of v over each unit's receptive field (padding excluded)
    s = tf.where(n > 0, R_dead / tf.where(n > 0, n, 1.0), 0.0)
    with tf.GradientTape() as tape:
        tape.watch(ones)
        tmp = tf.reduce_sum(forward_op(ones, v) * s)
    return tape.gradient(tmp, ones)


def _split_share(forward_op, part_a, part_b, R_j, layer, stabilizer):
    """Distribute R_j over two (input, weight) parts in proportion to their joint response.

    R_i = x_a * d/dx_a[f(x_a, w_a) s] + x_b * d/dx_b[f(x_b, w_b) s],  s = R_j / (f_a + f_b + stab).
    The two inputs are disjoint sign components of one activation, so their sum is the
    relevance of that activation.
    """
    (x_a, w_a), (x_b, w_b) = part_a, part_b
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_a, x_b])
        z_a = forward_op(x_a, w_a)
        z_b = forward_op(x_b, w_b)
        _check_shape(layer, z_a, R_j)
        s = tf.stop_gradient(R_j / (z_a + z_b + stabilizer))
        tmp_a = tf.reduce_sum(z_a * s)
        tmp_b = tf.reduce_sum(z_b * s)
    R_i = x_a * tape.gradient(tmp_a, x_a) + x_b * tape.gradient(tmp_b, x_b)
    del tape
    return R_i


def _check_shape(layer, z, R_j):
    if tuple(z.shape) != tuple(R_j.shape):
        raise ValueError(f'{layer.name}: forward response {tuple(z.shape)} does not match '
                         f'relevance {tuple(R_j.shape)}')


##############################################################################
# 2. Forward trace over the Keras node graph
##############################################################################
def _as_functional(model):
    """Functional graph of a model (Sequential models keep one in `_functional`)."""
    if isinstance(model, keras.Sequential):
        functional = getattr(model, '_functional', None)
        if functional is None:
            raise UnsupportedLayerError(f'{model.name}: Sequential model without a defined input shape')
        return functional
    return model


def _trace(model, x):
    """Run the model node by node, recording every value and every executed step.

    Nodes are visited in the order keras' Functional uses (decreasing depth), so raw tensor
    operations are included. A nested model is expanded in place; ('alias', out, in) steps
    connect its inputs and outputs so that relevance can flow across the boundary.
    """
    values, steps = {}, []

    def run(functional, input_values):
        for t, v in zip(functional.inputs, input_values):
            values[id(t)] = v
        for depth in sorted(functional._nodes_by_depth, reverse=True):
            for node in functional._nodes_by_depth[depth]:
                op = node.operation
                if node.is_input or op is None:
                    continue
                if isinstance(op, keras.Model):
                    inner = _as_functional(op)
                    for outer_in, inner_in in zip(node.input_tensors, inner.inputs):
                        steps.append(('alias', inner_in, outer_in))
                    run(inner, [values[id(t)] for t in node.input_tensors])
                    for outer_out, inner_out in zip(node.outputs, inner.outputs):
                        values[id(outer_out)] = values[id(inner_out)]
                        steps.append(('alias', outer_out, inner_out))
                    continue
                args, kwargs = node.arguments.fill_in(values)
                outputs = tf.nest.flatten(op(*args, **kwargs))
                for t, v in zip(node.outputs, outputs):
                    values[id(t)] = v
                steps.append(('op', node))

    model_fn = _as_functional(model)
    if len(model_fn.inputs) != 1:
        raise UnsupportedLayerError(f'{model.name}: only single-input models are supported')
    run(model_fn, [x])
    return values, steps, model_fn.inputs[0]


def _shape(t):
    return tuple(t.shape[1:])


##############################################################################
# 2b. Generic AD-derived rule (exploratory option, generic_rule='ig')
##############################################################################
IG_LINEAR_ACTIVATIONS = ('linear', 'relu')


def _is_ig_node(op, n_inputs):
    """Nodes whose default treatment is a heuristic: gates and non-ReLU nonlinear unary operations."""
    otype = op.__class__.__name__
    if otype == 'Multiply' and n_inputs > 1:
        return True
    if otype == 'Activation':
        act = op.get_config().get('activation')
        return isinstance(act, str) and act not in IG_LINEAR_ACTIVATIONS
    return otype in ('LayerNormalization',)


def _node_ig(node, values, R, steps):
    """Node-level path-integrated rule: R_k = x_k * mean_a d/dx_k < f(a x), s >,  s = R / (f(x) - f(0)).

    By the completeness of the path integral, sum_k R_k = sum_j R_j (f_j(x) - f_j(0)) / (f_j(x) - f_j(0))
    = sum_j R_j up to the quadrature error (midpoint rule) for any differentiable operation f; for a
    linear f it equals LRP-0. The baseline is the all-zero input of the node.
    """
    ins = list(node.input_tensors)
    xs = [values[id(t)] for t in ins]

    def call(inputs):
        args, kwargs = node.arguments.fill_in({id(t): v for t, v in zip(ins, inputs)})
        return tf.nest.flatten(node.operation(*args, **kwargs))[0]

    diff = call(xs) - call([tf.zeros_like(x) for x in xs])
    s = tf.stop_gradient(R / (diff + tf.where(diff >= 0, 1e-9, -1e-9)))
    total = [tf.zeros_like(x) for x in xs]
    for k in range(steps):
        scaled = [(k + 0.5) / steps * x for x in xs]
        with tf.GradientTape() as tape:
            tape.watch(scaled)
            tmp = tf.reduce_sum(call(scaled) * s)
        grads = tape.gradient(tmp, scaled)
        total = [t_ + (g if g is not None else 0.0) for t_, g in zip(total, grads)]
    return [x * t_ / steps for x, t_ in zip(xs, total)]


##############################################################################
# 3. Backward relevance propagation
##############################################################################
def get_relevance_map_generalized(model, input_image, target_class_idx=None, use_logit=True,
                                  global_rule='z_plus', composite_preset=None,
                                  alpha=2.0, beta=1.0, epsilon=1e-1, renormalize=False,
                                  return_layer_log=False, verbose=False,
                                  add_rule='proportional', avgpool_rule='proportional',
                                  generic_rule=None, ig_steps=32):
    """Propagate the target score back to the input through a Keras functional graph.

    add_rule     : 'proportional' (default, R * x_k / sum_k x_k) or 'uniform' (R / n per branch, v1)
    avgpool_rule : 'proportional' (default, R_i proportional to x_i) or 'gradient' (uniform within the
                   window, v1); applies to AveragePooling2D and GAP, not MaxPooling2D
    generic_rule : None (default) or 'ig' (exploratory): multiplicative gates, non-ReLU activations and
                   layer normalisation use the node-level path-integrated rule (_node_ig, ig_steps
                   midpoint steps) instead of signal-take-all / identity
    The proportional defaults follow the contribution principle of TFRP (contribution of an input
    proportional to activation x weight; the weights of a sum are 1 and of a mean 1/n) and coincide
    with the LRP-0 / Norm rule of standard LRP implementations (ablation E3).

    Returns:
        (input_R, initial_sum, final_sum, total_leakage_pct[, layer_log])
        input_R is the raw propagated relevance unless renormalize=True.
        layer_log lists (operation name, type, sum of upper R, sum of distributed R) per step.
    """
    if not tf.is_tensor(input_image):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    if len(input_image.shape) == 3:
        input_image = tf.expand_dims(input_image, axis=0)

    values, steps, model_input = _trace(model, input_image)

    if use_logit:
        dense_steps = [i for i, s in enumerate(steps)
                       if s[0] == 'op' and s[1].operation.__class__.__name__ == 'Dense']
        if not dense_steps:
            raise ValueError('No Dense layer found in model for logit extraction.')
        start = dense_steps[-1]
        node = steps[start][1]
        dense_in = values[id(node.input_tensors[0])]
        preds = tf.matmul(dense_in, node.operation.kernel)  # bias-free logit
        target_id = id(node.outputs[0])
    else:
        start = len(steps) - 1
        last = steps[start]
        target_id = id(last[1].outputs[0]) if last[0] == 'op' else id(last[1])
        preds = values[target_id]

    if target_class_idx is None:
        target_class_idx = int(tf.argmax(preds[0]))

    R_start = tf.one_hot([target_class_idx], preds.shape[-1]) * preds
    initial_sum = float(tf.reduce_sum(R_start))
    rel = {target_id: R_start}
    layer_log = []

    def add(t, r):
        rel[id(t)] = rel.get(id(t), 0) + r

    for kind, *payload in reversed(steps[:start + 1]):
        if kind == 'alias':
            out_t, in_t = payload
            if id(out_t) in rel:
                add(in_t, rel.pop(id(out_t)))
            continue

        node = payload[0]
        out_t = node.outputs[0]
        if id(out_t) not in rel:
            continue
        op = node.operation
        otype = op.__class__.__name__
        name = getattr(op, 'name', otype)
        inputs = list(node.input_tensors)
        current_R = rel[id(out_t)]
        rule = composite_preset.get(otype, global_rule) if composite_preset else global_rule
        distributed = []

        if generic_rule == 'ig' and _is_ig_node(op, len(inputs)):
            for inp, r in zip(inputs, _node_ig(node, values, current_R, ig_steps)):
                add(inp, r)
                distributed.append(r)

        elif otype in WEIGHTED_LAYERS and isinstance(op, keras.layers.Layer):
            dist_R = distribute_relevance_gen(op, values[id(inputs[0])], current_R,
                                              rule=rule, alpha=alpha, beta=beta, epsilon=epsilon)
            add(inputs[0], dist_R)
            distributed.append(dist_R)

        elif otype == 'Add' and len(inputs) > 1:  # Add layer or keras.ops add
            if any(_shape(t) != _shape(out_t) for t in inputs):
                raise UnsupportedLayerError(f'{name}: broadcasting Add {[_shape(t) for t in inputs]}')
            if add_rule == 'proportional':
                xs = [values[id(t)] for t in inputs]
                z = tf.add_n(xs)
                s = current_R / (z + tf.where(z >= 0, 1e-9, -1e-9))
                for inp, x_k in zip(inputs, xs):
                    add(inp, x_k * s)
                    distributed.append(x_k * s)
            else:
                split_R = current_R / len(inputs)   # uniform split across residual branches
                for inp in inputs:
                    add(inp, split_R)
                    distributed.append(split_R)

        elif otype == 'Multiply' and len(inputs) > 1:
            # Gating (e.g. squeeze-and-excitation): the signal keeps all relevance, the gate none.
            signal = [t for t in inputs if _shape(t) == _shape(out_t)]
            if len(signal) != 1:
                raise UnsupportedLayerError(f'{name}: Multiply without a unique signal input '
                                            f'{[_shape(t) for t in inputs]} -> {_shape(out_t)}')
            add(signal[0], current_R)
            distributed.append(current_R)

        elif otype == 'Concatenate' and isinstance(op, keras.layers.Layer):
            axis = op.axis
            start_idx = 0
            for inp in inputs:
                size = inp.shape[axis]
                spec = [slice(None)] * len(current_R.shape)
                spec[axis] = slice(start_idx, start_idx + size)
                split_R = current_R[tuple(spec)]
                add(inp, split_R)
                distributed.append(split_R)
                start_idx += size

        elif otype == 'ZeroPadding2D':
            (top, bottom), (left, right) = op.padding
            h, w = current_R.shape[1], current_R.shape[2]
            dist_R = current_R[:, top:h - bottom, left:w - right, :]
            add(inputs[0], dist_R)
            distributed.append(dist_R)

        elif otype in ('Flatten', 'Reshape'):
            dist_R = tf.reshape(current_R, [-1] + list(_shape(inputs[0])))
            add(inputs[0], dist_R)
            distributed.append(dist_R)

        elif otype in GRADIENT_LAYERS:
            p_i = values[id(inputs[0])]
            proportional = avgpool_rule == 'proportional' and 'Average' in otype
            if proportional:
                z = op(p_i)
                weight = current_R / (z + tf.where(z >= 0, 1e-9, -1e-9))
            else:
                weight = current_R
            with tf.GradientTape() as tape:
                tape.watch(p_i)
                tmp = tf.reduce_sum(op(p_i) * weight)
            dist_R = tape.gradient(tmp, p_i)
            if proportional:
                dist_R = p_i * dist_R
            add(inputs[0], dist_R)
            distributed.append(dist_R)

        elif len(inputs) == 1 and _shape(inputs[0]) == _shape(out_t):
            # Shape-preserving unary op (BatchNormalization, Activation, LayerNormalization,
            # Rescaling, LayerScale, Dropout, arithmetic with a constant, ...)
            add(inputs[0], current_R)
            distributed.append(current_R)

        else:
            raise UnsupportedLayerError(
                f'{name}: no TFRP propagation rule for {otype} '
                f'({len(inputs)} input(s), {[_shape(t) for t in inputs]} -> {_shape(out_t)})')

        layer_log.append((name, otype, float(tf.reduce_sum(current_R)),
                          float(sum(tf.reduce_sum(r) for r in distributed))))

    final_input_R = rel[id(model_input)]
    final_sum = float(tf.reduce_sum(final_input_R))
    total_leakage = (initial_sum - final_sum) / (initial_sum + 1e-10) * 100
    if renormalize:
        final_input_R = final_input_R * (initial_sum / (final_sum + 1e-10))
    if verbose:
        print(f'Initial {initial_sum:.6f} | Final {final_sum:.6f} | Leakage {total_leakage:.6f}%')

    out = (final_input_R.numpy(), initial_sum, final_sum, total_leakage)
    return out + (layer_log,) if return_layer_log else out
