###############################################################################
# General TFRP library (Verion 1.0)
# TFRP : Tensor-Flow Relevance Propagation
# Keras tensorflow
###############################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1. Generalized Contribution Distribution Function
##############################################################################
def distribute_relevance_gen(layer, p_i, R_j, rule='z_plus', alpha=2.0, beta=1.0, epsilon=1e-1):
    """
    Generalized TFRP distribution supporting epsilon, alpha-beta, and z+ rules with Shape Alignment.
    """
    # 1. Weight handling: 레이어 타입에 따른 가중치 추출
    if hasattr(layer, 'depthwise_kernel'):
        w = layer.depthwise_kernel
    else:
        w = layer.kernel
    
    # 레이어 연산 타입(Dense/Conv2D) 정의
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

    # 2. 형상 일치(Shape Alignment)를 위한 예비 연산
    # 규칙에 따라 기준이 되는 Forward Response(z)의 형상을 파악합니다.
    if rule == 'epsilon':
        z_sample = forward_op(p_i, w)
    else:
        z_sample = forward_op(p_i, tf.maximum(w, 0.0))

    # --- [핵심] Defensive Shape Alignment 로직 ---
    # 상위 레이어의 R_j와 현재 레이어의 z_sample 간 공간적 크기가 다를 경우 자동 조정합니다.
    if len(R_j.shape) == 4 and len(z_sample.shape) == 4:
        if R_j.shape[1:3] != z_sample.shape[1:3]:
            R_j = tf.image.resize(R_j, z_sample.shape[1:3], method='nearest')

    # --- 3. 규칙별 기여도 재분배 로직 실현 ---
    
    # [Case A] LRP-Epsilon: 전체 가중치와 안정화 지수 활용
    if rule == 'epsilon':
        z_raw = forward_op(p_i, w)
        
        # --- [수정] 분모가 절대 0이 되지 않도록 보정 ---
        # 분모의 절대값이 epsilon보다 작아지지 않도록 강제 (1e-9는 언더플로우 방지).
        z = z_raw + tf.sign(z_raw + 1e-9) * epsilon
        
        s = R_j / z
        with tf.GradientTape() as tape:
            tape.watch(p_i)
            z_prime = forward_op(p_i, w)
            tmp = tf.reduce_sum(z_prime * s)
        return p_i * tape.gradient(tmp, p_i)


    # [Case B] LRP-AlphaBeta: 양수/음수 기여도의 개별적 제어[cite: 2]
    elif rule == 'alpha_beta':
        w_plus = tf.maximum(w, 0.0)
        w_minus = tf.minimum(w, 0.0)

        # Alpha Pass (Positive)[cite: 1, 2]
        z_p = forward_op(p_i, w_plus) + 1e-9
        s_p = R_j / z_p
        with tf.GradientTape() as tape_p:
            tape_p.watch(p_i)
            z_p_prime = forward_op(p_i, w_plus)
            tmp_p = tf.reduce_sum(z_p_prime * s_p)
        R_alpha = p_i * tape_p.gradient(tmp_p, p_i)

        # Beta Pass (Negative)[cite: 2]
        z_m = forward_op(p_i, w_minus) - 1e-9
        s_m = R_j / z_m
        with tf.GradientTape() as tape_m:
            tape_m.watch(p_i)
            z_m_prime = forward_op(p_i, w_minus)
            tmp_m = tf.reduce_sum(z_m_prime * s_m)
        R_beta = p_i * tape_m.gradient(tmp_m, p_i)

        return alpha * R_alpha - beta * R_beta

    # [Case C] LRP-z+ (Default): 흥분성 가중치 중심의 안정적 분산[cite: 1, 2]
    else: # z_plus
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
    """
    현대적 딥러닝 구조를 위한 통합 LRP 해석 및 누수 추적 함수
    composite_preset: dict, 예: {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}
    """
    # 1. 입력 데이터 전처리 및 텐서화[cite: 3]
    if not tf.is_tensor(input_image):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    if len(input_image.shape) == 3:
        input_image = tf.expand_dims(input_image, axis=0)


    # 2. 활성화 맵 추출 (Tensor ID 매핑)[cite: 3]
    layer_outputs = [l.output for l in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model(input_image)
    act_dict = {id(l.output): act for l, act in zip(model.layers, activations)}

    # 3. 초기 기여도(R) 설정: 예측 확률값 기준[cite: 1, 3]
    if use_logit:

        target_layer_idx = -1
        for idx in range(len(model.layers)-1, -1, -1):
            if 'Dense' in model.layers[idx].__class__.__name__:
                target_layer_idx = idx
                break
        
        if target_layer_idx == -1: raise ValueError("Dense 레이어를 찾을 수 없습니다.")
            
        target_layer = model.layers[target_layer_idx]
        input_to_dense_model = tf.keras.Model(inputs=model.input, outputs=target_layer.input)
        input_to_dense = input_to_dense_model(input_image)
        
        # [핵심 수정 1] 편향(Bias)을 제외하고 가중치 곱만으로 로짓 계산
        # 이렇게 해야 레이어별 기여도 분배 수식(R = R * (aw / z))과 수학적으로 일치합니다.
        weights = target_layer.get_weights()[0] 
        preds = tf.matmul(input_to_dense, weights)
        
        start_idx = target_layer_idx
        # [핵심 수정 2] 역전파의 시작 ID를 정확히 타겟 레이어의 출력으로 고정
        target_id = id(target_layer.output)


    else:
        preds = model(input_image)
        start_idx = len(model.layers) - 1
        target_id = id(model.layers[-1].output)
    
    if target_class_idx is None:
        target_class_idx = tf.argmax(preds[0])


    # 예측된 클래스의 확률값으로 초기화 (보존 법칙의 시작점)
    R_start = tf.one_hot([target_class_idx], preds.shape[-1]) * preds
    initial_sum = tf.reduce_sum(R_start).numpy()
    
    #rel_dict = {id(model.layers[-1].output): R_start}
    rel_dict = {target_id: R_start}

    # print(f"\n{'Layer Name':<30} | {'Upper Sum':<12} | {'Lower Sum':<12} | {'Leakage (%)':<10}")
    # print("-" * 75)

    # 4. 역전파 그래프 탐색 시작[cite: 3]
    for i in range(start_idx, -1, -1):
        layer = model.layers[i]
        layer_output_id = id(layer.output)

        if layer_output_id not in rel_dict:
            continue

        current_R = rel_dict[layer_output_id]
        # 상위에서 내려온 기여도 합산 측정
        sum_upper = tf.reduce_sum(current_R).numpy()
        
        inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        layer_type = layer.__class__.__name__

        # 레이어별 적용 규칙 결정
        active_rule = global_rule
        if composite_preset and layer_type in composite_preset:
            active_rule = composite_preset[layer_type]

        # --- 기여도 분배 실행 ---
        distributed_R_list = [] # 현재 레이어에서 분배된 R들을 임시 저장

        if 'Dense' in layer_type or 'Conv2D' in layer_type or 'DepthwiseConv2D' in layer_type:
            p_i = act_dict[id(inputs[0])]
            # Ver 6.0의 일반화 분배 함수 호출[cite: 1, 3]
            dist_R = distribute_relevance_gen(layer, p_i, current_R, 
                                              rule=active_rule, alpha=alpha, beta=beta, epsilon=epsilon)
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        elif 'Add' in layer_type:
            # 보존 법칙을 준수하는 균등 분할[cite: 1, 3]
            split_R = current_R / len(inputs)
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R
                distributed_R_list.append(split_R)

        elif 'Concatenate' in layer_type:
            # 축(axis) 기준 슬라이싱 분배[cite: 3]
            axis = layer.axis if hasattr(layer, 'axis') else -1
            start_idx = 0
            for inp in inputs:
                dim_size = inp.shape[axis]
                slice_spec = [slice(None)] * len(current_R.shape)
                slice_spec[axis] = slice(start_idx, start_idx + dim_size)
                split_R = current_R[tuple(slice_spec)]
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + split_R
                distributed_R_list.append(split_R)
                start_idx += dim_size

        elif 'ZeroPadding2D' in layer_type:
            # 패딩 크기를 확인하여 기여도 맵을 역으로 잘라냄(Cropping)[cite: 2]**
            pad = layer.padding
            (top, bottom), (left, right) = pad
            h, w = current_R.shape[1], current_R.shape[2]
            dist_R = current_R[:, top:h-bottom, left:w-right, :]
            rel_dict[id(inputs[0])] = rel_dict.get(id(inputs[0]), 0) + dist_R
            distributed_R_list.append(dist_R)

        elif 'Flatten' in layer_type:
                # 입력 텐서의 원래 형상(예: [1, 7, 7, 512])을 가져옵니다.
                input_shape = inputs[0].shape
                # 2차원 기여도를 다시 4차원 형상으로 변환합니다.
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

        else: # Identity transfer (BatchNormalization, Flatten 등)[cite: 3]
            for inp in inputs:
                rel_dict[id(inp)] = rel_dict.get(id(inp), 0) + current_R
                distributed_R_list.append(current_R)

        # --- 누수 추적 계산 및 출력 ---
        sum_lower = sum([tf.reduce_sum(r).numpy() for r in distributed_R_list])
        leakage_pc = (abs(sum_upper - sum_lower) / (abs(sum_upper) + 1e-10)) * 100
        
        # 가독성을 위해 레이어 이름이 길 경우 생략
        # display_name = (layer.name[:27] + '..') if len(layer.name) > 27 else layer.name
        # print(f"{display_name:<30} | {sum_upper:12.6f} | {sum_lower:12.6f} | {leakage_pc:9.4f}%")


    # 5. 최종 결과 확인
    final_input_R = rel_dict[id(model.input)]
    final_sum = tf.reduce_sum(final_input_R).numpy()

    # 최종 반환 전 보정 로직 추가 (global normalization)
    correction_factor = initial_sum / (final_sum + 1e-10)
    # 모든 픽셀에 동일한 비율로 보정값 곱하기
    final_input_R = rel_dict[id(model.input)] * correction_factor
    
    print("-" * 75)
    print(f"Initial Prediction Sum: {initial_sum:.6f}")
    print(f"Final Relevance Sum:    {final_sum:.6f}")
    print(f"Total Network Leakage:  {((initial_sum - final_sum)/initial_sum)*100:.6f}%")
    
    return final_input_R.numpy(), initial_sum, final_sum, ((initial_sum - final_sum)/initial_sum)*100

########################################################################################################
# VISUALIZE
########################################################################################################
##################################################################################
from skimage.segmentation import slic
from skimage.util import img_as_float
import cv2

def apply_superpixel_pooling(original_image, relevance_map, n_segments=200, compactness=10):
    # 1. 텐서라면 넘파이 배열로 변환
    if tf.is_tensor(relevance_map):
        relevance_map = relevance_map.numpy()

    # 2. 차원 축소: (1, H, W, 1) -> (H, W)
    # np.squeeze는 1인 차원을 모두 제거하므로 (1, 224, 224, 1) -> (224, 224)가 됩니다.
    heatmap_2d = np.squeeze(relevance_map)

    # 만약 채널이 3개여서 (H, W, 3)으로 남았다면 평균을 내서 (H, W)로 만듦
    if len(heatmap_2d.shape) == 3:
        heatmap_2d = np.mean(heatmap_2d, axis=-1)

    # 3. 입력값 범위 디버깅 (선택)
    print(f"[Superpixel] Input Range: Min={heatmap_2d.min():.4f}, Max={heatmap_2d.max():.4f}")

    # 4. SLIC 슈퍼픽셀 분할 (원본 이미지 기준)
    # original_image도 (H, W, 3) 형태여야 함
    if tf.is_tensor(original_image):
        original_image = original_image.numpy()
    original_image = np.squeeze(original_image) # 혹시 모를 배치 차원 제거

    segments = slic(img_as_float(original_image), n_segments=n_segments, compactness=compactness, start_label=1)

    refined_map = np.zeros_like(heatmap_2d)

    # 5. 슈퍼픽셀 평균 할당
    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        refined_map[mask] = np.mean(heatmap_2d[mask])

    # 6. 결과 정규화 (0~1)
    if refined_map.max() > 0:
        refined_map = refined_map / refined_map.max()

    # 7. 차원 복구: (H, W) -> (1, H, W, 1)
    # 입력 형태와 동일하게 맞춰줍니다.
    h, w = refined_map.shape
    return refined_map.reshape(1, h, w, 1)


##################################################
def get_gradcam_pure_tf_fixed(model, img_array, layer_name, class_idx):
    # 1. Functional 모델에서 타겟 레이어와 최종 출력 뽑기
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # 2. 입력을 텐서로 변환
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # ⭐️ [가장 중요한 마법의 1줄] 
        # 모델 가중치가 유실되었더라도, 첫 출발점인 입력을 감시하면 파이프라인 전체가 강제 기록됩니다!
        tape.watch(img_tensor) 
        
        # 3. 모델 통과
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]
        
    # 4. 기울기 계산 (감시 카메라가 켜졌으므로 절대 None이 나오지 않습니다!)
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError("The gradient flow has been disconnected. Please verify that the target layer name is correct.")
        #raise ValueError("기울기가 끊겼습니다. 타겟 레이어 이름이 정확한지 확인하세요.")
        
    # 5. Grad-CAM 수식 계산
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 6. 음수 제거(ReLU) 및 0~1 정규화
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


####################
##################################################
def _visualize_refined_heatmap2(original_image, relevance_map, alpha=0.5, sigma=5):
    """
    점 형태의 히트맵을 영역 기반(Region-based)으로 확장하여 시각화합니다.
    """
    relevance_map = np.array(relevance_map)

    # 1. 전처리: 데이터 형식 맞추기
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

    # 2. 영역 확장을 위한 알고리즘 적용

    # (1) R+ 추출 및 Outlier 제거 (너무 튀는 값 억제)
    res_plus = np.maximum(relevance_map, 0)
    # 상위 1% 값을 99% 지점의 값으로 클리핑하여 특정 점만 타는 현상 방지
    v_max = np.percentile(res_plus, 99)    # 99 -> 95
    res_clipped = np.clip(res_plus, 0, v_max)

    # (2) 로그 스케일링 (Log Scaling)
    # 값의 동적 범위를 압축하여 약한 기여도 영역도 시각화에 참여하게 함
    res_log = np.log1p(res_clipped)

    # (3) 모폴로지 팽창 (Dilation)
    # 작은 점들을 주변으로 확장시켜 영역화함
    kernel = np.ones((3, 3), np.uint8)            #(5,5) -> (3,3)
    res_dilated = cv2.dilate(res_log, kernel, iterations=1)

    # (4) 강한 가우시안 블러 (Large Sigma)
    # sigma 값을 높이고 커널 크기를 확장하여 부드러운 영역 형성
    # sigma가 클수록 점들이 서로 연결되어 영역처럼 보입니다.
    res_smoothed = cv2.GaussianBlur(res_dilated, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # (5) 배경 노이즈 제거 (Thresholding)
    # 하위 30~40% 정도만 제거하여 영역의 형태를 최대한 보존
    threshold = np.percentile(res_smoothed, 80)   # 40
    res_refined = np.where(res_smoothed >= threshold, res_smoothed, 0)

    # 3. 정규화 및 컬러맵 적용
    if res_refined.max() > 0:
        res_norm = (res_refined - res_refined.min()) / (res_refined.max() - res_refined.min() + 1e-10)
    else:
        res_norm = res_refined

    res_uint8 = (res_norm * 255).astype(np.uint8)

    # COLORMAP_JET 대신 조금 더 부드러운 COLORMAP_PARULA나 COLORMAP_TURBO 권장 (OpenCV 버전에 따라 다름)
    heatmap_color = cv2.applyColorMap(res_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 4. Overlay 생성
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)

    # # 5. 결과 시각화
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(original_image)
    # plt.axis('off')

    # plt.subplot(1, 3, 2)
    # plt.title(f"Region-based Heatmap (sigma={sigma})")
    # plt.imshow(res_norm, cmap='jet')
    # plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title("Overlay Result")
    # plt.imshow(overlay)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    return res_norm, overlay



####################

def visualize_TFRP_new(model,original_image, model_input, relevance_map, top_class, grad_th=0.2,
                       alpha=0.5, sigma=1, segment_n=400, step=3):
    """
    점 형태의 히트맵을 영역 기반(Region-based)으로 확장하여 시각화합니다.
    """

    if len(original_image.shape) ==4:
        img_h = original_image.shape[1]
        img_w = original_image.shape[2]
    else:
        img_h = original_image.shape[0]
        img_w = original_image.shape[1]

    print(f"Original Image Size: {img_h}x{img_w}")

    if step >= 3:
        # apply grad cam
        # gradCAM mask

        new_model = model  

        #score = CategoricalScore([top_class])
        #gradcam = Gradcam(new_model, model_modifier=ReplaceToLinear(), clone=True)

        #cam = gradcam(score, model_input, penultimate_layer=-1)  # 마지막 conv layer
        ############################
        target_layer_name = None
        last_conv_name = None 

        # 1. 가장 마지막 'convolution' 연산의 이름을 찾습니다.
        for layer in new_model.layers:
            if 'convolution' in layer.name.lower() or 'conv' in layer.name.lower():
                last_conv_name = layer.name

        if last_conv_name is None:
            #raise ValueError("모델 내부에서 Conv 레이어를 전혀 찾을 수 없습니다!")
            raise ValueError("No convolutional layers could be found inside the model!")
        
        # 2. 찾은 마지막 convolution 바로 다음에 오는 'relu' (활성화) 레이어를 찾습니다.
        found_last_conv = False
        for layer in new_model.layers:
            if layer.name == last_conv_name:
                found_last_conv = True
                continue # convolution 다음 레이어부터 검사
            if found_last_conv and 'relu' in layer.name.lower():
                target_layer_name = layer.name
                break

        # 만약 relu를 못 찾았다면, convolution 자체를 타겟으로 씁니다.
        if target_layer_name is None:
            target_layer_name = last_conv_name

        safe_model_input = model_input.copy()
        if safe_model_input.ndim == 3: # (28, 28, 1) 이라면
            safe_model_input = safe_model_input[np.newaxis, ...] # (1, 28, 28, 1) 로 변환
        elif safe_model_input.ndim == 2: # 혹시 (28, 28) 이라면
            safe_model_input = safe_model_input[np.newaxis, ..., np.newaxis]

        cam = get_gradcam_pure_tf_fixed(new_model, safe_model_input, target_layer_name, top_class)
        ############################
        gradcam_map = cv2.resize(cam, (img_h, img_w)) # 이미지 크기에 맞춤
        gradcam_map = gradcam_map[:, :, np.newaxis]   # (h,w,1)

        srm_pos = relevance_map.copy()
        srm_pos = np.maximum(srm_pos, 0)

        # 정규화
        pixel_map = (srm_pos  - srm_pos .min()) / (srm_pos.max() - srm_pos.min())
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min())

        gradcam_map = gradcam_map[np.newaxis, :, :,:]

        threshold = grad_th
        gradcam_map[gradcam_map < threshold] = 0

        # # 4. 결합 (Element-wise Multiplication)
        relevance_map = pixel_map * gradcam_map

    if step >= 4:
        # super pixel
        relevance_map = apply_superpixel_pooling(original_image, relevance_map, n_segments=segment_n)

    res_norm, overlay = _visualize_refined_heatmap2(original_image, relevance_map, alpha=alpha, sigma=sigma)

    return res_norm, overlay