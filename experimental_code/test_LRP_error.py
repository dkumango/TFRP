###############################################################################
# Execution Environment: Python 3.7 (.venv_37)
# Note: Requires iNNvestigate and TensorFlow 1.15.x / Keras 2.2.x static graph mode.
###############################################################################
"""LRP Error Analysis and Debugging Benchmark on ImageNet.

Environment:
    - Python Version: Python 3.7 (.venv_37)
    - Required Dependencies: iNNvestigate, TensorFlow 1.15.x / Keras 2.2.x (static graph mode)

Evaluates potential numerical instability or propagation errors in conventional LRP
(z+, epsilon, alpha-beta, composite preset) across models like VGG16.
"""

import os
import random
import gc

import numpy as np
import time
from scipy.special import softmax

import innvestigate
import eval_lib as EVAL

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50, VGG16, InceptionV3, MobileNetV2
)
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
model = VGG16(weights='imagenet')

model_type = "vgg16"
out_fname = "exp_result/lrp_vgg16_100.csv"


def set_seeds(seed=42):
    """Enforce deterministic operations and fix random seeds."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


set_seeds(42)

# Remove softmax activation to analyze pre-softmax logits
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

folder_path = "imagen_500"
file_list = []

for file_name in os.listdir(folder_path):
    full_path = os.path.join(folder_path, file_name)

    # Filter files only
    if os.path.isfile(full_path):
        file_list.append(file_name)

random.seed(42)
images = random.sample(file_list, 100)
print(images)

n_images = len(images)
if 'inception' in model.name.lower():
    img_width, img_height = (299, 299)
else:
    img_width, img_height = (224, 224)

top_classes =[]
top_class_labels =[]     # top class label
top_class_probs =[]      # top class probability
org_imgs = np.zeros((n_images, img_width, img_height, 3)) # for org image
input_imgs = np.zeros((n_images, img_width, img_height, 3)) # for preprocessed image

for i in range(n_images):
    print(i, end=",")
    img = image.load_img(folder_path+"/"+images[i], target_size=(img_width, img_height))
    x = image.img_to_array(img)
    org_imgs[i] = x.copy()
    x = preprocess_input(x)
    input_imgs[i] = x.copy()


imgs = []; global_rules =[]; results = [];  leakage_percents = []; sparsity= []; heatmap_entropy=[]; 
st_add =[]; st_padding = []; st_gap = []; times = [] ;


for i in range(n_images):
    x = np.expand_dims(input_imgs[i], axis=0)
    print(i, f"\nAnalyzing {images[i]} ")

    pred = model.predict(x)
    top_class_idx = np.argmax(pred[0])
    final_relevance_sum = pred[0][top_class_idx]

    ## z_plus #######################################################
    start_time = time.time()
    analyzer = innvestigate.create_analyzer(
        "lrp.z_plus", model, neuron_selection_mode="max_activation"
    )
    result = analyzer.analyze(x)
    end_time = time.time()

    results.append(result)
    imgs.append(images[i])
    global_rules.append('z_plus')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)


    # strucxture-tollerance test #######################################################

    if model_type == "resnet50" :
        model_st = EVAL.create_add_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.z_plus", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_logit = pred_st[0][0]
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else : 
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_add.append(new_leakage)

    if model_type in ["resnet50", "mobilenet_v2"] :
        model_st = EVAL.create_padding_test_model()

        analyzer = innvestigate.create_analyzer(
            "lrp.z_plus", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100        
    else :
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_padding.append(new_leakage)

    if model_type not in ["vgg16"] :
        model_st = EVAL.create_gap_test_model()

        analyzer = innvestigate.create_analyzer(
            "lrp.z_plus", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100        
    else :
        leakage_percent_st = None

    new_leakage = np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_gap.append(new_leakage)



    ## epsilon #######################################################
    start_time = time.time()
    analyzer = innvestigate.create_analyzer(
        "lrp.epsilon", model, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
    )
    result = analyzer.analyze(x)
    end_time = time.time()

    results.append(result)
    imgs.append(images[i])
    global_rules.append('epsilon')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)


    # strucxture-tollerance test #######################################################

    if model_type == "resnet50" :
        model_st = EVAL.create_add_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.epsilon", model_st, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else : 
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_add.append(new_leakage)

    if model_type in ["resnet50", "mobilenet_v2"] :
        model_st = EVAL.create_padding_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.epsilon", model_st, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100        
    else :
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_padding.append(new_leakage)

    if model_type not in ["vgg16"] :
        model_st = EVAL.create_gap_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.epsilon", model_st, neuron_selection_mode="max_activation", **{"epsilon": 0.01}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100        
    else :
        leakage_percent_st = None

    new_leakage = np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_gap.append(new_leakage)


    ## alpha_beta #######################################################
    start_time = time.time()
    analyzer = innvestigate.create_analyzer(
        "lrp.alpha_beta", model, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
    )
    result = analyzer.analyze(x)
    end_time = time.time()

    results.append(result)
    imgs.append(images[i])
    global_rules.append('alpha_beta')

    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)

    # strucxture-tollerance test #######################################################

    if model_type == "resnet50" :
        model_st = EVAL.create_add_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.alpha_beta", model_st, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else : 
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_add.append(new_leakage)

    if model_type in ["resnet50", "mobilenet_v2"] :
        model_st = EVAL.create_padding_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.alpha_beta", model_st, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100        
    else :
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_padding.append(new_leakage)

    if model_type not in ["vgg16"] :
        model_st = EVAL.create_gap_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.alpha_beta", model_st, neuron_selection_mode="max_activation", **{"alpha": 2, "beta": 1}
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else :
        leakage_percent_st = None

    new_leakage = np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_gap.append(new_leakage)



    ## composite #######################################################
    start_time = time.time()
    analyzer = innvestigate.create_analyzer(
        "lrp.sequential_preset_a", model, neuron_selection_mode="max_activation"
    )
    result = analyzer.analyze(x)
    end_time = time.time()

    results.append(result)
    imgs.append(images[i])
    global_rules.append('composite')
    leakage_percent = (np.sum(result) - final_relevance_sum) / (final_relevance_sum + 1e-10) * 100
    leakage_percents.append(np.abs(leakage_percent))
    sparsity.append(EVAL.relevance_sparsity(result))
    heatmap_entropy.append(EVAL.heatmap_entropy(result))
    times.append(end_time - start_time)

    # strucxture-tollerance test #######################################################

    if model_type == "resnet50" :
        model_st = EVAL.create_add_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.sequential_preset_a", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100

    else : 
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_add.append(new_leakage)

    if model_type in ["resnet50", "mobilenet_v2"] :
        model_st = EVAL.create_padding_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.sequential_preset_a", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else :
        leakage_percent_st = None

    new_leakage =  np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_padding.append(new_leakage)

    if model_type not in ["vgg16"] :
        model_st = EVAL.create_gap_test_model()
        analyzer = innvestigate.create_analyzer(
            "lrp.sequential_preset_a", model_st, neuron_selection_mode="max_activation"
        )
        result = analyzer.analyze(x)
        pred_st = model_st.predict(x) ; target_class = np.argmax(pred_st[0]); target_logit = pred_st[0][target_class]        
        leakage_percent_st = (np.sum(result) - target_logit) / (target_logit + 1e-10) * 100
    else :
        leakage_percent_st = None

    new_leakage = np.abs(leakage_percent_st) if leakage_percent_st is not None else None
    st_gap.append(new_leakage)


    import pandas as pd
    df = pd.DataFrame({
        "Image": imgs,
        "Global_Rule": global_rules,
        "Total_Leakage_(%)": leakage_percents,
        "Sparsity": sparsity,
        "Heatmap_Entropy": heatmap_entropy,
        "Execution_Time": times,
        "Structure-Tolerance_Add": st_add,
        "Structure-Tolerance_Padding": st_padding,
        "Structure-Tolerance_Gap": st_gap
    })
    print(df)
    df.to_csv(out_fname, index=False)



###############################################################
# Visualization
###############################################################
# import matplotlib.pyplot as plt

# import innvestigate.utils as iutils
# import innvestigate.utils.visualizations as ivis


# ## heatmap ----------------
# heatmap = ivis.heatmap(a[0])

# plt.imshow(heatmap, cmap="seismic")
# plt.colorbar()
# plt.show()

# # overlay --------------
# plt.imshow(org_imgs[0].astype(np.uint8))
# plt.imshow(heatmap, cmap="seismic", alpha=0.5)
# plt.axis("off")
# plt.show()


# #################################################
# from keras.models import Sequential
# from keras.layers import Conv1D, Dense, Embedding, GlobalMaxPooling1D


# # Create Keras Sequential Model
# model = Sequential()
# model.add(Embedding(input_dim=219, output_dim=8))
# model.add(Conv1D(filters=64, kernel_size=8, padding="valid", activation="relu"))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(16, activation="relu"))
# model.add(Dense(2, activation=None))

# # Analyze model
# model.predict(np.random.randint(1, 219, (1, 100)))  # [[0.04913538 0.04234646]]

# analyzer = innvestigate.create_analyzer(
#     "lrp.epsilon", model, neuron_selection_mode="max_activation", **{"epsilon": 1}
# )
# a = analyzer.analyze(np.random.randint(1, 219, (1, 100)))
# print(a[0], a[0].shape)