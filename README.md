## TFRP: TensorFlow Relevance Propagation
TFRP (TensorFlow Relevance Propagation) is a high-performance, fine-grained Explainable AI (XAI) framework that reformulates the traditional Layer-wise Relevance Propagation (LRP) into a unified automatic differentiation task.  

By leveraging the underlying computational graph of TensorFlow, TFRP provides a mathematically consistent and computationally efficient way to attribute model decisions back to input pixels, ensuring strict adherence to the conservation law across complex neural architectures.  

### Key Features
◉ Auto-Diff Based Reformulation: Replaces complex heuristic rules with a unified redistribution principle based on the product of input activations ($p_i$) and connection weights ($w_{ij}$), implemented via the automatic differentiation engine.  

◉ Architectural Versatility: Seamlessly supports complex non-linear architectures, including ResNet (Skip-connections), Inception (Parallel branches), and DenseNet (Concatenations), without manual layer-by-layer rule tuning.  

◉ High-Resolution Precision: Achieves exceptional Sparsity (up to 0.929) and Stability ($10^{-5}$ range), delivering clear and noise-free attribution maps compared to standard gradient-based methods.  

◉ Sub-second Inference: Optimized for real-time applications, generating faithful explanations in a single backward pass (0.38s - 0.82s for heavy models).  


### Benchmark Results
TFRP demonstrates superior performance across various quantitative metrics:
<img width="630" height="255" alt="image" src="https://github.com/user-attachments/assets/81201f5c-2862-4ba5-806d-2e2cfe46c249" />
(Grad-CAM (GC), Guided Backpropagation (GB), Integrated Gradients (IG), SmoothGrad (SG))  

### Citation
If you find this work useful in your research, please consider citing:  
(Under review)
