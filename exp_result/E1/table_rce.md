# E1 median RCE (%) [bootstrap 95% CI of the median]

### z+

| implementation | vgg16 | resnet50 | inception_v3 | mobilenet_v2 |
|---|---|---|---|---|
| TFRP v2 | 9.6e-06 [9.2e-06, 1e-05] | 1.1e-05 [1.1e-05, 1.2e-05] | 1.1e-05 [1e-05, 1.2e-05] | 1e+02 [1e+02, 1e+02] |
| Zennit (canonized) | 1.8e-05 [1.7e-05, 1.9e-05] | 0.0041 [0.004, 0.0041] | 0.0011 [0.0011, 0.0011] | 1e+02 [1e+02, 1e+02] |
| Zennit (not canonized) | 1.8e-05 [1.7e-05, 1.9e-05] | 6.6e+06 [6.6e+06, 6.6e+06] | 0.0015 [0.0015, 0.0015] | 35 [32, 37] |
| Captum | 6.2 [5.9, 6.5] | 8.7e-06 [8.2e-06, 9.3e-06] | 8.2e-06 [7.9e-06, 8.5e-06] | CRASH |
| iNNvestigate | 1.5e-05 [1.4e-05, 1.7e-05] | 53 [52, 54] | 97 [97, 97] | 1e+02 [1e+02, 1e+02] |

### epsilon

| implementation | vgg16 | resnet50 | inception_v3 | mobilenet_v2 |
|---|---|---|---|---|
| TFRP v2 | 1.1 [1.1, 1.1] | 4.5e+16 [4e+16, 5.1e+16] | 5e+21 [4.2e+21, 5.8e+21] | 4.2e+35 [3.5e+35, 5.1e+35] |
| Zennit (canonized) | 1.1 [1.1, 1.1] | 7.7e+21 [6.1e+21, 8.6e+21] | 6.2e+23 [5.3e+23, 7.1e+23] | 3.5e+35 [2.9e+35, 4e+35] |
| Zennit (not canonized) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Captum | 43 [42, 44] | 96 [94, 98] | 1e+02 [1e+02, 1e+02] | CRASH |
| iNNvestigate | 1.1 [1.1, 1.1] | 2.4e+11 [2.2e+11, 2.8e+11] | 8.8e+14 [7.4e+14, 1e+15] | 5e+25 [4.4e+25, 5.7e+25] |

### alpha_beta

| implementation | vgg16 | resnet50 | inception_v3 | mobilenet_v2 |
|---|---|---|---|---|
| TFRP v2 | 13 [13, 14] | 0.065 [0.055, 0.077] | 0.0043 [0.0036, 0.0052] | 1e+02 [1e+02, 1e+02] |
| Zennit (canonized) | 13 [13, 14] | 0.066 [0.055, 0.076] | 0.0042 [0.0038, 0.0051] | 1e+02 [1e+02, 1e+02] |
| Zennit (not canonized) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Captum | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| iNNvestigate | 13 [13, 14] | 5.5e+07 [5.1e+07, 5.9e+07] | 6.9e+09 [6.2e+09, 7.5e+09] | 4.8e+05 [4.3e+05, 5.2e+05] |

### composite

| implementation | vgg16 | resnet50 | inception_v3 | mobilenet_v2 |
|---|---|---|---|---|
| TFRP v2 | 12 [11, 13] | 0.073 [0.069, 0.079] | 0.1 [0.1, 0.1] | 1e+02 [1e+02, 1e+02] |
| Zennit (canonized) | 12 [11, 13] | 0.072 [0.069, 0.078] | 0.1 [0.1, 0.1] | 1e+02 [1e+02, 1e+02] |
| Zennit (not canonized) | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Captum | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| iNNvestigate | 12 [11, 13] | 83 [82, 84] | 1e+02 [1e+02, 1e+02] | 1.8e+05 [1.6e+05, 2.2e+05] |
