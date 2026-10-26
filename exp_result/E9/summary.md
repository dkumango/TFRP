# E9 epsilon sweep: median RCE % (finite) / overflow % / RCE>100 %


## inception_v3

```
               RCE_median_finite                    overflow_pct                    exploded_pct                   
implementation           TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)
epsilon                                                                                                            
1.000000e-09                 NaN                NaN          100                100            0                  0
1.000000e-06                 NaN                NaN          100                100            0                  0
1.000000e-04                 NaN                NaN          100                100            0                  0
1.000000e-02               5e+21           5.91e+23            0                  0          100                100
1.000000e-01             6.1e+05            3.4e+06            0                  0          100                100
1.000000e+00                 100                100            0                  0           17                  6
1.000000e+01                 100                100            0                  0            8                  8
1.000000e+02                 100                100            0                  0            0                  0
```

## mobilenet_v2

```
               RCE_median_finite                    overflow_pct                    exploded_pct                   
implementation           TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)
epsilon                                                                                                            
1.000000e-09                 NaN                NaN          100                100            0                  0
1.000000e-06                 NaN                NaN          100                100            0                  0
1.000000e-04                 NaN                NaN          100                100            0                  0
1.000000e-02            2.92e+35           2.77e+35            6                  2           94                 98
1.000000e-01            1.49e+19           1.34e+18            0                  0          100                100
1.000000e+00                 100                100            0                  0           52                 52
1.000000e+01                 100                100            0                  0           16                  2
1.000000e+02                 100                100            0                  0            0                  0
```

## resnet50

```
               RCE_median_finite                    overflow_pct                    exploded_pct                   
implementation           TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)
epsilon                                                                                                            
1.000000e-09                 NaN                NaN          100                100            0                  0
1.000000e-06                 NaN                NaN          100                100            0                  0
1.000000e-04                 NaN                NaN          100                100            0                  0
1.000000e-02             4.3e+16           5.15e+21            0                  0          100                100
1.000000e-01                79.1           1.26e+04            0                  0           41                100
1.000000e+00                99.4               98.5            0                  0            1                  0
1.000000e+01                 100                100            0                  0            1                  4
1.000000e+02                 100                100            0                  0            3                  7
```

## vgg16

```
               RCE_median_finite                    overflow_pct                    exploded_pct                   
implementation           TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)      TFRP v2 Zennit (canonized)
epsilon                                                                                                            
1.000000e-09             0.00219            0.00208            0                  0            0                  0
1.000000e-06             0.00194            0.00163            0                  0            0                  0
1.000000e-04               0.014             0.0135            0                  0            0                  0
1.000000e-02                1.07               1.08            0                  0            0                  0
1.000000e-01                 8.1                8.1            0                  0            0                  0
1.000000e+00                45.5               45.5            0                  0            0                  0
1.000000e+01                95.9               95.9            0                  0            0                  0
1.000000e+02                 100                100            0                  0            0                  0
```
