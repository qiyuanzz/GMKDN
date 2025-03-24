# Multi-modal Knowledge Decomposition based Online Distillation for Biomarker Prediction in Breast Cancer Histopathology

This repository is an official PyTorch implementation of the paper "Online Teaching: Distilling Decomposed Multimodal Knowledge for Breast Cancer Biomarker Prediction".


## framework
 <p align="center">
  <img align="center" src="fm.png" width="800px"/>
 </p>

## Environment
Pytorch 2.0.1
Python 3.9


 ### Run code

```
# train with 5-fold cross validation and test
python train_kd.py --weighted_sample --early_stopping
python train_kd.py --stage 'test' --weighted_sample --early_stopping
```

