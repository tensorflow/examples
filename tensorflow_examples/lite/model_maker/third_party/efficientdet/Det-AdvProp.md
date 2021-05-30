# Det-AdvProp
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google/automl/blob/master/efficientdet/det_advprop_tutorial.ipynb)

[1] Xiangning Chen, Cihang Xie, Mingxing Tan, Li Zhang, Cho-Jui Hsieh, Boqing
Gong. CVPR 2021. Arxiv link: https://arxiv.org/abs/2103.13886

Det-AdvProp is a data augmentation technique specifically designed for the
fine-tuning process of object detectors. It can consistently and substantially
outperform the vanilla training and AutoAugment under various settings. The
obtained detector is not only more accurate on clean images, but also more
robust to image distortions and domain shift.

<p align="center">
<img src="./g3doc/Det-AdvProp.png" width="100%" />
</p>

## 1. Accurate on Clean Images

The following table includes a list of models trained with Det-AdvProp +
AutoAugment (AA):

Model                                                                                                                                                                                                                          | AP<sup>test</sup> | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> | AP<sup>val</sup> |     | #params | #FLOPs
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | --------------- | --------------- | -------------- | -------------- | -------------- | ---------------- | --- | ------- | :----:
EfficientDet-D0 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d0.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d0.txt)) | 35.3              | 54.1            | 37.8            | 12.7           | 39.9           | 53.2           | 35.1             |     | 3.9M    | 2.54B
EfficientDet-D1 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d1.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d1.txt)) | 40.9              | 60.0            | 44.1            | 19.1           | 45.6           | 57.2           | 40.8             |     | 6.6M    | 6.10B
EfficientDet-D2 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d2.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d2.txt)) | 44.3              | 63.5            | 47.9            | 23.5           | 48.5           | 59.9           | 44.3             |     | 8.1M    | 11.0B
EfficientDet-D3 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d3.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d3.txt)) | 48.0              | 67.1            | 52.2            | 28.1           | 51.8           | 62.8           | 47.7             |     | 12.0M   | 24.9B
EfficientDet-D4 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d4.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d4.txt)) | 50.4              | 69.5            | 54.9            | 30.9           | 54.3           | 64.4           | 50.4             |     | 20.7M   | 55.2B
EfficientDet-D5 + Det-AdvProp + AA ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/efficientdet-d5.tar.gz), [test-dev](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/advprop/d5.txt)) | 52.5              | 71.8            | 57.2            | 34.6           | 55.9           | 65.2           | 52.2             |     | 33.7M   | 130B

<sup>Unlike the vanilla EfficientDet that scales the image with mean and std,
here we scale the input to the range of [-1, 1] to make it easier for performing
adversarial attack. Please see [this Colab](https://github.com/google/automl/blob/master/efficientdet/det_advprop_tutorial.ipynb) for reproducing the
results.</sup>

## 2. Robust Against Common Corruptions

We test the detectors' robustness against common image corruptions (e.g.,
Gaussian Noise, Snow, etc.) based on the COCO-C dataset in
[this paper](https://arxiv.org/abs/1907.07484). The table below shows the
comparison between vanilla training and Det-AdvProp + AutoAugment (AA):

Model                  | mAP
---------------------- | ---------------
EfficientDet-D0        | 21.4
**+ Det-AdvProp + AA** | **22.7 (+1.3)**
EfficientDet-D1        | 24.4
**+ Det-AdvProp + AA** | **26.7 (+2.3)**
EfficientDet-D2        | 26.7
**+ Det-AdvProp + AA** | **28.9 (+2.2)**
EfficientDet-D3        | 28.8
**+ Det-AdvProp + AA** | **32.0 (+3.2)**
EfficientDet-D4        | 30.1
**+ Det-AdvProp + AA** | **33.9 (+3.8)**
EfficientDet-D5        | 31.4
**+ Det-AdvProp + AA** | **35.0 (+3.6)**

## 3. Robust Against Domain Shift

PASCAL VOC 2012 only contains 20 classes, which are much smaller than the 80
labeled classes in COCO. The underlying distributions of the two datasets are
also different in the image content or the bounding box sizes and locations. We
use the trained detectors to run inference directly on the VOC dataset to test
their transferibility. We maintain the COCO evaluation metrics in this
experiment:

Model                  | mAP             | AP<sub>50</sub> | AP<sub>75</sub>
---------------------- | --------------- | --------------- | ---------------
EfficientDet-D0        | 55.6            | 77.6            | 61.4
**+ Det-AdvProp + AA** | **56.2 (+0.6)** | **78.3 (+0.7)** | **62.3 (+0.9)**
EfficientDet-D1        | 60.8            | 82.0            | 66.7
**+ Det-AdvProp + AA** | **61.3 (+0.5)** | **82.5 (+0.5)** | **67.6 (+0.9)**
EfficientDet-D2        | 63.3            | 83.6            | 69.3
**+ Det-AdvProp + AA** | **63.6 (+0.3)** | **84.0 (+0.4)** | **70.0 (+0.7)**
EfficientDet-D3        | 65.7            | 85.3            | 71.8
**+ Det-AdvProp + AA** | **66.4 (+0.7)** | **85.9 (+0.6)** | **72.8 (+1.0)**
EfficientDet-D4        | 67.0            | 86.0            | 73.0
**+ Det-AdvProp + AA** | **67.8 (+0.8)** | **87.0 (+1.0)** | **74.3 (+1.3)**
EfficientDet-D5        | 67.4            | 86.9            | 73.8
**+ Det-AdvProp + AA** | **68.7 (+1.3)** | **88.0 (+1.1)** | **75.4 (+1.6)**
