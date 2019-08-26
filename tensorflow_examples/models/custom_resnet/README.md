# Custom ResNet with TinyImageNet
### TinyImageNet Image Classification
---
#### Dependencies
```
opencv_python==4.0.0.21
h5py==2.9.0
pandas==0.23.4
numpy==1.16.4
Keras==2.2.4
tensorflow==2.0.0b1
imutils==0.5.2
scikit_learn==0.21.2
absl-py==0.7.0
```
#### Downloads
- To download all the dependencies, simply execute 
```
pip install -r requirements.txt
```
- To download the TinyimageNet dataset, simply execute the `load_data.py` file. This also exracts the data to make it easier to work with.
```
python load_data.py
```
#### Training
- The `model.py` file contains the code to train the TinyImageNet with progressive resizing to enable the model to learn size independent semantic features. It automatically stores the weights after the specified/default number of epochs have completed. Note that the weights will be stored in the `./checkpoints/` present at the same directory level as `main.py`.
```
python main.py
```
- The model may need a few more epochs to converge very well. In such cases, train it it for 10 epochs on the smaller sizes and 20 epochs on the 64x64 sized images.
---
#### Reference Papers
1. **Building blocks of interpretability**: [[Link](https://distill.pub/2018/building-blocks/)]
2. **Deep Residual Learning for image classification**: [[Link](https://arxiv.org/abs/1512.03385)] (Resnet Paper)
3. **Bag of tricks for image classification**: [[Link](https://arxiv.org/abs/1812.01187)] (Tweaks and tricks to Resnet for increased performance paper)
2. **Imbalanced Deep Learning by Minority Class
Incremental Rectification**: [[Link](https://arxiv.org/pdf/1804.10851.pdf)] (Selectively Sampling Data paper)
2. **Improved Regularization of Convolutional Neural Networks with Cutout**: [[Link](https://arxiv.org/pdf/1708.04552.pdf)] (Cutout/Occlusion Augmentation paper)
3. **Survey of resampling techniques for improving
classification performance in unbalanced datasets** [[Link](https://arxiv.org/pdf/1608.06048v1.pdf)]
