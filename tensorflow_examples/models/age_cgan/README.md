# Age-cGAN
### Face aging - FaceApp
---
#### Dependencies
```
tensorflow==2.0.0b1
absl_py==0.7.0
numpy==1.16.4
matplotlib==2.2.3
scipy==1.1.0
```
#### Downloads
- To download all the dependencies, simply execute 
```
pip install -r requirements.txt
```
- To download the Wiki Cropped Faces dataset, simply execute the `data_download.py` file
```
python data_download.py
```
#### Training
There are 3 stages of training:
1. Conditional GAN Training (Generator and Discriminator)
2. Initial Latent Vector Approximation (Encoder)
3. Latent vector optimization (Encoder and Generator)
- The `model.py` file contains the code to run all 3 stages of training. It automatically stores the weights after the specified/default number of epochs have completed. Note that the weights will be stored at the same directory level as `model.py`.
```
python model.py
```
- An important distinction to note is that there are 3 bool variables, namely `TRAIN_GAN`, `TRAIN_ENCODER` and `TRAIN_ENC_GAN` wich respectively represent the 3 stages of training. They have been set to `True` by default. If the requirement arises such that the training be completed individually, set the respective variables to `False` and ensure the requirements are met before proceeding.
---
#### Architecture
1. Encoder: Learns the reverse mapping of input face images and the latent vector
2. FaceNet: Face recognition network
3. Generator: Takes a hidden representation of the face and a condition vector to generate a face
4. Discriminator: Distinguishes between real and fake images
#
- **Encoder**: 4 CNN blocks with BatchNorm and Activation layers followed by 2 FC layers
- **Generator**: CNNs, Upsampling layers and dense layers
- **Face Recognition**: Pretrained `Inception-ResNet-V2` or `ResNet-50`
- **Discriminator**: CNN blocks with BatchNorm and Activation layers
#
#### Reference Papers
1. Face Aging with Conditional Generative Adversarial Networks [[Arxiv Link](https://arxiv.org/abs/1702.01983)]
2. Improved Techniques for Training GANs [[Arxiv Link](https://arxiv.org/pdf/1606.03498.pdf)]
