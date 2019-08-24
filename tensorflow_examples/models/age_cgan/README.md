# Age-cGAN
### Face aging - FaceApp
---
#### Download
-
---
#### Reference Paper
1. Face Aging with Conditional Generative Adversarial Networks [Arxiv Link](https://arxiv.org/abs/1702.01983)
2. Improved Techniques for Training GANs [Arxiv Link](https://arxiv.org/pdf/1606.03498.pdf)
#### Architecture
1. Encoder: Learns the reverse mapping of input face images and the latent vector
2. FaceNet: Face recognition network
3. Generator: Takes a hidden representation of the face and a condition vector to generate a face
4. Discriminator: Distinguishes between real and fake images
#
- **Encoder**: 4 CNN blocks with BatchNorm and Activation layers followed by 2 FC layers
- **Generator**: CNNs, Upsampling layers and dense layers
- **Face Recognition**: `Pretrained Inception-ResNet-V2` or `ResNet-50`
- **Discriminator**: CNN blocks with BatchNorm and Activation layers
#
#### Stages of training:
1. cGAN Training (Generator and Discriminator)
2. Initial Latent Vector Approximation (Encoder)
3. Latent vector optimization (Encoder and Generator)

<img src="../assets/face_aging.png" width="960px" height="246px"/>

