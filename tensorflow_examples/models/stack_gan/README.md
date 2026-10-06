# StackGAN
### Text to Photo-Realistic Image Synthesis
---
#### Dependencies
```
tensorflow==2.0.0b1
numpy==1.16.4
absl_py==0.7.0
matplotlib==2.2.3
pandas==0.23.4
Pillow==6.1.0
```
#### Downloads
- To download all the dependencies, simply execute 
```
pip install -r requirements.txt
```
- To download the CUB 200 dataset, simply execute the `data_download.py` file
```
python data_download.py
```
- Download the Char-RNN-CNN embeddings from this link: [**download link**](https://drive.google.com/file/d/0B3y_msrWZaXLT1BZdVdycDY5TEE) and unzip it in place. 
```
unzip birds.zip
```
#### Training
- The `model.py` file contains the bare minimum code to run the stage 1 and stage 2 architecture. It automatically stores the weights after the specified/default number of epochs have completed. Note that the weights will be stored at the same directory level as `model.py`.
```
python model.py
```
#### Architecture
- Stage 1
	- Text Encoder Network
		- Text description to a 1024 dimensional text embedding
		- Learning Deep Representations of Fine-Grained Visual Descriptions [Arxiv Link](https://arxiv.org/abs/1605.05395)
	- Conditioning Augmentation Network
		- Adds randomness to the network
		- Produces more image-text pairs
	- Generator Network
	- Discriminator Network
	- Embedding Compressor Network
	- Outputs a 64x64 image
#
- Stage 2
	- Text Encoder Network
	- Conditioning Augmentation Network
	- Generator Network
	- Discriminator Network
	- Embedding Compressor Network
	- Outputs a 256x256 image
---
#### Reference Papers
1. **StackGAN: Text to photo-realistic image synthesis** [[Arxiv Link](https://arxiv.org/pdf/1612.03242.pdf)]
2. **Improved Techniques for Training GANs** [[Arxiv Link](https://arxiv.org/pdf/1606.03498.pdf)]
3. **Generative Adversarial Text to Image Synthesis** [[Arxiv Link](https://arxiv.org/pdf/1605.05396.pdf)]
4. **Learning Deep Representations of Fine-Grained Visual Descriptions** [[Arxiv Link](https://arxiv.org/abs/1605.05395)]
---
