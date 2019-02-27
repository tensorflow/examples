# TensorFlow Lite Speech Command Recognition

## Getting Started

### Prerequisites
- Python 3.5+
- Keras 2.1.6 or higher
- pandas and pandas-ml
- TensorFlow 1.5 or higher

## TensorFlow Speech Commands Dataset
The dataset has 65,000 one-second long utterances of 30 short words, by thousands of different people

### Classes
In this example, only 10 classes will be picked for the TensorFlow Lite speech commands application.
`stop down off right up go on yes left no`

### Downloading
Run the download script to load the dataset into the local filesystem.
```
python download.py
```

## Audio Processing
Data generation involves producing raw PCM wavform data containing a desired number of samples and at fixed sample rate and the following configuration is used

| Samples        | Sample Rate           | Clip Duration (ms)  |
| ------------- |:-------------:| -----:|
| 16000      | 16000 | 1000 |

## Model Architecture
It is mostly a time stacked VGG-esque model with 1D Convolutions for temporal data such as audio waveforms. A one-dimensional dilated convolutional layer serving as the context convolution (`context_conv`) is employed to extract a wider field of the data. This is followed by a dimensionality reduction in the `reduce_conv` layer which employes 1-D MaxPooling to reduce the number of parameters passed to the subsequent layer.

## Training
The model can be trained by running train.py

### Usage
```
python train.py [-h] [-sample_rate SAMPLE_RATE] \
              [-batch_size BATCH_SIZE] \
              [-output_representation OUTPUT_REPRESENTATION] \
              -data_dirs DATA_DIRS [DATA_DIRS ...]
```

### Example of training the model
```
python train.py -sample_rate 16000 -batch_size 64 -output_representation raw -data_dirs data/train
```

## Results
After the training the model for 100 epochs, the following confusion matrix was generated for assessing classification performance.

`[099]: val_categorical_accuracy: 0.94`
| Predicted     | _silence_     | down   | go  | left   | no  | off   | on  | right  | stop   | two   | up  | yes |
| ------------- |:-------------:| ------:| ---:| ------:| ---:| -----:| ----:| -------:| -------:| ------:| ----:|-----:|
| Actual |
_silence_ | 322 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0
down | 0 | 240 | 6 | 0 | 7 | 0 | 1 | 0 | 0 | 4 | 0 | 0
go | 5 | 3 | 223 | 0 | 2 | 1 | 1 | 0 | 0 | 15 | 2 | 0
left | 0 | 0 | 0 | 221 | 2 | 0 | 0 | 1 | 0 | 8 | 0 | 7
no | 0 | 0 | 4 | 0 | 246 | 0 | 2 | 0 | 0 | 14 | 0 | 0
off | 0 | 0 | 1 | 0 | 0 | 229 | 2 | 0 | 0 | 5 | 15 | 0
on | 3 | 0 | 0 | 0 | 0 | 7 | 227 | 0 | 0 | 14 | 1 | 0
right | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 222 | 0 | 21 | 3 | 1
stop | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 224 | 8 | 3 | 0
two | 6 | 4 | 5 | 7 | 6 | 2 | 6 | 3 | 0 | 1468 | 4 | 0
up | 1 | 0 | 0 | 0 | 0 | 10 | 1 | 0 | 0 | 11 | 230 | 0
yes | 2 | 1 | 0 | 2 | 4 | 0 | 1 | 0 | 0 | 6 | 0 | 240

## Built With

* [Keras](https://keras.io/) - Deep Learning Framework
* [TensorFlow](http://tensorflow.org/) - Machine Learning Library

## References
For more details on the dataset, visit [here](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
