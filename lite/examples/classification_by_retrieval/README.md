# CbR: Classification-by-Retrieval

Classification-by-retrieval provides an easy way to create a neural
network-based classifier without computationally expensive backpropagation
training.
Using this technology, you can create a lightweight mobile model with as little
as [one image per class](#model-accuracy-comparison-with-few-shot-learning), or
you can create an on-device model that can classify as many as tens of thousands
of classes.
For example, we created [mobile models that can recognize tens of thousands of
landmarks](https://tfhub.dev/google/collections/landmarks/1) with the
classification-by-retrieval technology.
We provide an [iOS app](ios/README.md), where you can choose images from the
photo library and label them to create a TfLite classifier within seconds
(if the number of images are small), and test the created model right away.
We also provide a C++ command line tool (in lib/tests) to build a classifier.

There are many use-cases for classification-by-retrieval, including:

*  Machine learning education (e.g., an educational hackathon event).
*  Easily prototyping, or demonstrating ML classification.
*  Custom product recognition (e.g., developing a product recognition app for a
   small/medium business without the need to gather extensive training data or
   write lots of code).


## Technical background

Classification and retrieval are two distinct methods of image recognition. A
typical object recognition approach is to build a neural network classifier and
train it with a large amount of training data (often thousands of images, or
more). On the contrary, the retrieval approach uses a pre-trained feature
extractor (e.g., an image embedding model) with feature matching based on a
nearest neighbor search algorithm. The retrieval approach is scalable and
flexible. For example, it can handle a large number of classes (say, > 1
million), and adding or removing classes does not require extra training. One
would need as little as a single training data per class, which makes it
effectively few-shot learning. A downside of the retrieval approach is that
it requires extra infrastructure, and is less intuitive to use than a
classification model.
Classification-by-retrieval (CbR) is a neural network model with image retrieval
layers baked into it. With the CbR technology, you can easily create a
TensorFlow classification model without any training.

![Conventional Approaches](docs/images/conventional-approaches.svg)
![Classification-by-Retrieval](docs/images/classification-by-retrieval.svg)

## How do the retrieval layers work?

A classification-by-retrieval model is an extension of an embedding model with
extra retrieval layers.
The retrieval layers are computed (not trained) from the training data, i.e.,
the index data.
The retrieval layers consists of two components:

*  Nearest neighbor matching component
*  Result aggregation component

The nearest neighbor matching component is essentially a fully connected layer
where its weights are the normalized embeddings of the index data.
Note that a dot-product of two normalized vectors (cos similarity) is linear
(with a negative coefficient) to the squared L2 distance.
Therefore, the output of the fully connected layer is effectively identical to
the nearest neighbor matching result.

The retrieval result is given for each training instance, not for each class.
Therefore, we add another *result aggregation component* on top of the nearest
neighbor matching layer.
The aggregation component consists of a selection layer for each class followed
by an aggregation (e.g., max) layer for each of them.
Finally, the results are concatenated to form a single output vector.

## Base Embedding Model

One may choose a base embedding model that best fits the domain.
There are many embedding models available, for example, in
[TensorFlow hub](http://tfhub.dev), for various domains.
The provided [iOS demo](ios/README.md) uses a
[MobileNet V3 trained with ImageNet](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1),
which is a generic and efficient on-device model.

## Model Accuracy: Comparison with Few Shot Learning

In some sense, CbR (indexing) can be considered as a few-shot learning approach
without training.
Although it is not apples to apples to compare CbR with an arbitrary pre-trained
base embedding model with a typical few-shot learning approach where the whole
model trained with given training data, there is a
[research](https://arxiv.org/pdf/1911.04623.pdf) that compares
nearest neighbor retrieval (which is equivalent to CbR) with few-shot learning
approaches. It shows that nearest neighbor retrieval can be comparable or even
better than many few-shot learning approaches.

## Responsible Model Building

We encourage you to build a model that is fair and responsible. To learn more
about building a responsible model:

*   https://www.tensorflow.org/responsible_ai
*   https://design.google/library/fair-not-default/
*   https://developers.google.com/machine-learning/crash-course/fairness/video-lecture
