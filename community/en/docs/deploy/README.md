# Deploy

When it's time to show your trained models to the world you'll need to choose
one or more deployment options. Fortunately, TensorFlow offers the tools and
frameworks you'll need to deploy your models for a wide range of use cases.

## TensorFlow Serving

Machine Learning (ML) serving systems need to support model versioning (for
model updates with a rollback option) and multiple models (for A/B testing),
while ensuring that concurrent models achieve high throughput on hardware
accelerators (GPUs and TPUs) with low latency.
[TensorFlow Serving](https://www.tensorflow.org/serving) is currently handling
tens of millions of inferences per second for 1100+ of Google projects,
including Google’s Cloud ML Prediction.

## TensorFlow Extended (TFX)

When you’re ready to go beyond training a single model, or ready to put your
amazing model to work and move it to production,
[TFX](https://www.tensorflow.org/tfx) is there to help you build a complete ML
pipeline.

TFX pipelines can be deployed to on-premises infrastructure, or a hybrid of
on-prem and cloud, or a pure cloud deployment on the
[Google Cloud Platform](https://cloud.google.com/products/ai/). Your models can
be deployed to be served online, or in a mobile app, or in a JavaScript app, or
all of the above.

With a TFX pipeline you can continuously retrain and update your models, and
manage your model versioning and life cycle. TFX gives you the tools to validate
and transform new data, monitor model performance, perform A/B testing, serve
your trained models, and more. With TFX, your models are ready for production.

## TensorFlow.js

With [TensorFlow.js](https://js.tensorflow.org/) you can train and deploy your
TensorFlow models in the browser and in Node.js. Use flexible and intuitive
APIs to build and train models from scratch using the low-level JavaScript
linear algebra library or the high-level layers API. Use TensorFlow.js model
converters to run pre-existing TensorFlow models right in the browser or under
Node.js. Retrain pre-existing ML models using sensor data connected to the
browser, or other client-side data.

## TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite) is the official solution for
running machine learning models on mobile and embedded devices. It enables on
device machine learning inference with low latency and a small binary size on
Android, iOS, and other operating systems. Build a new model or retrain an
existing one, such as using transfer learning. Convert a TensorFlow model into a
compressed flat buffer with the TensorFlow Lite Converter. Take the compressed
`.tflite` file and load it into a mobile or embedded device.

## Other Deployment Scenarios

* [Distributed TensorFlow](distributed.md) - Learn how to create a cluster of
  TensorFlow servers
* [TensorFlow on Hadoop](hadoop.md) - Learn how to run TensorFlow on Hadoop
  using HDFS
* [TensorFlow on S3](s3.md) - Learn how Tensorflow supports reading and writing
  data to S3
