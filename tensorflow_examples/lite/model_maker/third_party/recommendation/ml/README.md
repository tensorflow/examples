# On-device recommendation model with Tensorflow Lite

## Overview
This code base provides a toolkit to train and serve on-device recommendation
model. This approach personalizes recommendations by leveraging on-device data,
and protects user privacy without having user data leave device.

This code demonstrates the approach with public
[movielens](https://grouplens.org/datasets/movielens/) dataset, but you could
adapt the data processing script for your dataset and train your own
recommendation model.

## Model
We leverage a dual-encoder model architecture, with context-encoder to encode
sequential user history and label-encoder to encode predicted recommendation
candidate. Similarity between context and label encodings is used to represent
the likeliness predicted candidate meets user's needs.

Three different sequential user history encoding techniques are provided with
this code base:

* Bag of words encoder (BOW): averaging user activities' embeddings without
considering context order.
* Convolutional neural-network encoder (CNN): applying multiple layers of
convolutional neural-network to generate context encoding.
* Recurrent neural-network encoder (RNN): applying recurrent neural network
(LSTM in this example) to understand context sequence.

## Prerequisites
Please download the source code, setup your virtualenv and requirements:

```
cd lite/examples/recommendation/ml
pip install -r requirements.txt
```

Note that this code base requires python3.5 or later versions.

## Generate training examples
Please run:

```
python -m data.example_generation_movielens \
  --data_dir=/tmp/recommendation/raw \
  --output_dir=/tmp/recommendation/examples \
  --max_context_length=10
```

## Train model
* Setup json file as sample_config.json, to config model architecture.
* Execute recommendation_model_launcher_keras.py with run_model as
"train_and_eval". Encoder type and training parameters could be config through
flags.

```
python -m model.recommendation_model_launcher_keras \
  --run_mode "train_and_eval" \
  --encoder_type "bow" \
  --training_data_filepattern "/tmp/recommendation/examples/train_movielens_1m.tfrecord" \
  --testing_data_filepattern "/tmp/recommendation/examples/test_movielens_1m.tfrecord" \
  --model_dir "/tmp/recommendation/reco_model" \
  --params_path "model/sample_config.json" \
  --batch_size 32 \
  --learning_rate 0.01 \
  --steps_per_epoch 1000 \
  --num_epochs 1000 \
  --num_eval_steps 1000 \
  --gradient_clip_norm 1.0 \
  --max_history_length 10
```

## Export model

```
python -m model.recommendation_model_launcher_keras \
  --run_mode "export" \
  --model_dir "/tmp/recommendation/reco_model" \
  --params_path "model/sample_config.json"\
  --checkpoint_path "/tmp/recommendation/reco_model/ckpt-1000000" \
  --num_predictions 100 \
  --max_history_length 10
```

## Further reading

If you want to read more about the technology related to on-device
recommendation, please check out our
[blogpost](https://blog.tensorflow.org/2020/09/introduction-to-tflite-on-device-recommendation.html).
