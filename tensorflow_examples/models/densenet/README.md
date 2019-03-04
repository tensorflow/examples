# Densely Connected Convolutional Networks.

Image Classification using the Densenet model as described in [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## Content
*  `densenet.py`: Model definition.
*  `train.py`: Training the model with custom loop and keras fit.
*  `densenet_test.py`: Unit tests and benchmarks.
*  `distributed_train.py`: Training the model using tf.distribute.Strategy with custom loops.
*  `densenet_distributed_test.py`: Unit tests and benchmarks using multiple GPUs.
*  `utils.py`: Configuration and dataset preprocessing.

## Benchmarks

Each of these benchmarks were performed on the CIFAR10 dataset with image augmentation (C10+ as described in the [paper](https://arxiv.org/abs/1608.06993)) with a batch size of 64 per replica. All of the benchmarks were performed on the
NVIDIA V100 GPUs.

Model                 | Benchmark name                                          | accuracy   | seconds/epoch |
--------------------- | ------------------------------------------------------  | ---------  | ------------- |
Densenet (L=40, k=12) | benchmark_with_keras_fit_300_epochs                     |     94.67% |            51 |
Densenet (L=40, k=12) | benchmark_with_function_custom_loops_300_epochs         |     94.75% |            42 |
Densenet (L=40, k=12) | benchmark_with_function_custom_loops_300_epochs_2_gpus  |     94.69% |            23 |


To build (and run benchmarks) from source:

```python
bazel run -c opt --config=cuda :densenet_test -- --benchmarks=.

bazel run -c opt --config=cuda :densenet_distributed_test -- --benchmarks=.
```

## Reference

```
@article{DBLP:journals/corr/HuangLW16a,
  author    = {Gao Huang and
               Zhuang Liu and
               Kilian Q. Weinberger},
  title     = {Densely Connected Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1608.06993},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.06993},
  archivePrefix = {arXiv},
  eprint    = {1608.06993},
  timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/HuangLW16a},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
