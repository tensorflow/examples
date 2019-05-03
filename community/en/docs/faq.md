# Frequently Asked Questions

This document provides answers to some of the frequently asked questions about
TensorFlow.

## Features and Compatibility

#### Can I run distributed training on multiple GPUs?

Yes! TensorFlow gained support for distributed computation back in
version 0.8, see the [distributed computation guide](./deploy/distributed.md).
TensorFlow supports multiple devices (CPUs and GPUs) on one or more computers.

#### Does TensorFlow work with Python 3?

Yes.

## TensorFlow graphs and eager execution

See the [graphs and sessions guide](https://www.tensorflow.org/guide/graphs) and the
[eager execution guide](https://www.tensorflow.org/guide/eager).

#### Do TensorFlow operations return immediately?

If you enable eager execution, operations like `c = tf.matmul(a, b)` are executed
immediately. See the [eager execution guide](https://www.tensorflow.org/guide/graphs) for using eager execution
to create more readable, intuitive TensorFlow code.

However, without eager execution enabled, an operation like `tf.matmul` above does
*not* execute immediately, but, instead, builds a fragment of a TensorFlow graph.

Why graphs?  TensorFlow graphs can help with distribution, optimization, and putting
models into production. In the suggested expression, `a`, `b`, and `c` are `tf.Tensor`
objects. A `tf.Tensor` object is a symbolic handle to the result of an
operation, but does not actually hold the values of the operation's
output. Instead, you can build up complicated expressions (such as
entire neural networks and their gradients) as a dataflow graph. You
then offload the computation of the entire dataflow graph (or a
subgraph of it) to a TensorFlow `tf.Session`, which is able to execute
the whole computation much more efficiently than executing the
operations one-by-one.

Note: In upcoming TensorFlow 2.0, all operations will be eagerly executed.

#### How are devices named?

The supported device names are `"/device:CPU:0"` (or `"/cpu:0"`) for the CPU
device, and `"/device:GPU:i"` (or `"/gpu:i"`) for the *i*th GPU device.

#### How do I place operations on a particular device?

To explicitly place a group of operations on a device, create them within a
`tf.device` context.  See the [using GPUs guide](https://www.tensorflow.org/guide/using_gpu) for details
about how TensorFlow assigns operations to devices.

You can also look at
[CIFAR-10 tutorial](https://www.tensorflow.org/tutorials/images/deep_cnn) for an example model that
uses multiple GPUs.

As of r1.12, we recommend trying
`tf.contrib.distribute.DistributionStrategy` as an easy way to
distribute computation with Keras and Estimator models. It is under
development.

## Running a TensorFlow computation

#### What is feeding and placeholders?

The recommended way of providing data to a model for training or
inference is via the `tf.data` API; see the
[Importing Data guide](https://www.tensorflow.org/guide/datasets).

However, in some older models you may find feeds and placeholders.
Feeding is a mechanism in the `tf.Session` API that allows you
to substitute different values for one or more tensors at run
time. The `feed_dict` argument to `tf.Session.run` is a dictionary
that maps `tf.Tensor` objects to numpy arrays (and some other types),
which will be used as the values of those tensors in the execution of
a step.

#### What is the difference between `Session.run()` and `Tensor.eval()`?

If `t` is a `tf.Tensor` object,
`tf.Tensor.eval` is shorthand for
`tf.Session.run`, where `sess` is the
current `tf.get_default_session`. The
two following snippets of code are equivalent:

```python
# Using `Session.run()`.
sess = tf.Session()
c = tf.constant(5.0)
print(sess.run(c))

# Using `Tensor.eval()`.
c = tf.constant(5.0)
with tf.Session():
  print(c.eval())
```

In the second example, the session acts as a
[context manager](https://docs.python.org/2.7/reference/compound_stmts.html#with),
which has the effect of installing it as the default session for the lifetime of
the `with` block. The context manager approach can lead to more concise code for
simple use cases (like unit tests); if your code deals with multiple graphs and
sessions, it may be more straightforward to make explicit calls to
`Session.run()`.

#### Do Sessions have a lifetime? What about intermediate tensors?

Sessions can own resources, such as
`tf.Variable`,
`tf.QueueBase`, and
`tf.ReaderBase`. These resources can sometimes use
a significant amount of memory, and can be released when the session is closed by calling
`tf.Session.close`.


#### Does the runtime parallelize parts of graph execution?

When you use graph execution, the TensorFlow runtime parallelizes
execution across many different dimensions:

* The individual ops have parallel implementations, using multiple cores in a
  CPU, or multiple threads in a GPU.
* Independent nodes in a TensorFlow graph can run in parallel on multiple
  devices, which makes it possible to speed up
  [CIFAR-10 training using multiple GPUs](https://www.tensorflow.org/tutorials/images/deep_cnn).
* The Session API allows multiple concurrent steps (i.e. calls to
  `tf.Session.run` in parallel). This
  enables the runtime to get higher throughput, if a single step does not use
  all of the resources in your computer.


#### Why does `Session.run()` hang when using a reader or a queue?

Note: Queues are discouraged in favor of `tf.data`, which provides a
simpler interface and improved performance.

The `tf.ReaderBase` and
`tf.QueueBase` classes provide special operations that
can *block* until input (or free space in a bounded queue) becomes
available. These operations allow you to build sophisticated
[input pipelines](../api_guides/python/reading_data.md), at the cost of making the
TensorFlow computation somewhat more complicated. See the how-to documentation
for
[using `QueueRunner` objects to drive queues and readers](../api_guides/python/reading_data.md#creating_threads_to_prefetch_using_queuerunner_objects)
for more information on how to use them.

## Variables

See the [variables guide](https://www.tensorflow.org/guide/variables).

#### Should I turn on `use_resource=True` when constructing variables?

Yes.  This uses safer memory behavior, and will be the default in
TensorFlow 2.0.

#### What is the lifetime of a variable?

A variable is created when you first run the
`tf.Variable.initializer`
operation for that variable in a session. It is destroyed when that
`tf.Session.close`.

In eager execution, variables are freed when their associated Python
objects are cleaned up.

#### How do variables behave when they are concurrently accessed?

Variables allow concurrent read and write operations. The value read from a
variable may change if it is concurrently updated. By default, concurrent
assignment operations to a variable are allowed to run with no mutual exclusion.
To acquire a lock when assigning to a variable, pass `use_locking=True` to
`tf.Variable.assign`.

## Tensor shapes

See also the
`tf.TensorShape`.

#### How can I determine the shape of a tensor in Python?

In TensorFlow, a tensor has both a static (inferred) shape and a dynamic (true)
shape. The static shape can be read using the
`tf.Tensor.get_shape`
method: this shape is inferred from the operations that were used to create the
tensor, and may be partially complete (the static-shape may contain `None`). If
the static shape is not fully defined, the dynamic shape of a `tf.Tensor`, `t`
can be determined using `tf.shape(t)`.

#### What is the difference between `x.set_shape()` and `x = tf.reshape(x)`?

The `tf.Tensor.set_shape` method updates
the static shape of a `Tensor` object, and it is typically used to provide
additional shape information when this cannot be inferred directly. It does not
change the dynamic shape of the tensor.

The `tf.reshape` operation creates
a new tensor with a different dynamic shape.

#### How do I build a graph that works with variable batch sizes?

It is often useful to build a graph that works with variable batch sizes
so that the same code can be used for (mini-)batch training, and
single-instance inference. The resulting graph can be
`tf.Graph.as_graph_def`
and
`tf.import_graph_def`.

When building a variable-size graph, the most important thing to remember is not
to encode the batch size as a Python constant, but instead to use a symbolic
`Tensor` to represent it. The following tips may be useful:

* Use [`batch_size = tf.shape(input)[0]`](https://www.tensorflow.org/api_docs/python/array_ops.md#shape)
  to extract the batch dimension from a `Tensor` called `input`, and store it in
  a `Tensor` called `batch_size`.

* Use `tf.reduce_mean` instead
  of `tf.reduce_sum(...) / batch_size`.


## TensorBoard


#### What is the simplest way to send data to TensorBoard?

Add summary ops to your TensorFlow graph, and write
these summaries to a log directory.  Then, start TensorBoard using

    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory


#### Every time I launch TensorBoard, I get a network security popup!

You can change TensorBoard to serve on localhost rather than '0.0.0.0' by
the flag --host=localhost. This should quiet any security warnings.

## Extending TensorFlow

#### My data is in a custom format. How do I read it using TensorFlow?

There are three main options for dealing with data in a custom format.

The easiest option is to write parsing code in Python that transforms the data
into a numpy array. Then, use `tf.data.Dataset.from_tensor_slices` to
create an input pipeline from the in-memory data.

If your data doesn't fit in memory, try doing the parsing in the Dataset
pipeline. Start with an appropriate file reader, like
`tf.data.TextLineDataset`. Then convert the dataset by mapping
`tf.data.Dataset.map` appropriate operations over it.
Prefer predefined TensorFlow operations such as `tf.decode_raw`,
`tf.decode_csv`, `tf.parse_example`, or `tf.image.decode_png`.

If your data is not easily parsable with the built-in TensorFlow operations,
consider converting it, offline, to a format that is easily parsable, such
as `tf.python_io.TFRecordWriter` format.


## Miscellaneous

#### What is TensorFlow's coding style convention?

The TensorFlow Python API adheres to the
[PEP8](https://www.python.org/dev/peps/pep-0008/) conventions.<sup>*</sup> In
particular, we use `CamelCase` names for classes, and `snake_case` names for
functions, methods, and properties. We also adhere to the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html).

The TensorFlow C++ code base adheres to the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

(<sup>*</sup> With one exception: we use 2-space indentation instead of 4-space
indentation.)
