# EfficientDet FQA

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'tanmingxing' reviewed: '2020-05-09' }
*-->

[TOC]

## 1. For Users

### 1.1 How can I convert the saved model to tflite?

Unfortunately, there is no way to do that with the current public tensorflow
release due to some issues in tf converter. We have some internal fixes, which
could potentially be available with the next TensorFlow release.

### 1.2 Why I see NaN during my training and how to debug it?

Because we use batch norm, which needs reasonable batch size. If your batch size
is too small, it may causes NaN. (We may add group norm to deal with this in
futurre)

If you see NaN, you can check the followings:

  - Is my batch size too small? It usually needs to be >=8.
  - Should I clip my gradient? How about h.clip_gradients_norm=5.0?
  - Should I use smaller jitter? How about jitter_min=0.8 and jitter_max=1.2?

If you want to debug it, you can use these tools:

```
tf.compat.v1.add_check_numerics_ops() # for Tensorflow 1.x
tf.debugging.disable_check_numerics() # for TensorFlow 2.x
```

### 1.3 Why my last class eval AP is always zero?

The current code assume class 0 is always reserved for background, so you if you K classes, then you should set num_classes=K+1.

See [#391](https://github.com/google/automl/issues/391) and [#398](https://github.com/google/automl/issues/398) for more discussion.

### 1.4 Why my input pipeline has assert failure?

This is most likely that your dataset has some images with many objects (more
than the 100 limit for COCO), you should set --hparams="max_instances_per_image=200" or larger.

See [#93](https://github.com/google/automl/issues/93) for more discussion.


## 2. For Developers

### 2.1 How can I format my code for PRs?

Please use [yapf](https://github.com/google/yapf) with option
--style='{based_on_style: yapf}'. You can also save the
following file to ~/.config/yapf/style:

    [style]
    based_on_style = yapf

If you want to check the format with lint, please run:

    !pylint --rcfile=../.pylintrc your_file.py

### 2.2 How can I run all tests?

    !export PYTHONPATH="`pwd`:$PYTHONPATH"
    !find . -name "*_test.py" | parallel python &> /tmp/test.log \
      && echo "All passed" || echo "Failed! Search keyword FAILED in /tmp/test.log"
