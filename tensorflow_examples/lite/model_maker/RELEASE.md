# Release 0.3

## 0.3.3

*   Minor fix: update tensorflow==2.6 and fix the failure of `evaluate_tflite`
    for the object detection task.

## 0.3.2

*   Minor fix: Move librosa to an optional library for audio task.

## 0.3.1

*   Polish document showing in
    https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker.
*   Add some 0.2.x APIs for backward compatibility.

## 0.3.0

*   Export TFLite Model Maker public APIs, and adapt colabs as demo code.
*   Add three new tasks: Object Detection, Audio Classification and
    Recommendation.

# Release 0.2

## 0.2.5

*   Refactor the code to be more flexible and extentable without changing API.
*   Add executable test script for pip package.
*   Update tensorflow-hub version in requirements.txt to fix incompatible error
    when installing pip package.

## 0.2.4

*   Update stable tensorflow version in requirements.txt.
*   Support exporting to TFJS models.
*   Refine Code: refactor the `gen_dataset` internal API in `DataLoader` and
    remove redundant methods.
*   Split requirements to stable version and nightly version.

## 0.2.3

*   Fix a bug for missing `default_batch_size`.
*   Add a demo of customized task.
*   Refine Code: Refactor `model_spec` to split different model specifications
    into image/text and add `ClassificationDataLoader`.

## 0.2.2

*   Update tf-nightly version in requirements.txt.
*   Refine Code: Change pre-defined model specification to function and extract
    Keras callback logic into a reusable function.

## 0.2.1

*   Set the required package version in requirements.txt.
*   Refine Code: Move `load_from_tfds` into `ImageDataLoader`, refactor the
    `DataLoader` code, unify the predict method and provide a default
    implementation for TFLite expo.

## 0.2.0

*   Setup virtualenv and test pip package.

# Release 0.1

Initial version for
[TFLite Model Maker](https://www.tensorflow.org/lite/guide/model_maker): a model
customization library for on-device applications.

Support tasks:

*   [image classification](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)
*   [text classification](https://www.tensorflow.org/lite/tutorials/model_maker_text_classification)
*   [question answer](https://www.tensorflow.org/lite/tutorials/model_maker_question_answer)

## 0.1.2

*   Add `metadata_writer` for nlp tasks and set `with_metadata=True` by default
    when exporting to TFLite.
*   Update `model_info` in metadata for image classifier.

## 0.1.1

*   Update README.
*   Add `mobilebert_qa_squad` model_spec, a mobilebert model pre-trained with
    Squad1.1 dataset.

## 0.1.0

*   Initial version.
