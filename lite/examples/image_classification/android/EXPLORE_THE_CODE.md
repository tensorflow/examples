# TensorFlow Lite Android image classification example

This document walks through the code of a simple Android mobile application that
demonstrates
[image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
using the device camera.

## Explore the code

We're now going to walk through the most important parts of the sample code.

### Get camera input

This mobile application gets the camera input using the functions defined in the
file
[`CameraActivity.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/CameraActivity.java).
This file depends on
[`AndroidManifest.xml`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/AndroidManifest.xml)
to set the camera orientation.

`CameraActivity` also contains code to capture user preferences from the UI and
make them available to other classes via convenience methods.

```java
model = Model.valueOf(modelSpinner.getSelectedItem().toString().toUpperCase());
device = Device.valueOf(deviceSpinner.getSelectedItem().toString());
numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
```

### Classifier

The file
[`Classifier.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/Classifier.java)
contains most of the complex logic for processing the camera input and running
inference.

Two subclasses of the file exist, in
[`ClassifierFloatMobileNet.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/ClassifierFloatMobileNet.java)
and
[`ClassifierQuantizedMobileNet.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/tflite/ClassifierQuantizedMobileNet.java),
to demonstrate the use of both floating point and
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
models. After the introduction of the
[TensorFlow Lite Android Support Library](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/java)
these subclasses mainly contain settings rather than processing logic.

The `Classifier` class implements a static method, `create`, which is used to
instantiate the appropriate subclass based on the supplied model type (quantized
vs floating point).

#### Load model and create interpreter

To perform inference, we need to load a model file and instantiate an
`Interpreter`. This happens in the constructor of the `Classifier` class, along
with loading the list of class labels. Information about the device type and
number of threads is used to configure the `Interpreter` via the
`Interpreter.Options` instance passed into its constructor. Note that if a GPU,
DSP (Digital Signal Processor) or NPU (Neural Processing Unit)
is available, a
[`Delegate`](https://www.tensorflow.org/lite/performance/delegates) can be
used to take full advantage of these hardware.

Please note that there are performance edge cases and developers are adviced to
test with a representative set of devices prior to production.

```java
protected Classifier(Activity activity, Device device, int numThreads) throws
    IOException {
  tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
  switch (device) {
    case NNAPI:
      nnApiDelegate = new NnApiDelegate();
      tfliteOptions.addDelegate(nnApiDelegate);
      break;
    case GPU:
      gpuDelegate = new GpuDelegate();
      tfliteOptions.addDelegate(gpuDelegate);
      break;
    case CPU:
      break;
  }
  tfliteOptions.setNumThreads(numThreads);
  tflite = new Interpreter(tfliteModel, tfliteOptions);
  labels = FileUtil.loadLabels(activity, getLabelPath());
...
```

For Android devices, we recommend pre-loading and memory mapping the model file
to offer faster load times and reduce the dirty pages in memory. The method
`FileUtil.loadMappedFile` does this, returning a `MappedByteBuffer` containing
the model.

The `MappedByteBuffer` is passed into the `Interpreter` constructor, along with
an `Interpreter.Options` object. This object can be used to configure the
interpreter, for example by setting the number of threads (`.setNumThreads(1)`)
or enabling [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)
(`.addDelegate(nnApiDelegate)`).

#### Pre-process bitmap image

Next in the `Classifier` constructor, we take the input camera bitmap image,
convert it to a `TensorImage` format for efficient processing and pre-process
it. The steps are shown in the private 'loadImage' method:

```java
/** Loads input image, and applys preprocessing. */
private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
  // Loads bitmap into a TensorImage.
  image.load(bitmap);

  // Creates processor for the TensorImage.
  int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
  int numRoration = sensorOrientation / 90;
  ImageProcessor imageProcessor =
      new ImageProcessor.Builder()
          .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
          .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.BILINEAR))
          .add(new Rot90Op(numRoration))
          .add(getPreprocessNormalizeOp())
          .build();
  return imageProcessor.process(inputImageBuffer);
}
```

The pre-processing is largely the same for quantized and float models with one
exception: Normalization.

In `ClassifierFloatMobileNet`, the normalization parameters are defined as:

```java
private static final float IMAGE_MEAN = 127.5f;
private static final float IMAGE_STD = 127.5f;
```

In `ClassifierQuantizedMobileNet`, normalization is not required. Thus the
nomalization parameters are defined as:

```java
private static final float IMAGE_MEAN = 0.0f;
private static final float IMAGE_STD = 1.0f;
```

#### Allocate output object

Initiate the output `TensorBuffer` for the output of the model.

```java
/** Output probability TensorBuffer. */
private final TensorBuffer outputProbabilityBuffer;

//...
// Get the array size for the output buffer from the TensorFlow Lite model file
int probabilityTensorIndex = 0;
int[] probabilityShape =
    tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, 1001}
DataType probabilityDataType =
    tflite.getOutputTensor(probabilityTensorIndex).dataType();

// Creates the output tensor and its processor.
outputProbabilityBuffer =
    TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

// Creates the post processor for the output probability.
probabilityProcessor =
    new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
```

For quantized models, we need to de-quantize the prediction with the NormalizeOp
(as they are all essentially linear transformation). For float model,
de-quantize is not required. But to uniform the API, de-quantize is added to
float model too. Mean and std are set to 0.0f and 1.0f, respectively. To be more
specific,

In `ClassifierQuantizedMobileNet`, the normalized parameters are defined as:

```java
private static final float PROBABILITY_MEAN = 0.0f;
private static final float PROBABILITY_STD = 255.0f;
```

In `ClassifierFloatMobileNet`, the normalized parameters are defined as:

```java
private static final float PROBABILITY_MEAN = 0.0f;
private static final float PROBABILITY_STD = 1.0f;
```

#### Run inference

Inference is performed using the following in `Classifier` class:

```java
tflite.run(inputImageBuffer.getBuffer(),
    outputProbabilityBuffer.getBuffer().rewind());
```

#### Recognize image

Rather than call `run` directly, the method `recognizeImage` is used. It accepts
a bitmap and sensor orientation, runs inference, and returns a sorted `List` of
`Recognition` instances, each corresponding to a label. The method will return a
number of results bounded by `MAX_RESULTS`, which is 3 by default.

`Recognition` is a simple class that contains information about a specific
recognition result, including its `title` and `confidence`. Using the
post-processing normalization method specified, the confidence is converted to
between 0 and 1 of a given class being represented by the image.

```java
/** Gets the label to probability map. */
Map<String, Float> labeledProbability =
    new TensorLabel(labels,
        probabilityProcessor.process(outputProbabilityBuffer))
        .getMapWithFloatValue();
```

A `PriorityQueue` is used for sorting.

```java
/** Gets the top-k results. */
private static List<Recognition> getTopKProbability(
    Map<String, Float> labelProb) {
  // Find the best classifications.
  PriorityQueue<Recognition> pq =
      new PriorityQueue<>(
          MAX_RESULTS,
          new Comparator<Recognition>() {
            @Override
            public int compare(Recognition lhs, Recognition rhs) {
              // Intentionally reversed to put high confidence at the head of
              // the queue.
              return Float.compare(rhs.getConfidence(), lhs.getConfidence());
            }
          });

  for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
    pq.add(new Recognition("" + entry.getKey(), entry.getKey(),
               entry.getValue(), null));
  }

  final ArrayList<Recognition> recognitions = new ArrayList<>();
  int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
  for (int i = 0; i < recognitionsSize; ++i) {
    recognitions.add(pq.poll());
  }
  return recognitions;
}
```

### Display results

The classifier is invoked and inference results are displayed by the
`processImage()` function in
[`ClassifierActivity.java`](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android/app/src/main/java/org/tensorflow/lite/examples/classification/ClassifierActivity.java).

`ClassifierActivity` is a subclass of `CameraActivity` that contains method
implementations that render the camera image, run classification, and display
the results. The method `processImage()` runs classification on a background
thread as fast as possible, rendering information on the UI thread to avoid
blocking inference and creating latency.

```java
@Override
protected void processImage() {
  rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth,
      previewHeight);
  final int imageSizeX = classifier.getImageSizeX();
  final int imageSizeY = classifier.getImageSizeY();

  runInBackground(
      new Runnable() {
        @Override
        public void run() {
          if (classifier != null) {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results =
                classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.v("Detect: %s", results);

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showResultsInBottomSheet(results);
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(imageSizeX + "x" + imageSizeY);
                    showCameraResolution(imageSizeX + "x" + imageSizeY);
                    showRotationInfo(String.valueOf(sensorOrientation));
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
          readyForNextImage();
        }
      });
}
```

Another important role of `ClassifierActivity` is to determine user preferences
(by interrogating `CameraActivity`), and instantiate the appropriately
configured `Classifier` subclass. This happens when the video feed begins (via
`onPreviewSizeChosen()`) and when options are changed in the UI (via
`onInferenceConfigurationChanged()`).

```java
private void recreateClassifier(Model model, Device device, int numThreads) {
  if (classifier != null) {
    LOGGER.d("Closing classifier.");
    classifier.close();
    classifier = null;
  }
  if (device == Device.GPU && model == Model.QUANTIZED) {
    LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
    runOnUiThread(
        () -> {
          Toast.makeText(this, "GPU does not yet supported quantized models.",
              Toast.LENGTH_LONG)
              .show();
        });
    return;
  }
  try {
    LOGGER.d(
        "Creating classifier (model=%s, device=%s, numThreads=%d)", model,
        device, numThreads);
    classifier = Classifier.create(this, model, device, numThreads);
  } catch (IOException e) {
    LOGGER.e(e, "Failed to create classifier.");
  }
}
```
