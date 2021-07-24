# TensorFlow Lite BERT QA Android example

This document walks through the code of a simple Android mobile application that
demonstrates
[BERT Question and Answer](https://www.tensorflow.org/lite/examples/bert_qa/overview).

## Explore the code

The app is written entirely in Java and uses the TensorFlow Lite
[Java library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java)
for performing BERT Question and Answer.

We're now going to walk through the most important parts of the sample code.

### Get the question and the context of the question

This mobile application gets the question and the context of the question using the functions defined in the
file
[`QaActivity.java`](https://github.com/tensorflow/examples/blob/master/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/ui/QaActivity.java).


### Answerer

This BERT QA Android reference app demonstrates two implementation
solutions,
[`lib_task_api`](https://github.com/SunitRoy2703/examples/tree/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_task_api)
that leverages the out-of-box API from the
[TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer),
and
[`lib_interpreter`](https://github.com/SunitRoy2703/examples/tree/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_interpreter)
that creates the custom inference pipleline using the
[TensorFlow Lite Interpreter Java API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_java).

Both solutions implement the file `QaClient.java` (see
[the one in lib_task_api](https://github.com/SunitRoy2703/examples/blob/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/bertqa/ml/QaClient.java)
and
[the one in lib_interpreter](https://github.com/SunitRoy2703/examples/blob/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_interpreter/src/main/java/org/tensorflow/lite/examples/bertqa/ml/QaClient.java)
that contains most of the complex logic for processing the text input and
running inference.

#### Using the TensorFlow Lite Task Library

Inference can be done using just a few lines of code with the
[`BertQuestionAnswerer`](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer)
in the TensorFlow Lite Task Library.

##### Load model and create BertQuestionAnswerer

`BertQuestionAnswerer` expects a model populated with the
[model metadata](https://www.tensorflow.org/lite/convert/metadata) and the label
file. See the
[model compatibility requirements](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_question_answerer#model_compatibility_requirements)
for more details.


```java
/**
 * Load TFLite model and create BertQuestionAnswerer instance.
 */
 public void loadModel() {
     try {
         answerer = BertQuestionAnswerer.createFromFile(context, MODEL_PATH);
     } catch (IOException e) {
         Log.e(TAG, e.getMessage());
     }
 }
```

`BertQuestionAnswerer` currently does not support configuring delegates and
multithread, but those are on our roadmap. Please stay tuned!

##### Run inference

The following code runs inference using `BertQuestionAnswerer` and predicts the possible answers

```java
 /**
  * Run inference and predict the possible answers.
  */
      List<QaAnswer> apiResult = answerer.answer(contextOfTheQuestion, questionToAsk);
     
```

The output of `BertQuestionAnswerer` is a list of [`QaAnswer`](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa/QaAnswer.java) instance, where
each `QaAnswer` element is a single head classification result. All the
demo models are single head models, therefore, `results` only contains one
`QaAnswer` object.

To match the implementation of
[`lib_interpreter`](https://github.com/SunitRoy2703/examples/tree/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_interpreter),
`results` is converted into List<[`Answer`](https://github.com/SunitRoy2703/examples/blob/bertQa-android-task-lib/lite/examples/bert_qa/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/bertqa/ml/Answer.java)>.

#### Using the TensorFlow Lite Interpreter

##### Load model and create interpreter

To perform inference, we need to load a model file and instantiate an
`Interpreter`. This happens in the `loadModel` method of the `QaClient` class. Information about number of threads is used to configure the `Interpreter` via the
`Interpreter.Options` instance passed into its constructor.

```java
Interpreter.Options opt = new Interpreter.Options();
      opt.setNumThreads(NUM_LITE_THREADS);
      tflite = new Interpreter(buffer, opt);
...
```

##### Pre-process query & content

Next in the `predict` method of the `QaClient` class, we take the input of query & content,
convert it to a `Feature` format for efficient processing and pre-process
it. The steps are shown in the public 'FeatureConverter.convert()' method:

```java

public Feature convert(String query, String context) {
    List<String> queryTokens = tokenizer.tokenize(query);
    if (queryTokens.size() > maxQueryLen) {
      queryTokens = queryTokens.subList(0, maxQueryLen);
    }

    List<String> origTokens = Arrays.asList(context.trim().split("\\s+"));
    List<Integer> tokenToOrigIndex = new ArrayList<>();
    List<String> allDocTokens = new ArrayList<>();
    for (int i = 0; i < origTokens.size(); i++) {
      String token = origTokens.get(i);
      List<String> subTokens = tokenizer.tokenize(token);
      for (String subToken : subTokens) {
        tokenToOrigIndex.add(i);
        allDocTokens.add(subToken);
      }
    }

```

##### Run inference

Inference is performed using the following in `QaClient` class:

```java
tflite.runForMultipleInputsOutputs(inputs, output);
```

### Display results

The QaClient is invoked and inference results are displayed by the
`presentAnswer()` function in
[`QaActivity.java`](https://github.com/SunitRoy2703/examples/blob/bertQa-android-task-lib/lite/examples/bert_qa/android/app/src/main/java/org/tensorflow/lite/examples/bertqa/QaActivity.java).

```java
private void presentAnswer(Answer answer) {
        // Highlight answer.
        Spannable spanText = new SpannableString(content);
        int offset = content.indexOf(answer.text, 0);
        if (offset >= 0) {
            spanText.setSpan(
                    new BackgroundColorSpan(getColor(R.color.tfe_qa_color_highlight)),
                    offset,
                    offset + answer.text.length(),
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        contentTextView.setText(spanText);

        // Use TTS to speak out the answer.
        if (textToSpeech != null) {
            textToSpeech.speak(answer.text, TextToSpeech.QUEUE_FLUSH, null, answer.text);
        }
    }
```
