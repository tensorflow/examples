# TensorFlow Lite BERT QA iOS Example Example Application

![UIKit screencast]       | ![SwiftUI screencast]
:-----------------------: | :-------------------------:
UIKit version screen cast | SwiftUI version screen cast

## Overview

This is an end-to-end example of [BERT] Question & Answer application built with
TensorFlow 2.0, and tested on [SQuAD] dataset version 1.1. The demo app provides
48 passages from the dataset for users to choose from, and gives 5 most possible
answers corresponding to the input passage and query.

This example includes two types of application and a test set, one application
is developed with [UIKit], and the other is developed with [SwiftUI]. Each
application can be run with this XCode project, by choosing the target to build.
Each application shares the core logic needed to run the BertQA model. The test
set tests this core logic.

Question input to the application is discarded after inference.

### Model used

[BERT], or Bidirectional Encoder Representations from Transformers, is a method
of pre-training language representations which obtains state-of-the-art results
on a wide array of Natural Language Processing tasks.

This app uses MobileBERT, a compressed version of [BERT] that runs 4x faster and
has 4x smaller model size.

For more information, refer to the [BERT github page][BERT].

### Preprocessing

While preprocessing, it tokenizes input query and content with given vocabulary
data. Tokenization process mostly followed the [bert tokenization], except
handling Chinese characters as we don’t have passages written in Chinese.

It hands over the IDs of tokens with two other data. One is its segment ID to
verify whether it is a question or a content. The other is a mask, which
indicates if the given token is valid input to be processed or just a padding to
fit the input tensor. As the model requires a fixed size of token ID array, some
entries of the array could have invalid data.

### Inference

Assign preprocessed data to the input tensor and run the model. Output data is
assigned to the output tensors as a result.

### Postprocessing

While postprocessing, it retrieves original string from the arrays of start &
end logits in the output tensor. A logit of a token from `start` to `end` is
derived from a summation of start logit array’s `start`th value and end logit
array’s `end`th value. The higher sum of two logits, the more likely the token
between starting point and end point could be an answer.

After finding an answer, it retrieves the original string in the given range and
calculates the score. The score of the answer is calculated by the softmax
function.

## Requirements

*   Xcode 11.0 or above

*   Valid Apple Developer ID

*   Real iOS device

    Note: You can also use an iOS emulator, but some of the functionality may
    not be fully supported.

*   iOS version 12.0 or above

*   Xcode command line tools (to install, `run xcode-select --install`)

*   CocoaPods (to install, `run sudo gem install cocoapods`)

## Build and run

1.  Clone the TensorFlow examples GitHub repository to your computer to get the
    demo application: `git clone https://github.com/tensorflow/examples`

1.  Install the pod to generate the workspace file: `cd
    examples/lite/examples/bert_qa/ios && pod install`

    Note: If you have installed this pod before and that command doesn't work,
    try `pod update`. At the end of this step you should have a directory called
    `BertQA.xcworkspace`.

1.  Open the project in Xcode with the following command: `open
    BertQA.xcworkspace`

    This launches Xcode and opens the BertQA project.

1.  In the Menu bar, select `Product` → `Destination` and choose your device.

1. Follow the direction below if you want to:
    *   Run the application:
        1.  In the Menu bar, select `Product` → `Scheme` and choose
            `BertQA-UIKit` or `BertQA-SwiftUI`.
        1.  In the Menu bar, select `Product` → `Run` to install the app on your
            device.
    *   Test the core logic:

        1.  In the Menu bar, select `Product` --> `Scheme` and choose
            `BertQA-UIKit`.
        1.  In the Menu bar, select `Product` --> `Test`.

[UIKit screencast]: https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/bertqa_ios_uikit_demo.gif
[SwiftUI screencast]: https://storage.googleapis.com/download.tensorflow.org/models/tflite/screenshots/bertqa_ios_swiftui_demo.gif
[BERT]: https://github.com/google-research/bert
[SQuAD]: https://rajpurkar.github.io/SQuAD-explorer/
[UIKit]: https://developer.apple.com/documentation/uikit
[SwiftUI]: https://developer.apple.com/documentation/swiftui
[bert tokenization]: https://github.com/google-research/bert#tokenization
