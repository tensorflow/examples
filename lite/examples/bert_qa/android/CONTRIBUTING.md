Overview:
This project is a fork of the TensorFlow BERT QA Android app from tensorflow/examples, updated to support modern Android development practices. It uses BERT (Bidirectional Encoder Representations from Transformers) for question-answering tasks on mobile devices.

Build Details:
Gradle: 8.7
Android Gradle Plugin (AGP): 8.5.0
Kotlin: 1.9.20
Compile SDK: 35
Tested on: API 36 Beta (Pixel 9 XL virtual emulator)

How to Build and Run :
Clone the repository: ```bash git clone https://github.com/Chris-Hamid/tensorflow-examples.git ```
Open the project in Android Studio (version 2024.1.1 or later recommended).
Ensure the BERT model files are in `lite/examples/bert_qa/android/app/src/main/assets/`:
Download `mobilebert_float.tflite` from https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/mobilebert_float.tflite
Download `vocab.txt` from https://storage.googleapis.com/download.tensorflow.org/models/tflite/bert_qa/vocab.txt
Sync the project with Gradle (`File > Sync Project with Gradle Files`).
Build and run on an emulator or device with API 26+.
or Build APK & install on device with API 26+

Contributions:
This project is maintained by Chris-Hamid. Contributions are welcome! Please see the TensorFlow contribution guidelines for more details.

License :
This project is licensed under the Apache License 2.0, as per the original TensorFlow repository. See LICENSE for details.
" > lite/examples/bert_qa/android/README.md
