#! /bin/bash

# Clone TensorFlow Text repo
git clone https://github.com/tensorflow/text.git tensorflow_text

cd tensorflow_text/
echo 'exports_files(["LICENSE"])' > BUILD

# Checkout 2.12 branch 
git checkout 2.12

# Apply tftext-2.12.patch
git apply ../tftext-2.12.patch

# Run config
./oss_scripts/configure.sh

# Run bazel build 
bazel build -c opt --cxxopt='--std=c++14' --config=monolithic --config=android_x86_64  --experimental_repo_remote_exec //tensorflow_text:tftext_tflite_flex

if [ $? -eq 0 ]; then
    # Print a message
    echo "Please find the aar file: tensorflow_text/bazel-bin/tensorflow_text/tftext_tflite_flex.aar"
fi
