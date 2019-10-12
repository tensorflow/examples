load("//tools/build_defs/android:rules.bzl", "android_binary", "android_library")

package(
    default_visibility = ["//third_party/py/tensorflow_examples:internal"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "assets",
    srcs = [
        "//third_party/py/tensorflow_examples/lite/examples/smart_reply/android/app/src/main/assets:assets_files",
    ],
)

APP_DIR = "app/src/main/"

android_binary(
    name = "SmartReply",
    srcs = ["app/src/main/java/org/tensorflow/lite/examples/smartreply/MainActivity.java"],
    aapt_version = "aapt2",
    assets = [":assets"],
    assets_dir = "",
    custom_package = "org.tensorflow.lite.examples.smartreply",
    manifest = APP_DIR + "AndroidManifest.xml",
    manifest_values = {
        "minSdkVersion": "15",
        "targetSdkVersion": "28",
    },
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob([APP_DIR + "res/**"]),
    tags = ["manual"],
    deps = [
        ":client_lib",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v13",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v4",
        "//third_party/java/android_libs/guava_jdk5:io",
    ],
)

android_library(
    name = "client_lib",
    srcs = [
        "app/src/main/java/org/tensorflow/lite/examples/smartreply/AssetsUtil.java",
        "app/src/main/java/org/tensorflow/lite/examples/smartreply/SmartReply.java",
        "app/src/main/java/org/tensorflow/lite/examples/smartreply/SmartReplyClient.java",
    ],
    assets = [":assets"],
    assets_dir = "",
    custom_package = "org.tensorflow.lite.examples.smartreply",
    manifest = APP_DIR + "AndroidManifest.xml",
    resource_files = glob([APP_DIR + "res/**"]),
    deps = [
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v13",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v4",
        "//third_party/java/android_libs/guava_jdk5:io",
        "//third_party/py/tensorflow_examples/lite/examples/smart_reply/android/app/libs/cc:smartreply_jni",
    ],
)
