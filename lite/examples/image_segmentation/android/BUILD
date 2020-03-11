load("//tools/build_defs/android:rules.bzl", "android_binary")
load("//tools/build_defs/kotlin:rules.bzl", "kt_android_library")

package(default_visibility = ["//third_party/py/tensorflow_examples:internal"])

licenses(["notice"])  # Apache 2.0

APP_DIR = "app/src/main/"

CUSTOM_PACKAGE = "org.tensorflow.lite.examples.imagesegmentation"

filegroup(
    name = "resources",
    srcs = glob([APP_DIR + "res/**"]),
)

android_binary(
    name = "ImageSegmentationApp",
    custom_package = CUSTOM_PACKAGE,
    manifest = APP_DIR + "AndroidManifest.xml",
    multidex = "native",
    nocompress_extensions = [
        ".tflite",
    ],
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        ":ImageSegmentationLib",
    ],
)

kt_android_library(
    name = "ImageSegmentationLib",
    srcs = glob([APP_DIR + "java/**/*.kt"]),
    assets = [
        "//third_party/py/tensorflow_examples/lite/examples/image_segmentation/android/app/src/main/assets",
    ],
    assets_dir = "",
    custom_package = CUSTOM_PACKAGE,
    manifest = APP_DIR + "AndroidManifest.xml",
    resource_files = [":resources"],
    tags = ["manual"],  # Do not build automatically, because DO_NOT_DEPEND_RES complains.
    deps = [
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/arch_components/lifecycle",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/arch_components/lifecycle/extensions",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/constraint_layout",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/coordinatorlayout",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/exifinterface",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v7/appcompat",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/v7/recyclerview",
        "//third_party/java/android_libs/glide",
        "//third_party/java/android_libs/material_components:bottomsheet",
        "//third_party/java/android_libs/material_components:chip",
        "//third_party/java/android_libs/material_components:floatingactionbutton",
        "//third_party/java/android_libs/material_components:theme",
        "//third_party/kotlin/kotlinx_coroutines",
        "//third_party/kotlin/kotlinx_coroutines:kotlinx_coroutines_android",  # unuseddeps: keep
        "//third_party/py/tensorflow_examples/lite/tools/shared/android:resource_lib",
        "//third_party/tensorflow/lite/java:tensorflowlite",
        "//third_party/tensorflow/lite/java:tensorflowlite_gpu",
    ],
)
