cmake_minimum_required(VERSION 3.16.3)

project("tflite-cc-sample")

#-----------------------------------------------------------------------------
# Set up FlatBuffers dependency.
# (tflite_cc_api requires absl, and absl requires flatbuffers.)

include(FetchContent)
FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        # Keep in sync with the FlatBuffers version in the most recent version
        # of TF Lite (Lite RT) for Play services published on maven.google.com
        # <https://maven.google.com/web/index.html#com.google.android.gms:play-services-tflite-java>
        # and in particular as determined by the `static_assert` in the file
        # `tensorflow/lite/acceleration/configuration/configuration_generated.h`
        # in the `prefab/modules/tensorflowlite_jni_gms_client/include/` directory
        # of the `play-services-tflite-java-<version>.aar` file there.
        #
        # That version is currently "GIT_TAG v24.3.25",
        # but due to a build error in v24.3.25, we use a snapshot that is three PRs
        # later (all bug fixes), which fixes the build error.
        # See <https://github.com/google/flatbuffers/commit/e6463926479bd6b330cbcf673f7e917803fd5831>.
        GIT_TAG e6463926479bd6b330cbcf673f7e917803fd5831
)
set(FLATBUFFERS_BUILD_FLATC OFF)
set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)
FetchContent_MakeAvailable(flatbuffers)

#-----------------------------------------------------------------------------
# Set up ABSL dependency.

FetchContent_Declare(
        absl
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240722.0 # Released on Aug 1, 2024
)
FetchContent_MakeAvailable(absl)

#-----------------------------------------------------------------------------
# Set up TFLite in Play services C API (tensorflowlite_jni_gms_client) dependency.

find_package(tensorflowlite_jni_gms_client REQUIRED CONFIG)

#-----------------------------------------------------------------------------
# Set up TFLite in Play services C++ API (tflite_cc_api) dependency.

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Modules")

find_package(tflite_cc_api REQUIRED MODULE)
include_directories(${tflite_cc_api_INCLUDE_DIR})
add_subdirectory(${tflite_cc_api_DIR} tflite_cc_api_build)

#-----------------------------------------------------------------------------
# Set up compile definitions to enable use of TFLite in Play services
# (rather than regular TFLite bundled with the app).

add_compile_definitions(TFLITE_IN_GMSCORE)
add_compile_definitions(TFLITE_WITH_STABLE_ABI)
add_compile_definitions(TFLITE_USE_OPAQUE_DELEGATE)

#-----------------------------------------------------------------------------
# Define how to build this sample app's native code,
# which is embedded in a JNI library so that it can be
# called from the sample app's Java code.

add_library(tflite-cc-sample-jni SHARED
        com_google_samples_gms_tflite_cc_TfLiteJni.cc
        logging_assert.h
        java_interop.h)

target_link_libraries(tflite-cc-sample-jni
        tensorflowlite_jni_gms_client::tensorflowlite_jni_gms_client
        tflite_cc_api::tflite_cc_api
        flatbuffers
        android
        log)
