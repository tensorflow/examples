cmake_minimum_required(VERSION 3.16.3)

project("tflite-c-sample")

find_package(tensorflowlite_jni_gms_client REQUIRED CONFIG)

include(FetchContent)
FetchContent_Declare(
        flatbuffers
        GIT_REPOSITORY https://github.com/google/flatbuffers.git
        # Keep in sync with //third_party/tensorflow/third_party/flatbuffers/workspace.bzl
        # and //third_party/tensorflow/lite/tools/cmake/modules/flatbuffers.cmake
        # GIT_TAG v24.3.25
        GIT_TAG e6463926479bd6b330cbcf673f7e917803fd5831
)
set(FLATBUFFERS_BUILD_FLATC OFF)
set(FLATBUFFERS_BUILD_TESTS OFF)
set(FLATBUFFERS_INSTALL OFF)
FetchContent_MakeAvailable(flatbuffers)

add_compile_definitions(TFLITE_IN_GMSCORE)
add_compile_definitions(TFLITE_WITH_STABLE_ABI)
add_compile_definitions(TFLITE_USE_OPAQUE_DELEGATE)

add_library(tflite-jni SHARED
        com_google_samples_gms_tflite_c_TfLiteJni.cc
        logging_assert.h
        java_interop.h)

target_link_libraries(tflite-jni
        tensorflowlite_jni_gms_client::tensorflowlite_jni_gms_client
        flatbuffers
        android
        log)
