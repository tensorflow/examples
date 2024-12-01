cmake_minimum_required(VERSION 3.16.3)

project("tflite-c-sample")

find_package(tensorflowlite_jni_gms_client REQUIRED CONFIG)

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
