cmake_minimum_required(VERSION 3.4.1)

set(TFLITE_LIBPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libraries/tensorflowlite/jni")
set(TFLITE_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libraries/tensorflowlite/headers")
set(TFLITE_GPU_LIBPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libraries/tensorflowlite-gpu/jni")
set(TFLITE_GPU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libraries/tensorflowlite-gpu/headers")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
set(CMAKE_CXX_STANDARD 14)

add_library(SuperResolution SHARED SuperResolution_jni.cpp SuperResolution.cpp)

add_library(lib_tensorflowlite SHARED IMPORTED)
set_target_properties(lib_tensorflowlite PROPERTIES IMPORTED_LOCATION
        ${TFLITE_LIBPATH}/${ANDROID_ABI}/libtensorflowlite_jni.so)

add_library(lib_tensorflowlite_gpu SHARED IMPORTED)
set_target_properties(lib_tensorflowlite_gpu PROPERTIES IMPORTED_LOCATION
        ${TFLITE_GPU_LIBPATH}/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so)

find_library(log-lib log)

include_directories(${TFLITE_INCLUDE})
target_include_directories(SuperResolution PRIVATE
        ${TFLITE_INCLUDE})

include_directories(${TFLITE_GPU_INCLUDE})

target_include_directories(SuperResolution PRIVATE
        ${TFLITE_GPU_INCLUDE})

target_link_libraries(SuperResolution
                      android
                      lib_tensorflowlite
                      lib_tensorflowlite_gpu
                      # Links the target library to the log library
                      # included in the NDK.
                      ${log-lib})
