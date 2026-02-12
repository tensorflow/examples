# This CMake code fragment determines the location of the tflite_cc_api package.

set(gradle_src_root "${CMAKE_SOURCE_DIR}/../../..")

# tflite_cc_api_DIR is the directory containing the CMakeLists.txt file
# for building the TFLite-in-Play-services C++ API client SDK source files.
if (NOT EXISTS "${gradle_src_root}/build/tflite_cc_sdk/CMakeLists.txt")
        message(SEND_ERROR "CMakeLists.txt not found in ${gradle_src_root}/build/tflite_cc_sdk")
elseif (NOT EXISTS "${gradle_src_root}/build/tflite_cc_sdk/tensorflow/lite/abi/cc/interpreter.cc")
        message(SEND_ERROR "tensorflow/lite/abi/cc/interpreter.cc not found in ${gradle_src_root}/build/tflite_cc_sdk")
else()
        set(tflite_cc_api_DIR "${gradle_src_root}/build/tflite_cc_sdk")
endif()

# tflite_cc_api_INCLUDE_DIR is the include directory containing the header
# files for the TFLite-in-Play-services C++ API.
if (NOT EXISTS "${gradle_src_root}/build/tflite_cc_sdk/tensorflow/lite/interpreter.h")
        message(SEND_ERROR "tensorflow/lite/interpreter.h not found in ${gradle_src_root}/build/tflite_cc_sdk")
elseif (NOT EXISTS "${gradle_src_root}/build/tflite_cc_sdk/tensorflow/lite/abi/cc/interpreter.h")
        message(SEND_ERROR "tensorflow/lite/abi/cc/interpreter.h not found in ${gradle_src_root}/build/tflite_cc_sdk")
else()
        set(tflite_cc_api_INCLUDE_DIR "${gradle_src_root}/tflite_cc_sdk")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tflite_cc_api DEFAULT_MSG
                                  tflite_cc_api_DIR tflite_cc_api_INCLUDE_DIR)
