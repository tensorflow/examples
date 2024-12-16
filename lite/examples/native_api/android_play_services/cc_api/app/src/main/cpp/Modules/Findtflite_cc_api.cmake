# This CMake code fragment determines the location of the tflite_cc_api package.

# tflite_cc_api_DIR is the directory containing the CMakeLists.txt file
# for building the TFLite-in-Play-services C++ API client SDK source files.
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/tflite_cc_sdk/CMakeLists.txt")
        message(SEND_ERROR "CMakeLists.txt not found in ${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
elseif (NOT EXISTS "${CMAKE_SOURCE_DIR}/tflite_cc_sdk/tensorflow/lite/abi/cc/interpreter.cc")
        message(SEND_ERROR "tensorflow/lite/abi/cc/interpreter.cc not found in ${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
else()
        set(tflite_cc_api_DIR "${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
endif()

# tflite_cc_api_INCLUDE_DIR is the include directory containing the header
# files for the TFLite-in-Play-services C++ API.
if (NOT EXISTS "${CMAKE_SOURCE_DIR}/tflite_cc_sdk/tensorflow/lite/interpreter.h")
        message(SEND_ERROR "tensorflow/lite/interpreter.h not found in ${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
elseif (NOT EXISTS "${CMAKE_SOURCE_DIR}/tflite_cc_sdk/tensorflow/lite/abi/cc/interpreter.h")
        message(SEND_ERROR "tensorflow/lite/abi/cc/interpreter.h not found in ${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
else()
        set(tflite_cc_api_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tflite_cc_api DEFAULT_MSG
                                  tflite_cc_api_DIR tflite_cc_api_INCLUDE_DIR)
