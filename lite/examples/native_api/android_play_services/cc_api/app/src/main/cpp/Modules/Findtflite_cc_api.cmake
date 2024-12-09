# This CMake code fragment determines the location of the tflite_cc_api package.
set(tflite_cc_api_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tflite_cc_sdk")
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tflite_cc_api DEFAULT_MSG
                                  tflite_cc_api_INCLUDE_DIR)
