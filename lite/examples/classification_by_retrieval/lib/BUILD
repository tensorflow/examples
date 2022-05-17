# Description:
#    Classification-by-retrieval model builder library (C++).

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "tflite_builder",
    srcs = ["tflite_builder.cc"],
    hdrs = ["tflite_builder.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
    ],
)

cc_library(
    name = "tflite_cbr_builder",
    srcs = ["tflite_cbr_builder.cc"],
    hdrs = ["tflite_cbr_builder.h"],
    deps = [
        ":tflite_builder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:embeddings_proto_inc",
    ],
)

cc_library(
    name = "model_builder",
    srcs = ["model_builder.cc"],
    hdrs = ["model_builder.h"],
    deps = [
        ":tflite_cbr_builder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision:image_embedder",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/core:frame_buffer",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:embeddings_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:image_embedder_options_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/metadata/cc:metadata_populator",
    ],
)

cc_library(
    name = "labeled_image_helper",
    srcs = ["labeled_image_helper.cc"],
    hdrs = ["labeled_image_helper.h"],
    deps = [
        ":model_builder",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:status_macros",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/core:frame_buffer",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/utils:frame_buffer_common_utils",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/utils:image_utils",
    ],
)
