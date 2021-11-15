load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

MINIMUM_OS_VERSION = "15.0"

ios_application(
    name = "ImageClassifierBuilder",
    app_icons = glob(["ImageClassifierBuilder/Assets.xcassets/AppIcon.appiconset/**"]),
    bundle_id = "com.tensorflow.lite.swift.ImageClassifierBuilder",
    families = ["iphone"],
    infoplists = ["ImageClassifierBuilder/Info.plist"],
    minimum_os_version = MINIMUM_OS_VERSION,
    provisioning_profile = ":ProvisioningProfile.mobileprovision",
    deps = [
        ":ImageClassifierBuilderLib",
    ],
)

# Swift files target.
swift_library(
    name = "ImageClassifierBuilderLib",
    srcs = glob(["ImageClassifierBuilder/*.swift"]),
    data = glob(["ImageClassifierBuilder/TFLiteDocumentIcons/**"]),
    deps = [
        ":ImageClassifierBuilderLibObjC",
    ],
)

# Hide header files that contain C++, as otherwise the `objc_library` can't be used in a
# `swift_library`.
PRIVATE_HEADERS = [
    "ImageClassifierBuilder/NSString+AbseilStringView.h",
]

# Objective-C(++) files target.
objc_library(
    name = "ImageClassifierBuilderLibObjC",
    srcs = glob([
        "ImageClassifierBuilder/*.m",
        "ImageClassifierBuilder/*.mm",
    ]) + PRIVATE_HEADERS,
    hdrs = glob(
        ["ImageClassifierBuilder/*.h"],
        exclude = PRIVATE_HEADERS,
    ),
    data = [
        "@imagenet-mobilenet_v3_small_100_224-feature_vector//file",
    ],
    sdk_frameworks = [
        "Accelerate",
        "CoreGraphics",
        "CoreVideo",
        "Foundation",
        "UIKit",
    ],
    deps = [
        "//lib:model_builder",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/port:statusor",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision:image_classifier",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/core:frame_buffer",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:classifications_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:image_classifier_options_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/proto:image_embedder_options_proto_inc",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc/task/vision/utils:frame_buffer_common_utils",
    ],
)

exports_files(["ProvisioningProfile.mobileprovision"])
