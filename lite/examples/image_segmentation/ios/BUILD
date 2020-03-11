load("//tools/build_defs/apple:ios.bzl", "ios_application")
load("//tools/build_defs/swift:swift.bzl", "swift_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

MINIMUM_OS_VERSION = "9.0"

swift_library(
    name = "ImageSegmentationLib",
    srcs = glob(["ImageSegmentation/*.swift"]),
    data = glob(["ImageSegmentation/Base.lproj/*.storyboard"]) + glob([
        "ImageSegmentation/model/*",
    ]),
    deps = [
        "//third_party/tensorflow/lite/experimental/swift:TensorFlowLite",
    ],
)

ios_application(
    name = "ImageSegmentation",
    app_icons = glob(["ImageSegmentation/Assets.xcassets/AppIcon.appiconset/**"]),
    bundle_id = "com.tensorflow.lite.swift.ImageSegmentation",
    families = [
        "iphone",
    ],
    infoplists = ["ImageSegmentation/Info.plist"],
    minimum_os_version = MINIMUM_OS_VERSION,
    provisioning_profile = "//googlemac/iPhone/Shared/ProvisioningProfiles:Google_Development.mobileprovision",
    sdk_frameworks = [
        "AVFoundation",
        "CoreGraphics",
        "CoreVideo",
        "CoreMedia",
    ],
    deps = [
        ":ImageSegmentationLib",
    ],
)
