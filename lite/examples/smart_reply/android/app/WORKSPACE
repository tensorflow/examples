"""SmartReply Workspace"""

workspace(name = "org_tensorflow_lite_examples_smartreply")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "ddce3b3a3909f99b28b25071c40b7fec7e2e1d1d1a4b2e933f3082aa99517105",
    strip_prefix = "rules_closure-316e6133888bfc39fb860a4f1a31cfcbae485aef",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/316e6133888bfc39fb860a4f1a31cfcbae485aef.tar.gz",  # 2019-03-21
    ],
)
http_archive(
    name = "bazel_skylib",
    sha256 = "2c62d8cd4ab1e65c08647eb4afe38f51591f43f7f0885e7769832fa137633dcb",
    strip_prefix = "bazel-skylib-0.7.0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/0.7.0.tar.gz"],
)
# END: Upstream TensorFlow dependencies

http_archive(
    name = "org_tensorflow",
    sha256 = "352819de805d58c859c8889bf279abde508dbd89d91c804c78cd8e700845a2c5",
    strip_prefix = "tensorflow-d9b09c432eb6fd176333277a83d9f364b540d529",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/d9b09c432eb6fd176333277a83d9f364b540d529.tar.gz",  # 2019-09-24
    ],
)

load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least(minimum_bazel_version="0.24.1")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")
tf_repositories(path_prefix="", tf_repo_name="org_tensorflow")

# Need to export environment variable ANDROID_HOME.
android_sdk_repository(
    name = "androidsdk",
)

# Need to export environment variable ANDROID_NDK_HOME.
android_ndk_repository(
    name = "androidndk",
)
