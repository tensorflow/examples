#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e  # Exit immediately when one of the commands fails.
set -x  # Verbose

# Prerequisites: The following envvars should be set when running this script.
#  - ANDROID_HOME: Android SDK location (tested with Android SDK 29)
#  - JAVA_HOME:    Java SDK location (tested with Open JDK 8)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EXAMPLES_DIR="$(realpath "${SCRIPT_DIR}/../examples")"

# Keep a list of android apps which should be excluded from the CI builds.
SKIPPED_BUILDS="
"

function build_android_example {
  # Check if this directory appears in the skipped builds list.
  RELATIVE_DIR="${1#"${EXAMPLES_DIR}/"}"
  if echo "${SKIPPED_BUILDS}" | grep -qx "${RELATIVE_DIR}"; then
    echo "WARNING: Skipping build for ${RELATIVE_DIR}."
    return 0
  fi

  echo "=== BUILD STARTED: ${RELATIVE_DIR} ==="

  pushd "$1" > /dev/null

  # Check if the directory contains a gradle wrapper.
  if [[ -x "$1/gradlew" ]]; then
    # Run the "build" task with the gradle wrapper.
    ./gradlew clean assembleRelease --stacktrace
  elif [[ -x "$1/finish/gradlew" ]]; then
    # Accommodate codelab directory
    ./finish/gradlew clean assembleRelease --stacktrace
  else
    echo "ERROR: Gradle wrapper could not be found under ${RELATIVE_DIR}."
    exit 1
  fi

  popd > /dev/null

  echo "=== BUILD FINISHED: ${RELATIVE_DIR} ==="
  echo
  echo
}


function build_smartreply_aar {
  # Builds once only after Smart Reply Android app.
  # It is to use bazel to build and create AAR for custom ops in cc.
  # TODO(tianlin): To generalize as pre-/post-build.
  RELATIVE_DIR="${1#"${EXAMPLES_DIR}/"}"

  # Run this only for smart_reply/android.
  if [[ "${RELATIVE_DIR}" != "smart_reply/android" ]]; then
    return 0
  fi
  WORKSPACE_DIR="${EXAMPLES_DIR}/smart_reply/android/app"
  echo "=== BUILD STARTED: ${RELATIVE_DIR} :: build_smartreply_aar ==="

  pushd "$1" > /dev/null

  cd "${WORKSPACE_DIR}"
  echo "-- Building in directory: ${WORKSPACE_DIR} --"
  /usr/bin/gcc -v
  bazel version  # Get bazel version info.
  # Add --sandbox_debug to provide more info for testing.
  bazel build --sandbox_debug //libs/cc/... //libs/cc:smartreply_runtime_aar
  bazel test //libs/cc/...

  popd > /dev/null

  echo "=== BUILD STARTED: ${RELATIVE_DIR} :: build_smartreply_aar ==="
  echo
  echo
}

time build_android_example "$1"

time build_smartreply_aar "$1"
