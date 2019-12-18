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

# Prerequisites: The following envvars should be set when running this script.
#  - ANDROID_HOME: Android SDK location (tested with Android SDK 29)
#  - JAVA_HOME:    Java SDK location (tested with Open JDK 8)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EXAMPLES_DIR="$(realpath "${SCRIPT_DIR}/../examples")"

# Keep a list of blacklisted android apps directories which should be excluded
# from the builds.
SKIPPED_BUILDS="
image_segmentation/android
model_personalization/android
style_transfer/android
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
  if ! [[ -x "$1/gradlew" ]]; then
    echo "ERROR: Gradle wrapper could not be found under ${RELATIVE_DIR}."
    exit 1
  fi

  # Run the "build" task with the gradle wrapper.
  ./gradlew clean build --stacktrace

  popd > /dev/null

  echo "=== BUILD FINISHED: ${RELATIVE_DIR} ==="
  echo
  echo
}

build_android_example "$1"
