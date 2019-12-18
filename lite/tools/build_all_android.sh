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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EXAMPLES_DIR="$(realpath "${SCRIPT_DIR}/../examples")"

# Finds all <example_name>/android directories under lite/examples.
# Runs the android build script for each of those directories.
function build_android_examples {
  # In case the one of the build fails, overwrite the exit code to 255, in order
  # to make xargs to terminate early and stop processing the remaining builds.
  find "${EXAMPLES_DIR}" -mindepth 2 -maxdepth 2 -type d -name android -print0 \
    | xargs -0 -n1 -I{} bash -c \
        "${SCRIPT_DIR}/build_android_app.sh \"{}\" || exit 255"
}

build_android_examples
