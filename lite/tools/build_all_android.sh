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
NUM_PROCESSES=16   # Run tests in parallel. Adjust this to your own machine.

# Finds all <example_name>/android directories under lite/examples.
# Runs the android build script for each of those directories.
#
# If two number arguments are provided, interpret them as "Part $1 of $2".
# That is, split the app_dir_list into $2 parts, and only run the $1-th part.
function build_android_examples {
  # NOTE: This script breaks if there are any spaces in the full path.
  app_dir_list=( $(find "${EXAMPLES_DIR}" -mindepth 2 -maxdepth 2 -type d -name android | sort) )
  target_list=( "${app_dir_list[@]}" )

  num_re="^[0-9]+$"
  if [[ "$1" =~ $num_re ]] && [[ "$2" =~ $num_re ]] && (( $1 <= $2 )); then
    total=${#app_dir_list[@]}
    batch_size=$(( $total / $2 ))
    start=$(( $batch_size * ($1 - 1) ))
    length=$batch_size

    # Make sure to process all the remainder if this is the last batch.
    if (( $1 == $2 )); then
      length=$(( $length + $total % $2 ))
    fi

    # Take the sublist.
    target_list=( "${app_dir_list[@]:$start:$length}" )
  fi

  # Run the builds in parallel.
  for app_dir in "${target_list[@]}"; do
    echo ${app_dir}
  done | xargs -0 -n 1 -P ${NUM_PROCESSES?} -I{} \
         ${SCRIPT_DIR}/build_android_app.sh {}
}

build_android_examples "$@"
