#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
EXAMPLES_DIR="$(realpath "${SCRIPT_DIR}/../examples")"

PROJECT_EXT=".xcodeproj"
WORKSPACE_EXT=".xcworkspace"

# Keep a list of iOS apps which should be excluded from the CI builds.
SKIPPED_BUILDS="
gesture_classification/ios
classification_by_retrieval/ios
"

INSTALL_LATEST_NIGHTLY_VERSION="${2:-false}"

echo "Example path: $1"
echo "Install latest nightly version: $2"

function build_ios_example {
  # Check if this directory appears in the skipped builds list.
  RELATIVE_DIR="${1#"${EXAMPLES_DIR}/"}"
  if echo "${SKIPPED_BUILDS}" | grep -qx "${RELATIVE_DIR}"; then
    echo "WARNING: Skipping build for ${RELATIVE_DIR}."
    return 0
  fi

  echo "=== BUILD STARTED: ${RELATIVE_DIR} ==="

  pushd "$1" > /dev/null

  # Cleanly install the dependencies
  # Retry a few times to workaround intermittent download errors.
  MAX_RETRY=3
  INSTALLED=false
  for i in $(seq 1 ${MAX_RETRY})
  do
    echo "Trying to install dependencies... (trial $i)"
    if "$INSTALL_LATEST_NIGHTLY_VERSION"; then
      if pod update --verbose --repo-update --clean-install; then
        INSTALLED=true
        break
      fi
    else
      if pod install --verbose --repo-update --clean-install; then
        INSTALLED=true
        break
      fi
    fi
  done

  if [[ "${INSTALLED}" == false ]]; then
    echo "Exceeded the max retry limit (${MAX_RETRY}) of pod install command."
    exit 1
  fi

  # Extract the scheme names.
  PROJECT_NAME="$(find * -maxdepth 0 -type d -name "*${PROJECT_EXT}")"
  WORKSPACE_NAME="$(find * -type d -name "*${WORKSPACE_EXT}")"
  SCHEMES="$(xcodebuild -list -project "${PROJECT_NAME}" -json | jq -r ".project.schemes[]")"

  # Build each scheme without code signing.
  for scheme in ${SCHEMES}; do
    # Due to an unknown issue prior to Xcode 11.4, a non-existing test scheme
    # might appear in the list of project schemes. For now, if a scheme name
    # contains the word "Tests", skip the build for that particular scheme.
    if [[ "${scheme}" == *"Tests"* ]]; then
      continue
    fi

    echo "--- BUILDING SCHEME ${scheme} FOR PROJECT ${RELATIVE_DIR} ---"
    set -o pipefail && xcodebuild \
        CODE_SIGN_IDENTITY="" \
        CODE_SIGNING_REQUIRED="NO" \
        CODE_SIGN_ENTITLEMENTS="" \
        CODE_SIGNING_ALLOWED="NO" \
        ARCHS="arm64" \
        -scheme "${scheme}" \
        -workspace "${WORKSPACE_NAME}" \
        | xcpretty  # Pretty print the build output.
    echo "--- FINISHED BUILDING SCHEME ${scheme} FOR PROJECT ${RELATIVE_DIR} ---"
  done

  popd > /dev/null

  echo "=== BUILD FINISHED: ${RELATIVE_DIR} ==="
  echo
  echo
}

build_ios_example "$1"
