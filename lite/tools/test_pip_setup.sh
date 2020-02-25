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

# The script is to pip install tensorflow-examples and test the package.
set -e  # Exit immediately when one of the commands fails.
set -x  # Verbose

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(realpath "${SCRIPT_DIR}/../..")"
PYTHON_BIN="$(which python3.7)"
PIP_BIN="$(which pip3.7)"
PIP_OPTIONS="--user"

function test_pip_setup {
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi
  if [[ "${PIP_BIN}" == "" ]]; then
    echo "pip is not available."
    exit 1
  fi

  echo "=== TEST SETUP STARTED IN: ${WORKSPACE_DIR} ==="

  pushd "${WORKSPACE_DIR}" > /dev/null

  # Replace version in setup.py to avoid error if there is no .git folder:
  # "version = subprocess.check_output(...)" -> "version = '0.0.1-test'"
  echo "--- Begin replacing version in setup.py ---"
  sed -i "s/^version = /version = '0.0.1-test' # /g" setup.py
  cat setup.py
  echo "--- End of setup.py ---"

  # Run pip install.
  ${PIP_BIN} install -e .[model_customization] ${PIP_OPTIONS}

  # Dry run to import the package
  ${PYTHON_BIN} -c "import tensorflow_examples"

  # Uninstall tensorflow-examples
  yes | ${PIP_BIN} uninstall tensorflow-examples

  popd > /dev/null

  echo "=== TEST SETUP FINISHED IN: ${WORKSPACE_DIR} ==="
  echo
  echo
}

test_pip_setup
