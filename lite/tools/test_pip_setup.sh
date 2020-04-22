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
# For pip install with --user, add pip local bin directory is in path.
export PATH=$PATH:~/.local/bin/

function test_pip_install {
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi
  if [[ "${PIP_BIN}" == "" ]]; then
    echo "pip is not available."
    exit 1
  fi

  ${PIP_BIN} install --upgrade pip ${PIP_OPTIONS}

  echo "=== TEST PIP INSTASLL IN: ${WORKSPACE_DIR} ==="

  pushd "${WORKSPACE_DIR}" > /dev/null

  # Replace version in setup.py to avoid error if there is no .git folder:
  # "version = subprocess.check_output(...)" -> "version = '0.0.1-test'"
  echo "--- Begin replacing version in setup.py ---"
  sed -i "s/^version = /version = '0.0.1-test' # /g" setup.py
  cat setup.py
  echo "--- End of setup.py ---"

  # Run pip install.
  ${PIP_BIN} install -e .[model_maker,tests,metadata] ${PIP_OPTIONS}

  popd > /dev/null
  echo
  echo
}

function test_model_maker() {
  TEST_DIR="${WORKSPACE_DIR}/tensorflow_examples/lite/model_maker"

  echo "=== BEGIN UNIT TESTS FOR: ${TEST_DIR} ==="
  pushd "${WORKSPACE_DIR}" > /dev/null

  # Set environment variables: test_srcdir for unit tests; and then run tests
  # one by one.
  export TEST_SRCDIR=${TEST_DIR}
  find "${TEST_DIR}" -name "*_test.py" -print0 | xargs -0 -I{} ${PYTHON_BIN?} {}

  popd > /dev/null
  echo "=== END UNIT TESTS: ${TEST_DIR} ==="
  echo
  echo
}

function test_pip_uninstall() {
  echo "=== TO UNINSTASLL PACKAGE ==="
  yes | ${PIP_BIN} uninstall tensorflow-examples
  echo
  echo
}

test_pip_install
# test_model_maker   # TODO(b/154565124): Import error occurs during tests.
test_pip_uninstall
