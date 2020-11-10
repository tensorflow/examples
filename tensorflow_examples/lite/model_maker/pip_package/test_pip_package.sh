#!/bin/bash
# Test Model Maker's pip package.
# bash test_pip_package.sh

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKSPACE_DIR="$(realpath "${SCRIPT_DIR}/../../../../")"

source "${SCRIPT_DIR}/create_venv.sh"
create_venv_or_activate  # Use virtualenv and activate.

PYTHON_BIN="$(which python3.7)"
PIP_BIN="$(which pip3.7)"
CLI_BIN="tflite_model_maker"

function build_pip_and_install {
  # Build and install pip package.
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi

  local ver=""
  local pkg="tflite_model_maker"
  if [[ "$1" == "--nightly" ]]; then
    pkg="${pkg}_nightly"
    ver="--nightly"
  fi

  echo "------ build pip and install -----"
  pushd "${SCRIPT_DIR}" > /dev/null

  rm -r -f dist   # Clean up distributions.
  ${PYTHON_BIN} setup.py ${ver?} sdist bdist_wheel
  local dist_pkg="$(ls dist/${pkg}*.whl)"
  ${PIP_BIN} install ${dist_pkg?} --ignore-installed

  popd > /dev/null
  echo
}

function uninstall_pip {
  # Uninstall pip package.
  echo "------ uninstall pip -----"

  local pip_pkg="tflite-model-maker"
  if [[ "$1" == "--nightly" ]]; then
    pip_pkg="${pip_pkg}-nightly"
  fi

  yes | ${PIP_BIN} uninstall ${pip_pkg?}
  echo
}

function test_import {
  # Test whether import is successful
  echo "------ Test import -----"
  ${PYTHON_BIN} -c "import tflite_model_maker"
  echo
}

function test_cli {
  # Test CLI
  echo "------ Test CLI -----"
  yes | ${CLI_BIN}
}

function test_unittest {
  TEST_DIR="${WORKSPACE_DIR}/tensorflow_examples/lite/model_maker"

  echo "=== BEGIN UNIT TESTS FOR: ${TEST_DIR} ==="

  # Tests are excluded from pip, so need to be in root folder to test.
  pushd "${WORKSPACE_DIR}" > /dev/null

  # Set environment variables: test_srcdir for unit tests; and then run tests
  # one by one.
  export TEST_SRCDIR=${TEST_DIR}
  # Tests all but "*v1_test".
  find "${TEST_DIR}" -name "*[^v][^1]_test.py" -print0 | xargs -0 -I{} ${PYTHON_BIN?} {}

  popd > /dev/null
  echo "=== END UNIT TESTS: ${TEST_DIR} ==="
  echo
  echo
}

function test_model_maker {
  if [[ "$1" == "--nightly" ]]; then
    echo "===== Test Model Maker (nightly) ====="
  else
   echo "===== Test Model Maker (stable) ====="
  fi

  build_pip_and_install $1
  test_import
  test_cli
  test_unittest
  uninstall_pip $1
  echo
}

test_model_maker --nightly
test_model_maker
