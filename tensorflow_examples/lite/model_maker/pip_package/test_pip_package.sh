#!/bin/bash
# Test Model Maker's pip package.
# sh test_pip_package.sh

set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="$(which python3.7)"
PIP_BIN="$(which pip3.7)"
CLI_BIN="tflite_model_maker"

function build_pip_and_install {
  # Build and install pip package.
  if [[ "${PYTHON_BIN}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi
  if [[ "${PIP_BIN}" == "" ]]; then
    echo "pip is not available."
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

echo "===== Test Model Maker (nightly) ====="
build_pip_and_install --nightly
test_import
test_cli
uninstall_pip --nightly
echo

echo "===== Test Model Maker (stable) ====="
build_pip_and_install
test_import
test_cli
uninstall_pip
echo
