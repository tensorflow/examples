#!/bin/bash
# Create virtual environment.

set -e
set -x

VENV_HOME=/tmp/venv
PIP_FLAG="--user"

function create_venv_or_activate {
  local PY="$(which python3.7)"
  local PIP="$(which pip3.7)"
  # Test whether pip exists.
  if [[ "${PY}" == "" ]]; then
    echo "python is not available."
    exit 1
  fi

  if [[ ! -d "${VENV_HOME}" ]]; then
    # Install virtualenv and create VENV_HOME
    "${PIP?}" install virtualenv ${PIP_FLAG}
    "${PY?}" -m venv "${VENV_HOME}"
    source "${VENV_HOME}/bin/activate"

    # Install required package: twine, wheel
    PY="$(which python3.7)"
    "${PY?}" -m pip install twine wheel
  else
    source "${VENV_HOME}/bin/activate"
  fi
}
