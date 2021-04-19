# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""CLI utility to generate APIs.

Usage:
python api_gen --output_dir=<path to output> --version=0.x.x
"""

import argparse
import json
import pathlib
from typing import Dict, Sequence

from tensorflow_examples.lite.model_maker.core.api import api_util

DEFAULT_API_FILE = 'golden_api.json'


def parse_arguments():
  """Parse arguments for API gen."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o', '--output_dir', type=str, help='Base dir to output generated APIs.')
  parser.add_argument(
      '-i',
      '--input_json',
      type=str,
      default=DEFAULT_API_FILE,
      help='JSON file for Golden APIs.')
  parser.add_argument(
      '-v',
      '--version',
      type=str,
      default='0.0.0dev',
      help='Version of the package.')
  return parser.parse_args()


def _read_golden_text(json_file: str) -> str:
  """Reads Golden file as text."""
  path = pathlib.Path(__file__).with_name(json_file)
  with open(path) as f:
    return f.read()


def load_golden(json_file: str) -> Dict[str, Sequence[str]]:
  """Loads Golden APIs."""
  content = _read_golden_text(json_file)
  return json.loads(content)


def run(output_dir: str, input_json: str, base_package: str,
        version: str) -> None:
  """Runs main."""
  imports = load_golden(input_json)
  api_util.write_packages(output_dir, imports, base_package, version)


def main() -> None:
  args = parse_arguments()
  run(args.output_dir, args.input_json, api_util.PACKAGE_PREFIX, args.version)


if __name__ == '__main__':
  main()
