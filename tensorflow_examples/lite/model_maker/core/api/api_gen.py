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
python api_gen --output_dir=<path to output>
"""

import argparse
import json
import pathlib
from typing import Dict, Sequence

from tensorflow_examples.lite.model_maker.core.api import api_util


def parse_arguments():
  """Parse arguments for API gen."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-o', '--output_dir', type=str, help='Base dir to output generated APIs.')
  parser.add_argument(
      '-i',
      '--input_json',
      type=str,
      default='golden_api.json',
      help='Json file for Golden APIs.')
  return parser.parse_args()


def load_golden(input_json: str) -> Dict[str, Sequence[str]]:
  """Loads Golden APIs."""
  path = pathlib.Path(__file__).with_name(input_json)
  with open(path) as f:
    return json.load(fp=f)


def run(output_dir: str, input_json: str) -> None:
  """Runs main."""
  imports = load_golden(input_json)
  api_util.write_packages(output_dir, imports)


def main() -> None:
  args = parse_arguments()
  run(args.output_dir, args.input_json)


if __name__ == '__main__':
  main()
