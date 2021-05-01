# Lint as: python3
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
r"""Generate python docs for tf.lite.

# How to run

```
python build_docs.py --output_dir=/path/to/output
```

"""
import pathlib

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_examples
import tflite_model_maker

import yaml


class OrderedDumper(yaml.dumper.Dumper):
  pass


def _dict_representer(dumper, data):
  """Force yaml to output dictionaries in order, not alphabetically."""
  return dumper.represent_dict(data.items())


OrderedDumper.add_representer(dict, _dict_representer)

flags.DEFINE_string('output_dir', '/tmp/mm_api/',
                    'The path to output the files to')

flags.DEFINE_string('code_url_prefix',
                    'https://github.com/tensorflow/examples/blob/master/',
                    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', '/lite/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS


def main(_):
  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Lite Model Maker',
      py_modules=[('tflite_model_maker', tflite_model_maker)],
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      callbacks=[public_api.explicit_package_contents_filter])

  doc_generator.build(output_dir=FLAGS.output_dir)

  toc_file = pathlib.Path(FLAGS.output_dir) / 'tflite_model_maker/_toc.yaml'
  toc = yaml.safe_load(toc_file.read_text())

  ## Nest all sub-modules under the root module instead of beside it.
  #
  # Before:
  #
  #  mm
  #  mm.compat
  #  mm.configs
  #
  # After:
  #
  #  mm
  #    compat
  #    configs

  # The first item of the toc is the root module.
  mm = toc['toc'][0]
  mm['status'] = 'experimental'
  # Shorten the title, and insert each sub-modules into the root module's
  # "section"
  sub_sections = mm['section']
  # The remaining items are the submodules
  for section in toc['toc'][1:]:
    section['title'] = section['title'].replace('tflite_model_maker.', '')
    sub_sections.append(section)
  # replace the list of (sub)modules with the root module.
  toc['toc'] = [mm]

  with toc_file.open('w') as f:
    yaml.dump(toc, f, Dumper=OrderedDumper)


if __name__ == '__main__':
  app.run(main)
