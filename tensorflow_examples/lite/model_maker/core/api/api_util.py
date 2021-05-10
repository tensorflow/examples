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
"""Utilities for exporting TFLite Model Maker symbols as API.

Usage:

To export a function or a class, use mm_export decorator. e.g.:
```python
@mm_export('bar.foo')
def foo(...):
  ...
```

If a function is assigned to a variable, you can export it by calling
mm_export explicitly. e.g.:
```python
def get_foo(...):
  ...
mm_export('bar.foo')(get_foo)
```

Exporting a constant.
```python
FOO = 1
mm_export('bar.FOO').export_constant(__name__, 'FOO')
```
where __name__ gets the current module name, and 'FOO' is the constant.
"""
import collections
from collections.abc import Callable  # pylint: disable=g-importing-member
import inspect
import os
import pathlib
from typing import Dict, List, Tuple, Sequence, Optional, Union

import dataclasses

# Model Maker API.
NAME_TO_SYMBOL = {}

LICENSE = """# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""

# Package prefix name for model maker.
PACKAGE_PLACEHOLDER = '$PKG'
ROOT_PACKAGE_KEY = ''


@dataclasses.dataclass
class Symbol:
  """Symbol to export."""

  exported_name: str  # Exported name of the symbol
  exported_parts: List[str]  # Parts after splitting.
  func: Optional[Callable]  # Function or class.
  imported_module: str  # Imported module name.
  imported_name: str  # Imported symbol name.

  @classmethod
  def from_callable(cls, exported_name: str, func: Callable) -> 'Symbol':
    """Creates a symbol from a callable (function or class)."""
    if func is None:
      raise ValueError('func should not be None: {}'.format(func))
    imported_module, imported_name = _get_module_and_name(func)

    exported_parts = split_name(exported_name)
    return cls(
        exported_name=exported_name,
        exported_parts=exported_parts,
        func=func,
        imported_module=imported_module,
        imported_name=imported_name)

  @classmethod
  def from_constant(cls, exported_name: str, module: str,
                    name: str) -> 'Symbol':
    """Creates a symbol from a constant."""
    exported_parts = split_name(exported_name)
    return cls(
        exported_name=exported_name,
        exported_parts=exported_parts,
        func=None,
        imported_module=module,
        imported_name=name)

  def get_package_name(self) -> str:
    """Generated path."""
    return as_package(self.exported_parts[:-1])

  def gen_import(self) -> str:
    """Generates import text line."""
    as_name = self.exported_parts[-1]
    if as_name == self.imported_name:
      import_line = 'from {} import {}'.format(self.imported_module,
                                               self.imported_name)
    else:
      import_line = 'from {} import {} as {}'.format(self.imported_module,
                                                     self.imported_name,
                                                     as_name)
    return import_line

  def gen_parents_import(self) -> Dict[str, Sequence[str]]:
    """Generates parents import."""
    length = len(self.exported_parts)
    parents_import = {}
    for i in range(length - 1):
      parts = self.exported_parts[:i]
      import_name = self.exported_parts[i]
      parent = as_package(parts)
      abs_package = split_name(PACKAGE_PLACEHOLDER) + parts
      abs_package = as_package(abs_package)
      import_line = 'from {} import {}'.format(abs_package, import_name)
      parents_import[parent] = [import_line]
    return parents_import


def split_name(name: str) -> List[str]:
  """Splits name and returns a list of segments.

  Args:
    name: str.

  Returns:
    list of str: package segments
  """
  parts = name.split('.')
  return list(filter(lambda n: n, parts))


def as_package(names: List[str]) -> str:
  """Joins names as a package name."""
  return '.'.join(names)


def as_path(names: List[str]) -> str:
  """Joins names as a file path."""
  if names:
    return os.path.join(*names)
  else:
    return ''


def _get_module_and_name(func: Callable) -> Tuple[str, str]:
  """Gets module and name, or raise error if not a function."""
  if not inspect.isfunction(func) and not inspect.isclass(func):
    raise ValueError('Expect a function or class, but got: {}'.format(func))
  return func.__module__, func.__name__


class mm_export:  # pylint: disable=invalid-name
  """Exports model maker APIs."""

  def __init__(self, name: str):
    if name in NAME_TO_SYMBOL:
      raise ValueError('API already exists: `{}`.'.format(name))
    self._exported_name = name  # API name.

  def __call__(self, func: Callable) -> Callable:
    """Exports function or class."""
    NAME_TO_SYMBOL[self._exported_name] = Symbol.from_callable(
        self._exported_name, func)
    return func

  def export_constant(self, module: str, name: str) -> None:
    """Exports constants."""
    NAME_TO_SYMBOL[self._exported_name] = Symbol.from_constant(
        self._exported_name, module, name)


def _reset_apis():
  """Resets all APIs."""
  global NAME_TO_SYMBOL
  NAME_TO_SYMBOL = {}


def _case_insensitive(s: str):
  """To sort with case insensitive."""
  return s.lower()


def generate_imports() -> Dict[str, Sequence[str]]:
  """Generates imports."""
  import_dict = collections.defaultdict(set)
  for _, symbol in NAME_TO_SYMBOL.items():
    package_name = symbol.get_package_name()
    import_line = symbol.gen_import()
    import_dict[package_name].add(import_line)

    for k, line in symbol.gen_parents_import().items():
      import_dict[k].update(line)

  # Add prefix and sort import values.
  abs_import_dict = {}
  for package_name, value_set in import_dict.items():
    parts = split_name(package_name)
    abs_package = as_package(parts)
    abs_import_dict[abs_package] = list(
        sorted(value_set, key=_case_insensitive))
  return abs_import_dict


def generate_package_doc(package_name):
  """Generates package doc."""
  return '"""Generated API for package: {}."""'.format(package_name)


def write_packages(
    base_dir: str,
    imports: Dict[str, Sequence[str]],
    doc_dict: Dict[str, str],
    base_package: str,
    version: str,
    deprecated_imports: Optional[Dict[str, Sequence[str]]] = None) -> None:
  """Writes packages as init files.

  Args:
    base_dir: str, base directory to write packages.
    imports: dict, pairs of (namespace, list of imports).
    doc_dict: dict, pairs of (namespace, package_doc).
    base_package: str, the base package name. (e.g. 'tflite_model_maker')
    version: str, version string. (e.g., 0.x.x).
    deprecated_imports: optinal dict, pairs of (namespace, list of imports).
  """
  if not deprecated_imports:
    deprecated_imports = {}

  for package_name, import_lines in imports.items():
    # Create parent dir.
    parts = as_path(split_name(package_name))
    parent_dir = os.path.join(base_dir, parts)
    make_dirs_or_not(parent_dir)

    # Write header and import lines.
    full_path = os.path.join(parent_dir, '__init__.py')

    lines = [
        line.replace(PACKAGE_PLACEHOLDER, base_package) for line in import_lines
    ]

    # Add deprecated imports for backward compatiblity..
    if package_name in deprecated_imports:
      lines.append(deprecated_imports[package_name])

    # For base package add __version__.
    if package_name == ROOT_PACKAGE_KEY:
      lines.append("""__version__ = '{}'""".format(version))

    full_package_name = as_package(
        split_name(base_package) + split_name(package_name))

    # Add package doc.
    if package_name in doc_dict:
      doc = '"""{}"""'.format(doc_dict[package_name])
    else:
      doc = generate_package_doc(full_package_name)

    write_python_file(full_path, doc, lines)


PathOrStrType = Union[pathlib.Path, str]


def make_dirs_or_not(dirpath: Union[PathOrStrType]):
  """Make dirs if not exists."""
  if not os.path.exists(dirpath):
    os.makedirs(dirpath)


def write_python_file(filepath: PathOrStrType, package_doc: Optional[str],
                      lines: Optional[Sequence[str]]):
  """Writes python file."""
  with open(filepath, 'w') as f:
    f.write(LICENSE)
    if package_doc:
      f.write(package_doc + '\n\n')
    if lines:
      for line in lines:
        f.write(line + '\n')
