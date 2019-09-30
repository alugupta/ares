# Allows per-user configuration settings, specified in JSON files.
#
# I got tired of the /home/<username>/... and sys.path hacks, so I wrote
# up a replacement that allows those kind of variables to be specified external
# to the source code.

import copy
import os
import json
import types

import sys
# https://docs.python.org/2/library/json.html#py-to-json-table
_CONVERTIBLE_VALID_TYPES = [
  dict,
  tuple,
  list,
  bytes,
  str,
  int,
  int,
  float,
  bool,
  type(None),
]

_TEST_DEFAULTS = {
  "results_dir": "/ares/results",
  "cache_dir": "/ares/cache",
  "ds_dir": "/ba-dls-deepspeech"
}

class Conf(object):
  # Class attribute.
  _conf = dict()
  _path = None

  def __init__(self):
    assert False, 'Do not instantiate this class. Use class methods directly.'

  @staticmethod
  def _validate(obj):
    assert type(obj) in _CONVERTIBLE_VALID_TYPES, 'Value cannot be converted to JSON.'

  @classmethod
  def set(cls, key, value=True):
    cls._validate(key)
    cls._validate(value)
    cls._conf[key] = value
    return cls._conf[key]

  @classmethod
  def get(cls, key):
    cls._validate(key)
    try:
      v = cls._conf[key]
    except KeyError:
      raise KeyError('\'%s\' not found in configuration file \'%s\'.'%(key,cls._path))
    return v

  @classmethod
  def purge(cls):
    cls._conf = dict()
    cls._path = None

  @classmethod
  def set_test_defaults(cls):
    cls._conf = copy.deepcopy(_TEST_DEFAULTS)
    cls._path = None

  @classmethod
  def set_env_default(cls):
    paths = os.environ['PYTHONPATH'].split(":")
    found_dl_models = False

    for path in paths:
      if 'ares/' in path:
        if not os.path.isdir(path):
          print("[Conf.set_env_default] ERROR: Default ares not found")
          break
        cls.set('dl_models_root', path)

        if not os.path.isdir(path + "/cache"):
          print("[Conf.set_env_default] ERROR: Default ares/cache not found")
          break
        cls.set('cache_dir', path + "/cache")

        if not os.path.isdir(path + "/results"):
          print("[Conf.set_env_default] ERROR: Default ares/results not found")
          break
        cls.set('results_dir', path + "/results")

        found_dl_models = True
        break

    if not found_dl_models:
      print("[Conf.set_env_default] ERROR: Could not find default ares paths ")

  @classmethod
  def load(cls, filename):
    cls._path = filename
    cls._conf = json.load(open(filename))

  @classmethod
  def save(cls, filename):
    json.dump(cls._conf, open(filename,'w'), indent=2)

  @staticmethod
  def find_config(filename):
    '''Looks in common locations for a ares configuration file.

    The location priority is:
      - A provided filename
      - Location given by a "DL_MODELS" environment variable.
      - "ares.conf" in current directory.
      - ".ares.conf" in home directory.
    '''

    files_attempted = []

    v = filename
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.environ.get('DL_MODELS_CONF',None)
    files_attempted.append('$WEIGHTLESS_CONF')
    if v is not None and os.path.isfile(v):
      return v
    v = 'ares.conf'
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v
    v = os.path.expanduser('.ares.conf')
    files_attempted.append(v)
    if v is not None and os.path.isfile(v):
      return v

    assert filename is not None, 'No valid configuration file found. Locations tried:\n'+'\n'.join(['  '+str(v) for v in files_attempted])
