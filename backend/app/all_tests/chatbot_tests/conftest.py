import pytest
import numpy as np

@pytest.fixture
def client_non_proxy_checker():
  def inner(client_without_proxy):
    transport = list(client_without_proxy._mounts.values())

    return len(transport) == 0

  return inner

@pytest.fixture
def client_proxy_checker():
  def inner(client_with_proxy, expected):
    transport = list(client_with_proxy._mounts.values())[0]
    _url = transport._pool._proxy_url

    return all([
      _url.scheme == expected['scheme'],
      _url.host == expected['host'],
      _url.port == expected['port'],
      _url.target == expected['target'],
    ])

  return inner

@pytest.fixture
def check_fields():
  def inner(fields, expected, ignores=None):
    if ignores is None:
      ignores = []

    dict_data = dict([_field.astuple() for _field in fields])
    key_name = 'proxy'

    if key_name in expected.keys():
      valid_proxy = dict_data[key_name] == expected[key_name]
    else:
      valid_proxy = dict_data.get(key_name, None) is None
    # Delete proxy
    dict_data.pop(key_name, None)

    return all([
      all([key not in dict_data.keys() for key in ignores]),
      valid_proxy,
      all([value == expected[name] for name, value in dict_data.items()]),
    ])

  return inner

@pytest.fixture
def get_normalizer():
  def inner(arr, axis=0):
    return arr / np.linalg.norm(arr, axis=axis)

  return inner