import pytest

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
def check_llm_fields():
  def inner(llm_fields, expected):
    use_proxy = 'proxy' in expected.keys()

    return all([
      llm_fields['model'] == expected['model'],
      llm_fields['temperature'] == expected['temperature'],
      llm_fields['stream'] == expected['stream'],
      llm_fields['max_retries'] == expected['max_retries'],
      llm_fields['proxy'] == expected['proxy'] if use_proxy else llm_fields['proxy'] is None,
    ])

  return inner

@pytest.fixture
def check_embedding_fields():
  def inner(embedinng_fields, expected):
    use_proxy = 'proxy' in expected.keys()
    ignore_keys = ['temperature', 'stream']

    return all([
      all([ignore_key not in embedinng_fields.keys() for ignore_key in ignore_keys]),
      embedinng_fields['model'] == expected['model'],
      embedinng_fields['max_retries'] == expected['max_retries'],
      embedinng_fields['proxy'] == expected['proxy'] if use_proxy else embedinng_fields['proxy'] is None,
    ])

  return inner