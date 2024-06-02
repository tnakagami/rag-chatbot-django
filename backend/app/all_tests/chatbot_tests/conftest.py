import pytest
import numpy as np
from chatbot.models.agents import AgentType, ToolType

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


@pytest.fixture
def get_agent_types():
  kwargs = {label: val for val, label in AgentType.choices}

  return kwargs

@pytest.fixture
def get_embedding_types():
  kwargs = {label: val for val, label in AgentType.embedding_choices}

  return kwargs

@pytest.fixture
def get_tool_types():
  kwargs = {label: val for val, label in ToolType.choices}

  return kwargs

def ids(value):
  return f'{value}'

@pytest.fixture(params=[label for _, label in AgentType.choices], ids=ids)
def get_target_agent_type(request):
  kwargs = request.getfixturevalue('get_agent_types')
  key = request.param
  val = kwargs[key]

  return key, val

@pytest.fixture(params=[label for _, label in AgentType.embedding_choices], ids=ids)
def get_target_embedding_type(request):
  kwargs = request.getfixturevalue('get_embedding_types')
  key = request.param
  val = kwargs[key]

  return key, val

@pytest.fixture(params=[
  ToolType.RETRIEVER.label,
  ToolType.ACTION_SERVER.label,
  ToolType.CONNERY_ACTION.label,
  ToolType.DALLE_TOOL.label,
  ToolType.KAY_SEC_FILINGS.label,
  ToolType.KAY_PRESS_RELEASES.label,
  ToolType.TAVILY_SEARCH.label,
  ToolType.TAVILY_ANSWER.label,
  ToolType.YOU_SEARCH.label,
], ids=ids)
def get_target_tool_type_with_config(request):
  kwargs = request.getfixturevalue('get_tool_types')
  key = request.param
  val = kwargs[key]

  return key, val

@pytest.fixture(params=[
  ToolType.ARXIV.label,
  ToolType.DDG_SEARCH.label,
  ToolType.PUBMED.label,
  ToolType.WIKIPEDIA.label,
], ids=ids)
def get_target_tool_type_without_config(request):
  kwargs = request.getfixturevalue('get_tool_types')
  key = request.param
  val = kwargs[key]

  return key, val