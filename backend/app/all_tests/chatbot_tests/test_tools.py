import pytest
from chatbot.models.utils import tools
# Classes for comparing instance
from langchain_core.tools import Tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import (
    TavilyAnswer as _TavilyAnswer,
    TavilySearchResults,
)

class DummyToolkit:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs
  @classmethod
  def create_instance(cls, *args, **kwargs):
    return cls(*args, **kwargs)
  def get_tools(self, *args, **kwargs):
    return []

@pytest.fixture
def get_api_key():
  kwargs = {
    'api_key': 'sample',
  }

  return kwargs

@pytest.fixture
def get_apikey_and_url(get_api_key):
  specific = {
    'url': 'http://running-server.com/config',
  }
  kwargs = get_api_key
  kwargs.update(specific)

  return kwargs

@pytest.fixture
def get_dall_e_without_proxy(get_api_key):
  specific = {
    'model': 'sample-model',
    'endpoint': 'http://endpoint.com/base',
  }
  kwargs = get_api_key
  kwargs.update(specific)

  return kwargs

@pytest.fixture
def get_dall_e_with_proxy(get_dall_e_without_proxy):
  specific = {
    'proxy': 'http://proxy.com:12345/target',
  }
  kwargs = get_dall_e_without_proxy
  kwargs.update(specific)

  return kwargs

@pytest.fixture
def get_target_tool():
  def inner(instance):
    try:
      tool = instance.get_tools()
    except Exception:
      pytest.fail('Raise Exception when the instance method `get_tools` is called.')

    return tool

  return inner

# ===============
# = Config Test =
# ===============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_apikey_config(get_api_key):
  kwargs = get_api_key
  config = tools._ApiKeyConfig(**kwargs)

  assert str(config.api_key) == kwargs['api_key']

@pytest.mark.chatbot
@pytest.mark.util
def test_check_action_server_config(get_apikey_and_url):
  kwargs = get_apikey_and_url
  config = tools._ActionServerConfig(**kwargs)

  assert str(config.api_key) == kwargs['api_key']
  assert str(config.url) == kwargs['url']

@pytest.mark.chatbot
@pytest.mark.util
def test_check_connery_config(get_apikey_and_url):
  kwargs = get_apikey_and_url
  config = tools._ConneryConfig(**kwargs)

  assert str(config.api_key) == kwargs['api_key']
  assert str(config.url) == kwargs['url']

@pytest.mark.chatbot
@pytest.mark.util
def test_check_dall_e_config_without_proxy(get_dall_e_without_proxy):
  kwargs = get_dall_e_without_proxy
  config = tools._DallEConfig(**kwargs)

  assert str(config.api_key) == kwargs['api_key']
  assert str(config.model) == kwargs['model']
  assert str(config.endpoint) == kwargs['endpoint']
  assert config.proxy is None

@pytest.mark.chatbot
@pytest.mark.util
def test_check_dall_e_config_with_proxy(get_dall_e_with_proxy):
  kwargs = get_dall_e_with_proxy
  config = tools._DallEConfig(**kwargs)

  assert str(config.api_key) == kwargs['api_key']
  assert str(config.model) == kwargs['model']
  assert str(config.endpoint) == kwargs['endpoint']
  assert str(config.proxy) == kwargs['proxy']

# =============
# = _BaseTool =
# =============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_basetool_without_config():
  instance = tools._BaseTool(config=None)
  config = instance.get_config_fields()

  with pytest.raises(NotImplementedError):
    instance.get_tools()
  assert config is None
  assert not str(instance.name)
  assert not str(instance.description)
  assert not instance.multi_use

@pytest.mark.chatbot
@pytest.mark.util
def test_check_basetool_with_config(get_dall_e_with_proxy):
  kwargs = get_dall_e_with_proxy
  instance = tools._BaseTool(config=kwargs)
  config = instance.get_config_fields()

  assert config is None

# =================
# = RetrievalTool =
# =================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_retrieval_tool(get_target_tool):
  getter = get_target_tool
  instance = tools.RetrievalTool()
  config = instance.get_config_fields()
  callback = getter(instance)
  target = callback(None)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config is None
  assert callable(callback)
  assert isinstance(target, Tool)

# ====================
# = ActionServerTool =
# ====================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_action_server_tool(get_apikey_and_url, get_target_tool, mocker):
  mocker.patch('chatbot.models.utils.tools.ActionServerToolkit', new=DummyToolkit)
  kwargs = get_apikey_and_url
  getter = get_target_tool
  instance = tools.ActionServerTool(kwargs)
  config = instance.get_config_fields()
  _ = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert instance.multi_use
  assert config == kwargs

# =============
# = ArxivTool =
# =============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_arxiv_tool(get_target_tool):
  getter = get_target_tool
  instance = tools.ArxivTool()
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config is None
  assert isinstance(target, ArxivQueryRun)

# ===============
# = ConneryTool =
# ===============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_connery_tool(get_apikey_and_url, get_target_tool, mocker):
  mocker.patch('chatbot.models.utils.tools.ConneryToolkit', new=DummyToolkit)
  kwargs = get_apikey_and_url
  getter = get_target_tool
  instance = tools.ConneryTool(kwargs)
  config = instance.get_config_fields()
  _ = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs

# =============
# = DallETool =
# =============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_dalle_tool(get_dall_e_with_proxy, get_target_tool):
  kwargs = get_dall_e_with_proxy
  getter = get_target_tool
  instance = tools.DallETool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, Tool)

# =================
# = DDGSearchTool =
# =================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_ddg_search_tool(get_target_tool):
  getter = get_target_tool
  instance = tools.DDGSearchTool()
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config is None
  assert isinstance(target, DuckDuckGoSearchRun)

# ==================
# = SecFilingsTool =
# ==================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_sec_filings_tool(get_api_key, get_target_tool):
  kwargs = get_api_key
  getter = get_target_tool
  instance = tools.SecFilingsTool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, Tool)

# =====================
# = PressReleasesTool =
# =====================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_press_release_tool(get_api_key, get_target_tool):
  kwargs = get_api_key
  getter = get_target_tool
  instance = tools.PressReleasesTool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, Tool)

# ==============
# = PubMedTool =
# ==============
@pytest.mark.chatbot
@pytest.mark.util
def test_check_pubmed_tool(get_target_tool):
  getter = get_target_tool
  instance = tools.PubMedTool()
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config is None
  assert isinstance(target, Tool)

# ====================
# = TavilySearchTool =
# ====================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_tavily_search_tool(get_api_key, get_target_tool):
  kwargs = get_api_key
  getter = get_target_tool
  instance = tools.TavilySearchTool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, TavilySearchResults)

# ====================
# = TavilyAnswerTool =
# ====================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_tavily_answer_tool(get_api_key, get_target_tool):
  kwargs = get_api_key
  getter = get_target_tool
  instance = tools.TavilyAnswerTool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, _TavilyAnswer)

# =================
# = YouSearchTool =
# =================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_you_search_tool(get_api_key, get_target_tool):
  kwargs = get_api_key
  getter = get_target_tool
  instance = tools.YouSearchTool(kwargs)
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config == kwargs
  assert isinstance(target, Tool)

# =================
# = WikipediaTool =
# =================
@pytest.mark.chatbot
@pytest.mark.util
def test_check_wikipedia_tool(get_target_tool):
  getter = get_target_tool
  instance = tools.WikipediaTool()
  config = instance.get_config_fields()
  target = getter(instance)

  assert str(instance.name)
  assert str(instance.description)
  assert not instance.multi_use
  assert config is None
  assert isinstance(target, Tool)
