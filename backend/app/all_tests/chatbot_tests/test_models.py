import pytest
from chatbot.models.utils.ingest import IngestBlobRunnable
from chatbot.models.agents import AgentArgs, ToolArgs, AgentType, ToolType
from chatbot.models.rag import (
  convert_timezone,
  BaseConfig,
  Agent,
  Embedding,
  Tool,
  Assistant,
  DocumentFile,
  Thread,
  EmbeddingStore,
  LangGraphCheckpoint,
)
# For test
import numpy as np
import chatbot.models.utils.llms as llms
import chatbot.models.utils.executors as executors
import chatbot.models.utils.tools as tools
from langchain_community.document_loaders import Blob
from chatbot.models.utils.vectorstore import DistanceStrategy, CustomVectorStore
from celery import states
from asgiref.sync import sync_to_async
from django.db import NotSupportedError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from django.utils.timezone import make_aware
from django_celery_results.models import TaskResult
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from . import factories

class PseudoLLM:
  def __init__(self, is_embedded, name, *args, **kwargs):
    self.is_embedded = is_embedded
    self.llm_name = name

class PseudoApp:
  def __init__(self, name):
    self.app_name = name

class PseudoTool:
  def __init__(self, name, *args, **kwargs):
    self.tool_name = name

class DummyLLMWrapper:
  def __init__(self, name, *args, **kwargs):
    self.llm_name = name
    self.config = {'name': self.llm_name, 'type': 'llm'}
  def get_llm(self, is_embedded=False):
    return PseudoLLM(is_embedded, self.llm_name)
  def delete_keys(self, target, ignore_keys):
    return self.config
  def get_fields(self, is_embedded=False):
    return self.config

class DummyToolExecutorWrapper:
  def __init__(self, name, *args, **kwargs):
    self.executor_name = name
    self.config = {'name': self.executor_name, 'type': 'executor'}
  def get_messages(self, messages, system_message):
    return self.executor_name
  def should_continue(self, messages):
    return self.executor_name
  def get_app(self, system_message: str):
    return PseudoApp(f'tool-{self.executor_name}')

class DummyXmlExecutorWrapper(DummyToolExecutorWrapper):
  def get_app(self, system_message: str):
    return PseudoApp(f'xml-{self.executor_name}')

class DummyToolInstanceWrapper:
  def __init__(self, name, *args, **kwargs):
    self.tool_name = name
    self.config = {'name': self.tool_name, 'type': 'tool'}
  def get_config_fields(self):
    return self.config
  def get_tools(self):
    return PseudoTool(self.tool_name)

class DummyToolListWrapper(DummyToolInstanceWrapper):
  def get_tools(self):
    return [PseudoTool(self.tool_name), PseudoTool(self.tool_name)]

# ====================
# = Global functions =
# ====================
@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'this_timezone',
  'target',
  'is_string',
  'expected',
], [
  ('UTC', datetime(2020,1,2,10,0,0, tzinfo=timezone.utc), False, datetime(2020,1,2,10,0,0, tzinfo=ZoneInfo('UTC'))),
  ('UTC', datetime(2020,1,2,10,0,0, tzinfo=timezone.utc), True, '2020-01-02 10:00:00.000000'),
  ('Asia/Tokyo', datetime(2020,1,2,10,0,0, tzinfo=timezone.utc), False, datetime(2020,1,2,19,0,0, tzinfo=ZoneInfo('Asia/Tokyo'))),
  ('Asia/Tokyo', datetime(2020,1,2,10,0,0, tzinfo=timezone.utc), True, '2020-01-02 19:00:00.000000'),
], ids=['to-utc-datetime', 'to-utc-string', 'to-asia-tokyo-datetime', 'to-asia-tokyo-string'])
def test_check_convert_timezone_function(settings, this_timezone, target, is_string, expected):
  settings.TIME_ZONE = this_timezone
  output = convert_timezone(target, is_string=is_string)

  assert output == expected

# ==========
# = Agents =
# ==========
@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'tool_type',
  'tool_name',
  'dummy_class',
], [
  (ToolType.RETRIEVER, 'retrieval', DummyToolInstanceWrapper),
  (ToolType.ACTION_SERVER, 'action-server', DummyToolListWrapper),
  (ToolType.ARXIV, 'arxiv', DummyToolInstanceWrapper),
  (ToolType.CONNERY_ACTION, 'connery', DummyToolListWrapper),
  (ToolType.DALLE_TOOL, 'dalle', DummyToolInstanceWrapper),
  (ToolType.DDG_SEARCH, 'ddg-search', DummyToolInstanceWrapper),
  (ToolType.KAY_SEC_FILINGS, 'sec-filings', DummyToolInstanceWrapper),
  (ToolType.KAY_PRESS_RELEASES, 'press-release', DummyToolInstanceWrapper),
  (ToolType.PUBMED, 'pubmed', DummyToolInstanceWrapper),
  (ToolType.TAVILY_SEARCH, 'tavily-search', DummyToolInstanceWrapper),
  (ToolType.TAVILY_ANSWER, 'tavily-answer', DummyToolInstanceWrapper),
  (ToolType.YOU_SEARCH, 'you-search', DummyToolInstanceWrapper),
  (ToolType.WIKIPEDIA, 'wikipedia', DummyToolInstanceWrapper),
], ids=[
  'tool-type-retrieval',
  'tool-type-action-server',
  'tool-type-arxiv',
  'tool-type-connery',
  'tool-type-dalle',
  'tool-type-ddg-search',
  'tool-type-sec-filings',
  'tool-type-press-release',
  'tool-type-pubmed',
  'tool-type-tavily-search',
  'tool-type-tavily-answer',
  'tool-type-you-search',
  'tool-type-wikipedia',
])
def test_check_tooltype(mocker, tool_type, tool_name, dummy_class):
  class DummyEmbeddings:
    def __init__(self):
      pass
    def get_distance_strategy(self):
      return object()
    def get_embedding(self):
      return object()

  args = ToolArgs(
    assistant_id=1,
    manager=object(),
    embedding=DummyEmbeddings(),
  )
  mocker.patch(
    'chatbot.models.agents.ToolType._tool_type',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: dummy_class(name=tool_name),
  )
  tool_id = tool_type.value
  instance = ToolType(tool_id)
  config_field = ToolType.get_config_field(tool_id, {})
  targets = ToolType.get_tool(tool_id, {}, args)
  _tool = instance._tool_type({})
  expected_config = {'name': tool_name, 'type': 'tool'}

  assert str(instance) == tool_type.label
  assert config_field == expected_config
  assert _tool.tool_name == tool_name
  assert targets[0].tool_name == tool_name if isinstance(targets, list) else targets.tool_name == tool_name

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'tool_type',
  'expected_tool',
], [
  (ToolType.RETRIEVER, tools.RetrievalTool),
  (ToolType.ACTION_SERVER, tools.ActionServerTool),
  (ToolType.ARXIV, tools.ArxivTool),
  (ToolType.CONNERY_ACTION, tools.ConneryTool),
  (ToolType.DALLE_TOOL, tools.DallETool),
  (ToolType.DDG_SEARCH, tools.DDGSearchTool),
  (ToolType.KAY_SEC_FILINGS, tools.SecFilingsTool),
  (ToolType.KAY_PRESS_RELEASES, tools.PressReleasesTool),
  (ToolType.PUBMED, tools.PubMedTool),
  (ToolType.TAVILY_SEARCH, tools.TavilySearchTool),
  (ToolType.TAVILY_ANSWER, tools.TavilyAnswerTool),
  (ToolType.YOU_SEARCH, tools.YouSearchTool),
  (ToolType.WIKIPEDIA, tools.WikipediaTool),
], ids=[
  'retrieval-tool',
  'action-server-tool',
  'arxiv-tool',
  'connery-tool',
  'dall-e-tool',
  'ddg-search-tool',
  'sec-filings-tool',
  'press-releases-tool',
  'pubmed-tool',
  'tavily-search-tool',
  'tavily-answer-tool',
  'you-search-tool',
  'wikipedia-tool',
])
def test_check_property_of_tooltype(tool_type, expected_tool):
  tool_id = tool_type.value
  instance = ToolType(tool_id)

  assert type(instance._tool_type) == type(expected_tool)

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'agent_type',
  'agent_name',
  'app_prefix',
  'dummy_llm_type',
  'dummy_executor_type',
], [
  (AgentType.OPENAI, 'openai', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
  (AgentType.AZURE, 'azure', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
  (AgentType.ANTHROPIC, 'anthropic', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
  (AgentType.BEDROCK, 'bedrock', 'xml', DummyLLMWrapper, DummyXmlExecutorWrapper),
  (AgentType.FIREWORKS, 'fireworks', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
  (AgentType.OLLAMA, 'ollama', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
  (AgentType.GEMINI, 'gemini', 'tool', DummyLLMWrapper, DummyToolExecutorWrapper),
], ids=[
  'openai',
  'azure',
  'anthropic',
  'bedrock',
  'fireworks',
  'ollama',
  'gemini',
])
def test_check_agenttype(mocker, agent_type, agent_name, app_prefix, dummy_llm_type, dummy_executor_type):
  args = AgentArgs(
    tools=DummyToolInstanceWrapper('dummy'),
    checkpoint=None,
    system_message='system',
    is_interrupt=False,
  )
  mocker.patch(
    'chatbot.models.agents.AgentType._llm_type',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: dummy_llm_type(name=agent_name),
  )
  mocker.patch(
    'chatbot.models.agents.AgentType._executor_type',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: dummy_executor_type(name=agent_name),
  )
  agent_id = agent_type.value
  instance = AgentType(agent_id)
  embedded_config_field = AgentType.get_llm_fields(agent_id, config={}, is_embedded=True)
  llm_config_field = AgentType.get_llm_fields(agent_id, config={}, is_embedded=False)
  app = AgentType.get_executor(agent_id, {}, args)
  embedded = AgentType.get_embedding(agent_id, {})
  _llm_wrapper = instance._llm_type({})
  llm = _llm_wrapper.get_llm(is_embedded=False)
  executor = instance._executor_type(llm, args.tools, args.is_interrupt, args.checkpoint)
  expected_llm_config = {'name': agent_name, 'type': 'llm'}
  expected_executor_config = {'name': agent_name, 'type': 'executor'}

  assert str(instance) == agent_type.label
  assert embedded_config_field == expected_llm_config
  assert llm_config_field == llm_config_field
  assert _llm_wrapper.llm_name == agent_name
  assert not llm.is_embedded and llm.llm_name == agent_name
  assert embedded.is_embedded and embedded.llm_name == agent_name
  assert app.app_name == f'{app_prefix}-{agent_name}'
  assert executor.executor_name == agent_name

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'agent_type',
  'expected_llm',
  'expected_executor',
], [
  (AgentType.OPENAI, llms.OpenAILLM, executors.ToolExecutor),
  (AgentType.AZURE, llms.AzureOpenAILLM, executors.ToolExecutor),
  (AgentType.ANTHROPIC, llms.AnthropicLLM, executors.ToolExecutor),
  (AgentType.BEDROCK, llms.BedrockLLM, executors.XmlExecutor),
  (AgentType.FIREWORKS, llms.FireworksLLM, executors.ToolExecutor),
  (AgentType.OLLAMA, llms.OllamaLLM, executors.ToolExecutor),
  (AgentType.GEMINI, llms.GeminiLLM, executors.ToolExecutor),
], ids=[
  'openai-llm',
  'azure-openai-llm',
  'anthropic-llm',
  'bedrock-llm',
  'fireworks-llm',
  'ollama-llm',
  'gemini-llm',
])
def test_check_property_of_agenttype(agent_type, expected_llm, expected_executor):
  agent_id = agent_type.value
  instance = AgentType(agent_id)

  assert type(instance._llm_type) == type(expected_llm)
  assert type(instance._executor_type) == type(expected_executor)

@pytest.mark.chatbot
@pytest.mark.model
def test_check_specific_method_of_agenttype():
  choices = AgentType.embedding_choices

  assert AgentType.ANTHROPIC not in choices

# ==========
# = Ingest =
# ==========
@pytest.mark.chatbot
@pytest.mark.model
def test_check_convert_input2blob_of_ingest_class(mocker):
  mocker.patch('chatbot.models.utils.ingest.IngestBlobRunnable._guess_mimetype', return_value='text/plain')
  file_field = SimpleUploadedFile(
    'test-file.txt',
    b'This is a sample file',
  )
  runnable = IngestBlobRunnable(store=CustomVectorStore(None, None, None), record_id=0)
  blob = runnable.convert_input2blob(file_field)
  out = blob.as_string()

  assert out == 'This is a sample file'

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize('bytedata,expected_mimetype', [
  (b'%PDF-2.0%%EOF', 'application/pdf'),
  (b'\x50\x4B\x03\x04\x20\x58\x4d\x4c', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
  (b'\x50\x4B\x05\x06\x4f\x50\x45\x4e', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
  (b'\x50\x4B\x07\x08\x20\x44\x4f\x43', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
  (b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x20\x57\x6f\x72\x64', 'application/msword'),
  (b'\x09\x00\xff\x00\x06\x00\x20\x45\x78\x63\x65\x6c', 'application/vnd.ms-excel'),
  (b'\x61\x2c\x62\x2c\x63\x0a\x78\x2c\x79\x2c\x7a', 'text/csv'),
  (b'\x61\x09\x62\x09\x63\x0a\x78\x09\x79\x09\x7a', 'text/csv'),
  (b'This is a sample text', 'text/plain'),
  ( ('a'*1024).encode(), 'text/plain'),
  ( ('b'*1025).encode(), 'text/plain'),
], ids=[
  'is-pdf',
  'is-openxml',
  'is-openoffice',
  'is-wordpress',
  'is-word',
  'is-excel',
  'is-csv-comma',
  'is-csv-tab',
  'is-plain-text-simple',
  'is-plain-text-length-1024',
  'is-plain-text-length-1025',
])
def test_check_guess_mimetype_method_of_ingest_class(mocker, bytedata, expected_mimetype):
  no_name = ''
  mocker.patch('chatbot.models.utils.ingest.mimetypes.guess_type', return_value=(False, ''))
  runnable = IngestBlobRunnable(store=CustomVectorStore(None, None, None), record_id=0)
  mime_type = runnable._guess_mimetype(no_name, bytedata)

  assert mime_type == expected_mimetype

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize('is_printable,is_eq,expected', [
  (True, False , 'text/plain'),
  (False, True, 'text/plain'),
  (False, False, 'application/octet-stream'),
], ids=['is-printable', 'decoded-value-is-empty', 'is-streaming-data'])
def test_check_customized_pattern_of_ingest_class(mocker, is_printable, is_eq, expected):
  class PseudoBytes:
    def __init__(self, val, is_printable, is_eq, *args, **kwargs):
      self._data = val
      self._printable = is_printable
      self._eq = is_eq

    def __eq__(self, other):
      return self._eq

    def __len__(self):
      return len(self._data)

    def __getitem__(self, key):
      return PseudoBytes(self._data[key], self._printable, self._eq)

    def __contains__(self, item):
      return False

    def decode(self, *args, **kwargs):
      return PseudoBytes(self._data, self._printable, self._eq)

    def isprintable(self):
      return self._printable

    def startswith(self, *args, **kwargs):
      return self._data.encode().startswith(*args, **kwargs)

  bytedata = PseudoBytes('sample', is_printable, is_eq)
  runnable = IngestBlobRunnable(store=CustomVectorStore(None, None, None), record_id=0)
  mime_type = runnable._guess_mimetype('', bytedata)

  assert mime_type == expected

@pytest.mark.chatbot
@pytest.mark.model
def test_check_specific_pattern_of_ingest_class():
  class CannotDecodeData:
    def __init__(self, val, *args, **kwargs):
      self._byte_val = val

    def __len__(self):
      return len(self._byte_val)

    def __getitem__(self, key):
      return CannotDecodeData(self._byte_val[key])

    def decode(self, *args, **kwargs):
      raise UnicodeDecodeError('utf-8', b'dummy-data', 0, 1, 'error')

    def startswith(self, *args, **kwargs):
      return self._byte_val.startswith(*args, **kwargs)

  runnable = IngestBlobRunnable(store=CustomVectorStore(None, None, None), record_id=0)
  # In the case of the UnicodeDecodeError is raised
  out = runnable._guess_mimetype('no-name', CannotDecodeData(b'sample'))

  assert out == 'application/octet-stream'

@pytest.mark.chatbot
@pytest.mark.model
def test_check_be_able_to_estimate_mimetypes_of_ingest_class(mocker):
  runnable = IngestBlobRunnable(store=CustomVectorStore(None, None, None), record_id=0)
  # In the case of being able to estimate mimetype
  mocker.patch('chatbot.models.utils.ingest.mimetypes.guess_type', return_value=('text/plain', ''))
  out = runnable._guess_mimetype('no-name', b'This is a dummy data')

  assert out == 'text/plain'

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize([
  'packed_num',
  'batch_size',
  'expected_call',
], [
  (None, 2, 0),
  ([1], 2, 1),
  ([2, 1], 2, 2),
], ids=['split-none', 'split-once', 'split-twice'])
def test_check_invoke_method_of_ingest_class(mocker, packed_num, batch_size, expected_call):
  class DummyDoc:
    def __init__(self, data):
      self.page_content = data

  if packed_num is None:
    targets = []
  else:
    targets = [[DummyDoc('a') for _ in range(val)] for val in packed_num]
  store = CustomVectorStore(None, None, None)
  runnable = IngestBlobRunnable(store=store, record_id=0, batch_size=batch_size)
  mocker.patch.object(
    runnable.parser,
    'lazy_parse',
    new_callable=mocker.PropertyMock,
    return_value=[arr[0] for arr in targets],
  )
  mocker.patch.object(
    runnable.splitter,
    'split_documents',
    side_effect=targets,
  )
  property_mock = mocker.patch.object(
    store,
    'add_documents',
    new_callable=mocker.PropertyMock,
    return_value=[1],
  )
  blob = Blob.from_data(
    data=b'test',
    path=None,
    mime_type='text/plain',
  )
  outputs = runnable.invoke(blob)

  assert property_mock.call_count == expected_call

# =======
# = RAG =
# =======
@pytest.fixture
def get_common_data():
  name = 'sample'
  config = {'target': 2}

  return name, config

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_baseconfig(get_common_data):
  name, config = get_common_data
  instance = BaseConfig.objects.create(name=name, config=config)
  target = instance.get_config()
  collected = BaseConfig.objects.get_or_none(pk=instance.pk)
  doesnot_exist = BaseConfig.objects.get_or_none(pk=instance.pk+1)
  default_type_id, default_config = BaseConfig.get_config_form_args()

  assert instance.name == name
  assert instance.config == config
  assert default_type_id == AgentType.OPENAI
  assert target == config
  assert all([isinstance(default_config, dict), len(default_config) == 0])
  assert collected is not None
  assert doesnot_exist is None

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_aget_or_none_method(get_common_data):
  name, config = get_common_data
  instance = await BaseConfig.objects.acreate(name=name, config=config)
  target = await sync_to_async(instance.get_config)()
  collected = await BaseConfig.objects.aget_or_none(pk=instance.pk)
  doesnot_exist = await BaseConfig.objects.aget_or_none(pk=instance.pk+1)
  default_type_id, default_config = await sync_to_async(BaseConfig.get_config_form_args)()

  assert instance.name == name
  assert instance.config == config
  assert default_type_id == AgentType.OPENAI
  assert target == config
  assert all([isinstance(default_config, dict), len(default_config) == 0])
  assert collected is not None
  assert doesnot_exist is None

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_agent(mocker, get_common_data):
  name, config = get_common_data
  ret_val = 3
  mocker.patch(
    'chatbot.models.agents.AgentType.get_executor',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: ret_val,
  )
  args = AgentArgs(tools=[], checkpoint=None)
  _agent_type = AgentType.AZURE
  user = factories.UserFactory()
  default_instance = Agent.objects.create(user=user, name='test', config={})
  specific_agent = factories.AgentFactory(
    name=name,
    config=config,
    user=user,
    agent_type=_agent_type,
  )
  random_agent = factories.AgentFactory()
  spec_val = specific_agent.get_executor(args)
  rand_val = random_agent.get_executor(args)
  _exists = Agent.objects.get_or_none(pk=default_instance.pk)
  default_type_id, default_config = Agent.get_config_form_args()
  instance_type_id, instance_config = Agent.get_config_form_args(instance=specific_agent)

  assert default_instance.agent_type == AgentType.OPENAI.value
  assert str(specific_agent) == f'{name} ({AgentType.AZURE.label})'
  assert default_type_id == AgentType.OPENAI
  assert all([isinstance(default_config, dict), len(default_config) == 0])
  assert instance_type_id == _agent_type
  assert instance_config == config
  assert specific_agent.user.pk == user.pk
  assert specific_agent.name == name
  assert specific_agent.config == config
  assert spec_val == ret_val
  assert rand_val == ret_val
  assert _exists is not None

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_embedding(mocker, get_common_data):
  name, config = get_common_data
  ret_val = 3
  mocker.patch(
    'chatbot.models.agents.AgentType.get_embedding',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: ret_val,
  )
  user = factories.UserFactory()
  _emb_type = AgentType.AZURE
  default_instance = Embedding.objects.create(user=user, name='test', config={})
  specific_embedding = factories.EmbeddingFactory(
    name=name,
    config=config,
    user=user,
    distance_strategy=Embedding.DistanceType.EUCLIDEAN,
    emb_type=_emb_type,
  )
  random_embedding = factories.EmbeddingFactory()
  default_ds = default_instance.get_distance_strategy()
  specific_ds = specific_embedding.get_distance_strategy()
  spec_val = specific_embedding.get_embedding()
  rand_val = random_embedding.get_embedding()
  _exists = Embedding.objects.get_or_none(pk=default_instance.pk)
  default_type_id, default_config = Embedding.get_config_form_args()
  instance_type_id, instance_config = Embedding.get_config_form_args(instance=specific_embedding)

  assert default_instance.emb_type == AgentType.OPENAI.value
  assert default_instance.distance_strategy == Embedding.DistanceType.COSINE.value
  assert str(specific_embedding) == f'{name} ({AgentType.AZURE.label})'
  assert default_type_id == AgentType.OPENAI
  assert all([isinstance(default_config, dict), len(default_config) == 0])
  assert instance_type_id == _emb_type
  assert instance_config == config
  assert default_ds == DistanceStrategy.COSINE
  assert specific_ds == DistanceStrategy.EUCLIDEAN
  assert spec_val == ret_val
  assert rand_val == ret_val
  assert _exists is not None

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_tool(mocker, get_common_data):
  name, config = get_common_data
  ret_val = 3
  mocker.patch(
    'chatbot.models.agents.ToolType.get_tool',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: ret_val,
  )
  args = ToolArgs(assistant_id=0, manager=None, embedding=None)
  user = factories.UserFactory()
  _tool_type = ToolType.DDG_SEARCH
  default_instance = Tool.objects.create(user=user, name='test', config={})
  specific_tool = factories.ToolFactory(
    name=name,
    config=config,
    user=user,
    tool_type=_tool_type,
  )
  random_tool = factories.ToolFactory()
  spec_val = specific_tool.get_tool(args)
  rand_val = random_tool.get_tool(args)
  _exists = Tool.objects.get_or_none(pk=default_instance.pk)
  default_type_id, default_config = Tool.get_config_form_args()
  instance_type_id, instance_config = Tool.get_config_form_args(instance=specific_tool)

  assert default_instance.tool_type == ToolType.RETRIEVER.value
  assert str(specific_tool) == f'{name} ({ToolType.DDG_SEARCH.label})'
  assert default_type_id == ToolType.RETRIEVER
  assert all([isinstance(default_config, dict), len(default_config) == 0])
  assert instance_type_id == _tool_type
  assert instance_config == config
  assert spec_val == ret_val
  assert rand_val == ret_val
  assert _exists is not None

# =================
# = Django Models =
# =================
@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('handler', [
  lambda user: factories.AgentFactory(user=user),
  lambda user: factories.EmbeddingFactory(user=user),
  lambda user: factories.ToolFactory(user=user),
  lambda user: factories.AssistantFactory(user=user),
  lambda user: factories.DocumentFileFactory(assistant=factories.AssistantFactory(user=user)),
  lambda user: factories.ThreadFactory(assistant=factories.AssistantFactory(user=user)),
], ids=['agent-owner', 'embedding-owner', 'tool-owner', 'assistant-owner', 'docfile-owner', 'thread-owner'])
def test_check_owner(handler):
  owner, other_user = factories.UserFactory.create_batch(2)
  own_instance = handler(owner)

  assert own_instance.is_owner(owner)
  assert not own_instance.is_owner(other_user)

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('factory_class,kwargs,expected', [
  (factories.AgentFactory,     {'name':  'a'*9, 'agent_type': AgentType.ANTHROPIC}, '{} ({})'.format('a'*9, 'Anthropic (Claude 2)')),
  (factories.AgentFactory,     {'name': 'a'*10, 'agent_type': AgentType.ANTHROPIC}, '{} (Anthropic (Claude 2)...'.format('a'*10)),
  (factories.EmbeddingFactory, {'name': 'b'*15, 'emb_type':   AgentType.BEDROCK},   '{} ({})'.format('b'*15, 'Amazon Bedrock')),
  (factories.EmbeddingFactory, {'name': 'b'*16, 'emb_type':   AgentType.BEDROCK},   '{} (Amazon Bedrock...'.format('b'*16)),
  (factories.ToolFactory,      {'name': 'c'*20, 'tool_type':  ToolType.RETRIEVER},  '{} ({})'.format('c'*20, 'Retriever')),
  (factories.ToolFactory,      {'name': 'c'*21, 'tool_type':  ToolType.RETRIEVER},  '{} (Retriever...'.format('c'*21)),
], ids=[
  'agent-name-length-is-less-than-or-equal-32',
  'agent-name-length-is-more-than-32',
  'embedding-name-length-is-less-than-or-equal-32',
  'embedding-name-length-is-more-than-32',
  'tool-name-length-is-less-than-or-equal-32',
  'tool-name-length-is-more-than-32',
])
def test_check_get_shortname_method(factory_class, kwargs, expected):
  _targets = factory_class(**kwargs)

  assert _targets.get_shortname() == expected

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('num_tools', [0, 1, 2], ids=['no-tools', 'use-only-one-tool', 'use-multi-tools'])
def test_check_assistant(mocker, num_tools):
  targets = [tools.Tool(name='dummy', func=None, description='test') for _ in range(num_tools)]
  name = 'test'
  mocker.patch(
    'chatbot.models.rag.Agent.get_executor',
    new_callable=mocker.PropertyMock,
    return_value=lambda args, *other, **kwargs: args.tools,
  )
  mocker.patch(
    'chatbot.models.rag.Tool.get_tool',
    new_callable=mocker.PropertyMock,
    return_value=lambda *args, **kwargs: targets[0] if num_tools == 1 else targets,
  )
  user = factories.UserFactory()
  _tool_records = factories.ToolFactory.create_batch(num_tools, user=user) if num_tools > 0 else []
  assistant = Assistant.objects.create(
    user=user,
    name=name,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  assistant.tools.add(*_tool_records)
  executor = assistant.get_executor()
  expected_retval = len(_tool_records) * num_tools

  assert str(assistant) == name
  assert 'You are a helpful assistant.' in assistant.system_message
  assert not assistant.is_interrupt
  assert len(executor) == expected_retval

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('docfile_ids,expected_size', [
  (None, 0),
  (2, 0),
  ([1,2,3], 3),
], ids=['no-docfile-ids', 'no-list-of-dcofile-ids', 'docfile-ids-exist'])
def test_check_get_executor_method_of_assistant(mocker, docfile_ids, expected_size):
  mocker.patch(
    'chatbot.models.rag.Agent.get_executor',
    new_callable=mocker.PropertyMock,
    return_value=lambda args, *other, **kwargs: args.tools,
  )
  mocker.patch(
    'chatbot.models.rag.Tool.get_tool',
    new_callable=mocker.PropertyMock,
    return_value=lambda args, *other, **kwargs: args.docfile_ids,
  )
  user = factories.UserFactory()
  _tool_records = factories.ToolFactory(user=user)
  assistant = Assistant.objects.create(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  assistant.tools.add(_tool_records)
  executor = assistant.get_executor(docfile_ids=docfile_ids)

  assert len(executor) == expected_size

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_set_task_result_method_of_assistant():
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  _task = factories.TaskResultFactory()
  task_id = _task.task_id
  assistant.set_task_result(task_id, user.pk)
  instance = TaskResult.objects.get_task(task_id)
  expected = f"{{'info': 'user={user.pk},assistant={assistant.pk}'}}"

  assert instance.task_kwargs == expected

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_collection_with_docfiles_method_of_assistant():
  user = factories.UserFactory()
  specific_assistant = factories.AssistantFactory.create_batch(
    2,
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  other = factories.UserFactory()
  assistants = factories.AssistantFactory.create_batch(
    3,
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
  )
  for _assistant in assistants:
    _ = factories.DocumentFileFactory.create_batch(5, assistant=_assistant)

  only_one_qs = Assistant.objects.collection_with_docfiles(pk=assistants[0].pk)
  qs_of_specific_user = Assistant.objects.collection_with_docfiles(user=user)
  all_qs = Assistant.objects.all()

  assert only_one_qs.count() == 1
  assert qs_of_specific_user.count() == len(specific_assistant)
  assert all_qs.count() == (len(specific_assistant) + len(assistants))

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('task_lists,expected_counts', [
  ([2], [2, 2, 2]),
  ([3, 2], [3, 5, 3]),
  ([4, 1, 5], [4, 10, 4]),
], ids=['single-assistant', 'double-assistants', 'triple-assistants'])
def test_check_collect_own_tasks_method_of_assistant(task_lists, expected_counts):
  # expected_counts: [total tasks for assistant, total tasks for user, total tasks for assistant and user]
  user, other = factories.UserFactory.create_batch(2)
  assistants_for_user = factories.AssistantFactory.create_batch(
    len(task_lists),
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  assistant_for_other = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
  )
  # Create user tasks
  for num, _assistant in zip(task_lists, assistants_for_user):
    for task in factories.TaskResultFactory.create_batch(num):
      _assistant.set_task_result(task.task_id, user.pk)
    task = factories.TaskResultFactory(status=states.SUCCESS)
    _assistant.set_task_result(task.task_id, user.pk)
  # Create other tasks
  for task in factories.TaskResultFactory.create_batch(3):
    assistant_for_other.set_task_result(task.task_id, other.pk)

  total_tasks_for_assistant = Assistant.objects.collect_own_tasks(assistant=assistants_for_user[0]).count()
  total_tasks_for_user = Assistant.objects.collect_own_tasks(user=user).count()
  total_tasks_for_both = Assistant.objects.collect_own_tasks(user=user, assistant=assistants_for_user[0]).count()

  assert total_tasks_for_assistant == expected_counts[0]
  assert total_tasks_for_user == expected_counts[1]
  assert total_tasks_for_both == expected_counts[2]

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_invalid_arguments_for_collect_own_tasks_method():
  with pytest.raises(NotSupportedError):
    _ = Assistant.objects.collect_own_tasks()

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_docfile():
  docfile_name = 'test-document-file'
  assistant_name = 'dummy-assistant'
  docfile = factories.DocumentFileFactory(
    assistant=factories.AssistantFactory(name=assistant_name),
    name=docfile_name,
  )

  assert str(docfile) == f'{docfile_name} ({assistant_name})'

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.parametrize('expected', ['.pdf', '.txt', '.html', '.docx'], ids=lambda val: f'{val[1:]}')
def test_check_valid_extensions(expected):
  valid_extensions = DocumentFile.get_valid_extensions()

  assert expected in valid_extensions

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_collect_own_files_method_of_documentfile():
  user, other = factories.UserFactory.create_batch(2)
  specific_assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  assistants = factories.AssistantFactory.create_batch(
    3,
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
  )
  active_statuses = [False, True, True]
  for _assistant, is_active in zip(assistants, active_statuses):
    _ = factories.DocumentFileFactory.create_batch(5, assistant=_assistant, is_active=is_active)
  _ = factories.DocumentFileFactory(assistant=specific_assistant, is_active=True)
  qs_user = DocumentFile.objects.collect_own_files(user)
  qs_other = DocumentFile.objects.collect_own_files(other)

  assert qs_user.count() == 1
  assert qs_other.count() == (active_statuses.count(True) * 5)

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('count_assistant', [False, True], ids=['count-all-documents', 'count-specific-assistant'])
def test_check_active_method_of_documentfile(count_assistant):
  user, other = factories.UserFactory.create_batch(2)
  assistant_for_user = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  assistant_for_other = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
  )
  active_for_user = [False, True, True, False, False]
  active_for_other = [True, False]

  for is_active in active_for_user:
    _ = factories.DocumentFileFactory(assistant=assistant_for_user, is_active=is_active)
  for is_active in active_for_other:
    _ = factories.DocumentFileFactory(assistant=assistant_for_other, is_active=is_active)

  if count_assistant:
    expected = active_for_user.count(True)
    total = assistant_for_user.docfiles.active().count()
  else:
    expected = active_for_user.count(True) + active_for_other.count(True)
    total = DocumentFile.objects.active().count()

  assert total == expected

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_thread():
  thread_name = 'test-thread'
  assistant_name = 'dummy-assistant'
  thread = factories.ThreadFactory(
    assistant=factories.AssistantFactory(name=assistant_name),
    name=thread_name,
  )

  assert str(thread) == f'{thread_name} ({assistant_name})'

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_get_executor_method_of_thread(mocker):
  expected = 3
  mocker.patch('chatbot.models.rag.Assistant.get_executor', return_value=expected)
  thread = factories.ThreadFactory()
  output = thread.get_executor()

  assert output == expected

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize([
  'num_assistant',
  'max_docfiles_of_each_assistant',
  'each_file_pairs', # this list length is equal the number of threads
  'expected_num_threads',
  'expected_selected_total_docfiles',
], [
  (1, 5, {0: [[0, 2, 3, 4]]}, 1, 4),
  (2, 3, {0: [[0]], 1: [[], [1, 2]]}, 3, 3),
], ids=['single-assistant', 'multi-assitants'])
def test_check_collect_own_threads_method_of_thread(
  num_assistant,
  max_docfiles_of_each_assistant,
  each_file_pairs,
  expected_num_threads,
  expected_selected_total_docfiles,
):
  user, other = factories.UserFactory.create_batch(2)
  # Create assistants
  user_assistants = factories.AssistantFactory.create_batch(
    num_assistant,
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  other_assistant = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
  )
  # Create document files
  user_docfiles = [
    factories.DocumentFileFactory.create_batch(max_docfiles_of_each_assistant, assistant=assistant)
    for assistant in user_assistants
  ]
  other_docfiles = factories.DocumentFileFactory.create_batch(2, assistant=other_assistant)
  # Create threads
  for idx, docfile_list in each_file_pairs.items():
    assistant = user_assistants[idx]
    _local_docfile = user_docfiles[idx]

    for indices in docfile_list:
      target_docfiles = [_file.pk for count, _file in enumerate(_local_docfile) if count in indices]
      thread = factories.ThreadFactory(assistant=assistant)
      thread.docfiles.add(*target_docfiles)

  _, thread = factories.ThreadFactory.create_batch(2, assistant=other_assistant)
  thread.docfiles.add(*[_file.pk for _file in other_docfiles])
  # Check
  targets = Thread.objects.collect_own_threads(user)

  assert targets.count() == expected_num_threads
  assert sum([thread.docfiles.all().count() for thread in targets]) == expected_selected_total_docfiles

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_check_str_method_of_embedding_store():
  screen_name = 'test-user'
  assistant_name = 'dummy'
  assistant = factories.AssistantFactory(
    user=factories.UserFactory(screen_name=screen_name),
    name=assistant_name,
  )
  docfile = factories.DocumentFileFactory(assistant=assistant)
  store = EmbeddingStore.objects.create(
    embedding=[1,2,3],
    document='sample-text',
    assistant_id=assistant.pk,
    docfile_id=docfile.pk,
  )

  assert str(store) == f'{assistant_name} ({screen_name})'

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('set_assistant,set_docfile',[
  (False, True),
  (True, False),
  (False, False),
], ids=['not-set-assistant', 'not-set-document-file', 'not-set-model-ids'])
def test_check_invalid_args_for_create_method_of_embedding_store(set_assistant, set_docfile):
  if set_assistant:
    assistant = factories.AssistantFactory()
    assistant_id = assistant.pk
  else:
    assistant_id = None
  if set_docfile:
    docfile = factories.DocumentFileFactory(assistant=factories.AssistantFactory())
    docfile_id = docfile.pk
  else:
    docfile_id = None

  with pytest.raises(NotSupportedError):
    _ = EmbeddingStore.objects.create(
      embedding=[1,2,3],
      document='sample-text',
      assistant_id=assistant_id,
      docfile_id=docfile_id,
    )

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('distance_strategy,calc_similarity,sort_direction', [
  (DistanceStrategy.EUCLIDEAN, lambda xvec, yvec: np.linalg.norm(xvec - yvec.T, axis=1), 1),
  (DistanceStrategy.COSINE, lambda xvec, yvec: np.dot(xvec, yvec) / np.linalg.norm(xvec, axis=1) / np.linalg.norm(yvec), -1),
  (DistanceStrategy.MAX_INNER_PRODUCT, lambda xvec, yvec: np.dot(xvec, yvec), -1),
], ids=['l2distance', 'cosine-similarity', 'max-inner-product'])
def test_check_similarity_search_method(get_normalizer, distance_strategy, calc_similarity, sort_direction):
  normalizer = get_normalizer
  ndim = 1436
  exact_vector = normalizer(np.linspace(0.1, 10.1, ndim))
  scales = np.array([-2.001, 1.881, 0.999, 0.301, -1.732, 0.123])
  embedding_vectors = normalizer(np.power(exact_vector[:, np.newaxis], scales))
  assistant = factories.AssistantFactory()
  docfile = factories.DocumentFileFactory(assistant=assistant)
  # Create embedding stores
  stores = [
    factories.EmbeddingStoreFactory(assistant=assistant, docfile=docfile, embedding=target_vector, ndim=ndim)
    for target_vector in embedding_vectors.T
  ]
  ids = np.array([store.pk for store in stores])
  # Calculate distance of each vector
  distance = calc_similarity(embedding_vectors.T, exact_vector)
  sorted_indices = np.argsort(sort_direction * distance)
  # Collect expected primary keys
  exact_ids = ids[sorted_indices]
  # Calculate distance using the Django manager's method
  queryset = EmbeddingStore.objects.similarity_search_with_distance_by_vector(exact_vector, distance_strategy, assistant_id=assistant.pk)
  estimated_ids = np.array(queryset.values_list('pk', flat=True))

  assert (exact_ids == estimated_ids).all()

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('distance_strategy', [
  DistanceStrategy.EUCLIDEAN,
  DistanceStrategy.COSINE,
  DistanceStrategy.MAX_INNER_PRODUCT,
], ids=['invalid-pk-of-l2distance', 'invalid-pk-of-cosine-similarity', 'invalid-pk-of-max-inner-product'])
def test_check_no_assistant_id_of_similarity_search_method(distance_strategy):
  empty_query = EmbeddingStore.objects.none()
  queryset = EmbeddingStore.objects.similarity_search_with_distance_by_vector([1,2,3], distance_strategy)

  assert isinstance(queryset, type(empty_query))

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('max_files,selected_num,expected_records',[
  (4, 0, 12),
  (4, 1, 3),
  (4, 2, 6),
  (4, 3, 9),
  (4, 4, 12),
], ids=['set-no-docfiles', 'set-one-docfile', 'set-two-docfiles', 'set-three-docfiles', 'set-all-docfiles'])
def test_check_similarity_search_method_with_docfile_ids(get_normalizer, max_files, selected_num, expected_records):
  normalizer = get_normalizer
  distance_strategy = DistanceStrategy.COSINE
  assistant = factories.AssistantFactory()
  docfiles = factories.DocumentFileFactory.create_batch(max_files, assistant=assistant)
  ndim = 5
  exact_vector = normalizer(np.linspace(0.1, 9.1, ndim))
  scales = np.array([0.9997, 0.9999, 1.0001])
  embedding_vectors = normalizer(np.power(exact_vector[:, np.newaxis], scales))
  # Create embedding stores
  for docfile in docfiles:
    for target_vector in embedding_vectors.T:
      factories.EmbeddingStoreFactory(assistant=assistant, docfile=docfile, embedding=target_vector, ndim=ndim)
  # Create embedding store of another assistant
  another_assistant = factories.AssistantFactory()
  factories.EmbeddingStoreFactory(
    assistant=another_assistant,
    docfile=factories.DocumentFileFactory(assistant=another_assistant),
    embedding=exact_vector,
    ndim=ndim
  )
  # Create target ids of document file
  docfile_ids = [docfile.pk for docfile in docfiles[:selected_num]]
  # Search similer document
  queryset = EmbeddingStore.objects.similarity_search_with_distance_by_vector(
    exact_vector,
    distance_strategy,
    assistant_id=assistant.pk,
    docfile_ids=docfile_ids,
  )

  assert queryset.count() == expected_records

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('is_open,is_raise,expected', [
  (False, False, 2),
  (False, True, 0),
  (True, False, 2),
  (True, True, 0),
], ids=[
  'file-is-not-open-and-not-raise-exception',
  'file-is-not-open-and-raise-exception',
  'file-is-open-and-not-raise-exception',
  'file-is-open-and-raise-exception',
])
def test_check_from_files_method_of_document_file(mocker, is_open, is_raise, expected):
  class DummyIngest:
    def __init__(self, *args, **kwargs):
      pass
    def convert_input2blob(self, *args, **kwargs):
      return object()
    def invoke(self, *args, **kwargs):
      if is_raise:
        raise Exception('error')

      return [1]

  class CustomUploadedFile(SimpleUploadedFile):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self._closed = False
      self.mode = 'rb'

    @property
    def closed(self):
      return self._closed

    @closed.setter
    def closed(self, is_closed):
      self._closed = is_closed

    def open(self, *args, **kwargs):
      return self

  mocker.patch('chatbot.models.rag.IngestBlobRunnable', new=DummyIngest)
  config = {
    'model': 'sample',
    'temperature': 0,
    'stream': True,
    'max_retries': 3,
    'api_key': 'open-ai-key',
    'endpoint': 'http://dummy-open-ai/endpoint',
  }
  assistant = factories.AssistantFactory(
    embedding=factories.EmbeddingFactory(
      config=config,
      emb_type=AgentType.OPENAI,
    )
  )
  files = [
    CustomUploadedFile('sample1.txt', b'This is a sample text.'),
    CustomUploadedFile('sample2.pdf', b'%PDF%%EOF'),
  ]

  if not is_open:
    for _file in files:
      _file.closed = not is_open

  out = DocumentFile.from_files(assistant, files)
  queryset = DocumentFile.objects.all()

  assert len(out) == expected
  assert queryset.count() == expected

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
def test_langgraph_checkpoint_creation():
  thread = factories.ThreadFactory()
  checkpoint = {'data': 2, 'value': 'a', 'name': 'hoge'}
  current_time = make_aware(datetime(2002, 3, 2, 11, 21, 0))
  previous_time = make_aware(datetime(2001, 3, 1, 10, 12, 0))
  instance, created = LangGraphCheckpoint.objects.update_or_create(
    thread_id=thread.pk,
    current_time=current_time,
    previous_time=previous_time,
    checkpoint=checkpoint,
  )
  total = LangGraphCheckpoint.objects.all().count()

  assert total == 1
  assert created
  assert instance.thread.pk == thread.pk
  assert instance.current_time == current_time
  assert instance.previous_time == previous_time
  assert instance.checkpoint == checkpoint

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize([
  'current_year',
  'previous_year',
  'new_year',
  'expected_current_year',
  'expected_previous_year',
  'expected_total',
  'expected_created',
], [
  (2000, 1950, 2000, 2000, 1950, 1, False),
  (2000, 1950, 2010, 2010, 2000, 2, True),
], ids=['current_time-eq-new_new', 'current_time-not-eq-new_time'])
def test_langgraph_checkpoint_updation(
  current_year,
  previous_year,
  new_year,
  expected_current_year,
  expected_previous_year,
  expected_total,
  expected_created,
):
  thread = factories.ThreadFactory()
  checkpoint = {'data': 2, 'value': 'a', 'name': 'hoge'}
  timestamp_getter = lambda _year: make_aware(datetime(_year, 3, 2, 11, 21, 0))
  current_time = timestamp_getter(current_year)
  previous_time = timestamp_getter(previous_year)
  new_time = timestamp_getter(new_year)
  expected_current_time = timestamp_getter(expected_current_year)
  expected_previous_time = timestamp_getter(expected_previous_year)
  _ = factories.CheckpointFactory(
    thread=thread,
    current_time=current_time,
    previous_time=previous_time,
    checkpoint={}
  )
  instance, created = LangGraphCheckpoint.objects.update_or_create(
    thread_id=thread.pk,
    current_time=new_time,
    previous_time=current_time,
    checkpoint=checkpoint,
  )
  total = LangGraphCheckpoint.objects.all().count()

  assert total == expected_total
  assert created == expected_created
  assert instance.thread.pk == thread.pk
  assert instance.current_time == expected_current_time
  assert instance.previous_time == expected_previous_time
  assert instance.checkpoint == checkpoint

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize([
  'shift_size_of_thread_id',
  'checkpoint',
], [
  (1, {'status': 'valid'}),
  (0, None),
], ids=['thread_id-is-invalid', 'checkpoint-doesnot-exist'])
def test_langgraph_checkpoint_invalid_patterns(shift_size_of_thread_id, checkpoint):
  thread = factories.ThreadFactory()
  current_time = make_aware(datetime(2022, 8, 12, 15, 31, 0))
  previous_time = make_aware(datetime(2021, 5, 11, 10, 12, 0))
  old = factories.CheckpointFactory(
    thread=thread,
    current_time=current_time,
    previous_time=previous_time,
    checkpoint={}
  )
  thread_id = thread.pk + shift_size_of_thread_id
  instance, created = LangGraphCheckpoint.objects.update_or_create(
    thread_id=thread_id,
    current_time=current_time,
    previous_time=previous_time,
    checkpoint=checkpoint,
  )
  total = LangGraphCheckpoint.objects.all().count()

  assert total == 1
  assert instance is None
  assert not created

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('target_year,expected_query_num', [
  (None, 2),
  (1950, 1),
  (1900, 0),
], ids=['match-multi-records', 'match-only-one-record', 'no-records-exists'])
def test_langgraph_checkpoint_collection(target_year, expected_query_num):
  thread = factories.ThreadFactory()
  timestamp_getter = lambda _year: make_aware(datetime(_year, 2, 2, 2, 2, 2))
  thread_id = thread.pk
  checkpoint = {'value': 3}
  factories.CheckpointFactory(
    thread_id=thread.pk,
    current_time=timestamp_getter(1925),
    previous_time=timestamp_getter(1900),
    checkpoint=checkpoint,
  )
  factories.CheckpointFactory(
    thread_id=thread_id,
    current_time=timestamp_getter(1950),
    previous_time=timestamp_getter(1930),
    checkpoint=checkpoint,
  )
  search_time = timestamp_getter(target_year) if target_year is not None else None
  queryset = LangGraphCheckpoint.objects.collect_checkpoints(thread_id=thread_id, thread_ts=search_time)
  total = queryset.count()

  assert total == expected_query_num