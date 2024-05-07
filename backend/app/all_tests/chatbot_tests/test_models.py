import pytest
from chatbot.models.agents import AgentArgs, ToolArgs, AgentType, ToolType
from chatbot.models.rag import (
  BaseConfig,
  Agent,
  Embedding,
  Tool,
  Assistant,
  Thread,
  EmbeddingStore,
  LangGraphCheckpoint,
)
# For test
import numpy as np
import chatbot.models.utils.llms as llms
import chatbot.models.utils.executors as executors
import chatbot.models.utils.tools as tools
from chatbot.models.utils.vectorstore import DistanceStrategy
from django.core.exceptions import ValidationError
from django.utils.timezone import make_aware
from datetime import datetime
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
  'expeceted_tool',
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
])
def test_check_property_of_tooltype(tool_type, expeceted_tool):
  tool_id = tool_type.value
  instance = ToolType(tool_id)

  assert type(instance._tool_type) == type(expeceted_tool)

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
  embedded_config_field = AgentType.get_llm_fields(agent_id, is_embedded=True)
  llm_config_field = AgentType.get_llm_fields(agent_id, is_embedded=False)
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
  'expeceted_llm',
  'expeceted_executor',
], [
  (AgentType.OPENAI, llms.OpenAILLM, executors.ToolExecutor),
  (AgentType.AZURE, llms.AzureOpenAILLM, executors.ToolExecutor),
  (AgentType.ANTHROPIC, llms.AnthropicLLM, executors.ToolExecutor),
  (AgentType.BEDROCK, llms.BedrockLLM, executors.XmlExecutor),
  (AgentType.FIREWORKS, llms.FireworksLLM, executors.ToolExecutor),
  (AgentType.OLLAMA, llms.OllamaLLM, executors.ToolExecutor),
  (AgentType.GEMINI, llms.GeminiLLM, executors.ToolExecutor),
])
def test_check_property_of_agenttype(agent_type, expeceted_llm, expeceted_executor):
  agent_id = agent_type.value
  instance = AgentType(agent_id)

  assert type(instance._llm_type) == type(expeceted_llm)
  assert type(instance._executor_type) == type(expeceted_executor)

@pytest.mark.chatbot
@pytest.mark.model
def test_check_specific_method_of_agenttype():
  choices = AgentType.get_embedding_choices()
  validator = AgentType.get_embedding_validator()
  value = AgentType.OPENAI.value

  assert AgentType.ANTHROPIC not in choices
  assert value == validator(value)
  with pytest.raises(ValidationError) as ex:
    validator(AgentType.ANTHROPIC.value)
  assert 'invalid AgentType' in str(ex.value)

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
  collected = BaseConfig.objects.get_or_none(pk=instance.pk)
  doesnot_exist = BaseConfig.objects.get_or_none(pk=instance.pk+1)

  assert instance.name == name
  assert instance.config == config
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

  assert default_instance.agent_type == AgentType.OPENAI.value
  assert str(specific_agent) == f'{name} ({_agent_type})'
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
    emb=_emb_type,
  )
  random_embedding = factories.EmbeddingFactory()
  default_ds = default_instance.get_distance_strategy()
  specific_ds = specific_embedding.get_distance_strategy()
  spec_val = specific_embedding.get_embedding()
  rand_val = random_embedding.get_embedding()
  _exists = Embedding.objects.get_or_none(pk=default_instance.pk)

  assert default_instance.emb == AgentType.OPENAI.value
  assert default_instance.distance_strategy == Embedding.DistanceType.COSINE.value
  assert str(specific_embedding) == f'{name} ({_emb_type})'
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

  assert default_instance.tool_type == ToolType.RETRIEVER.value
  assert str(specific_tool) == f'{name} ({_tool_type})'
  assert spec_val == ret_val
  assert rand_val == ret_val
  assert _exists is not None

@pytest.mark.chatbot
@pytest.mark.model
@pytest.mark.django_db
@pytest.mark.parametrize('num_tools,', [0, 1, 2])
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
  _tool_records = factories.ToolFactory.create_batch(num_tools) if num_tools > 0 else []
  assistant = Assistant.objects.create(
    user=factories.UserFactory(),
    name=name,
    agent=factories.AgentFactory(),
    embedding=factories.EmbeddingFactory(),
  )
  assistant.tools.add(*_tool_records)
  ret_items = assistant.get_assistant()
  expected_retval = len(_tool_records) * num_tools

  assert str(assistant) == name
  assert 'You are a helpful assistant.' in assistant.system_message
  assert not assistant.is_interrupt
  assert len(ret_items) == expected_retval

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
def test_check_str_method_of_embedding_store():
  screen_name = 'test-user'
  assistant_name = 'dummy'
  assistant = factories.AssistantFactory(
    user=factories.UserFactory(screen_name=screen_name),
    name=assistant_name,
  )
  store = EmbeddingStore.objects.create(
    assistant_id=assistant.pk,
    embedding=[1,2,3],
    document='sample-text',
  )

  assert str(store) == f'{assistant_name} ({screen_name})'

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
  # Create embedding stores
  stores = [
    factories.EmbeddingStoreFactory(assistant=assistant, embedding=target_vector, ndim=ndim)
    for target_vector in embedding_vectors.T
  ]
  ids = np.array([store.pk for store in stores])
  # Calculate distance of each vector
  distance = calc_similarity(embedding_vectors.T, exact_vector)
  sorted_indices = np.argsort(sort_direction * distance)
  # Collect expected primary keys
  print(sorted_indices, sorted_indices.shape, ids.shape)
  exact_ids = ids[sorted_indices]
  # Calculate distance by using the Django manager's method
  queryset = EmbeddingStore.objects.similarity_search_with_distance_by_vector(exact_vector, assistant.pk, distance_strategy)
  estimated_ids = np.array(queryset.values_list('pk', flat=True))

  assert (exact_ids == estimated_ids).all()

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
  print(instance, created)

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
])
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
