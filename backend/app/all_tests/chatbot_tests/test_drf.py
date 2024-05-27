import pytest
import json
from chatbot import drf_views, models
from chatbot.models.agents import AgentType, ToolType
# For test
from importlib import import_module, reload
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.test import APIRequestFactory
from rest_framework import status, serializers
from rest_framework.reverse import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test.client import MULTIPART_CONTENT, encode_multipart, BOUNDARY
from . import factories

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('viewset,target,funcname,urlname,choices', [
  (drf_views.AgentViewSet, 'agent', 'get_type_ids', 'types', AgentType.choices),
  (drf_views.EmbeddingViewSet, 'embedding', 'get_type_ids', 'types', AgentType.embedding_choices),
  (drf_views.EmbeddingViewSet, 'embedding', 'get_distances_types', 'distances', models.Embedding.DistanceType.choices),
  (drf_views.ToolViewSet, 'tool', 'get_type_ids', 'types', ToolType.choices),
], ids=['agent-type-ids', 'embedding-type-ids', 'embedding-distances', 'tool-type-ids'])
def test_drf_type_ids(viewset, target, funcname, urlname, choices):
  user = factories.UserFactory()
  refresh = RefreshToken.for_user(user)
  factory = APIRequestFactory()
  view = viewset.as_view({'get': funcname})
  url = reverse(f'api:chatbot:{target}_{urlname}')
  request = factory.get(url, HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  response = view(request)
  outputs = response.data
  expected_ids = [value for value, _ in choices]
  expected_names = [label for _, label in choices]

  assert urlname in outputs.keys()
  assert all([target['id'] in expected_ids for target in outputs[urlname]])
  assert all([target['name'] in expected_names for target in outputs[urlname]])

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('viewset,target,type_id,getter,kwargs', [
  (drf_views.AgentViewSet, 'agent', value, AgentType.get_llm_fields, {'is_embedded': False}) for value, _ in AgentType.choices
] + [
  (drf_views.EmbeddingViewSet, 'embedding', value, AgentType.get_llm_fields, {'is_embedded': True}) for value, _ in AgentType.embedding_choices
] + [
  (drf_views.ToolViewSet, 'tool', value, ToolType.get_config_field, {}) for value, _ in ToolType.choices
],
ids=[
  f'agent-{label}' for _, label in AgentType.choices
] + [
  f'embedding-{label}' for _, label in AgentType.embedding_choices
] + [
  f'tool-{label}' for _, label in ToolType.choices
])
def test_drf_config_formats(viewset, target, type_id, getter, kwargs):
  user = factories.UserFactory()
  refresh = RefreshToken.for_user(user)
  factory = APIRequestFactory()
  view = viewset.as_view({'get': 'get_config_format'})
  url = '{}?type_id={}'.format(reverse(f'api:chatbot:{target}_config_format'), type_id)
  request = factory.get(url, HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  response = view(request)
  outputs = response.data
  exact_fields = getter(type_id, {}, **kwargs)
  expected_config = {}

  for target in exact_fields:
    expected_config.update(target.asdict())
  estimated_keys = outputs.keys()

  assert 'name' in estimated_keys
  assert 'format' in estimated_keys
  assert expected_config == outputs['format']

# ==================
# = Common fixture =
# ==================
@pytest.fixture
def drf_specific_viewsets():
  list_create_kwargs = {'get': 'list', 'post': 'create'}
  targets = [
    (drf_views.AgentViewSet, 'agent', list_create_kwargs, {'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}),
    (drf_views.EmbeddingViewSet, 'embedding', list_create_kwargs, {'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}),
    (drf_views.ToolViewSet, 'tool', list_create_kwargs, {'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}),
    (drf_views.AssistantViewSet, 'assistant', list_create_kwargs, {'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}),
    (drf_views.DocumentFileViewSet, 'docfile', list_create_kwargs, {'delete': 'destroy'}),
    (drf_views.ThreadViewSet, 'thread', list_create_kwargs, {'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}),
  ]

  return targets

@pytest.fixture
def drf_list_create_settings(drf_specific_viewsets):
  targets = drf_specific_viewsets
  callbacks = {}
  factory = APIRequestFactory()

  for viewset, basename, list_kwargs, _ in targets:
    def wrapper(_viewset, _basename):
      view = _viewset.as_view(list_kwargs)
      url = reverse(f'api:chatbot:{_basename}_list')
      callback = lambda method, **kwargs: view(getattr(factory, method)(url, **kwargs))

      return callback

    callbacks[basename] = wrapper(viewset, basename)

  return callbacks

@pytest.fixture
def drf_retrieve_update_destroy_settings(drf_specific_viewsets):
  targets = drf_specific_viewsets
  callbacks = {}
  factory = APIRequestFactory()

  for viewset, basename, _, detail_kwargs in targets:
    def wrapper(_viewset, _basename):
      view = _viewset.as_view(detail_kwargs)

      def callback(pk, method, **kwargs):
        url = reverse(f'api:chatbot:{_basename}_detail', kwargs={'pk': pk})
        handler = getattr(factory, method)
        response = view(handler(url, **kwargs), pk=pk)

        return response

      return callback

    callbacks[basename] = wrapper(viewset, basename)

  return callbacks

# ===============
# = list method =
# ===============
@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('factory_class,target', [
  (factories.AgentFactory, 'agent'),
  (factories.EmbeddingFactory, 'embedding'),
  (factories.ToolFactory, 'tool'),
  (factories.AssistantFactory, 'assistant'),
], ids=['agent-list', 'embedding-list', 'tool-list', 'assistant-list'])
def test_drf_list_method(drf_list_create_settings, factory_class, target):
  _callbacks = drf_list_create_settings
  callback = _callbacks[target]
  expected_count = 4
  user, other = factories.UserFactory.create_batch(2)
  _ = factory_class.create_batch(expected_count, user=user)
  _ = factory_class.create_batch(3, user=other)
  refresh = RefreshToken.for_user(user)
  response = callback('get', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')

  assert response.status_code == status.HTTP_200_OK
  assert len(response.data) == expected_count

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('factory_class,target', [
  (factories.DocumentFileFactory, 'docfile'),
  (factories.ThreadFactory, 'thread'),
], ids=['documentfile-list', 'thread-list'])
def test_drf_docfile_and_thread_list_method(drf_list_create_settings, factory_class, target):
  _callbacks = drf_list_create_settings
  callback = _callbacks[target]
  expected_count = 4
  user, other = factories.UserFactory.create_batch(2)
  user_assistant = factories.AssistantFactory(user=user)
  other_assistant = factories.AssistantFactory(user=other)
  _ = factory_class.create_batch(expected_count, assistant=user_assistant)
  _ = factory_class.create_batch(3, assistant=other_assistant)
  refresh = RefreshToken.for_user(user)
  response = callback('get', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')

  assert response.status_code == status.HTTP_200_OK
  assert len(response.data) == expected_count

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_assistant_retrieve_method(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['assistant']

  def get_instances(owner, num):
    instances = []

    for _ in range(num):
      agent = factories.AgentFactory(user=owner)
      embedding = factories.EmbeddingFactory(user=owner)
      tools = factories.ToolFactory.create_batch(3, user=owner)
      target = factories.AssistantFactory(
        user=owner,
        agent=agent,
        embedding=embedding,
        tools=tuple(tools),
      )
      instances.append(target)

    return instances

  user, other = factories.UserFactory.create_batch(2)
  targets = get_instances(user, 4)
  _ = get_instances(other, 3)
  instance = targets[2]
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'get', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  expect_agent = instance.agent
  expect_embedding = instance.embedding
  expect_tool_pks = instance.tools.all().values_list('pk', flat=True)
  _agent = response.data['agent']
  _embedding = response.data['embedding']
  _tools = response.data['tools']

  assert response.status_code == status.HTTP_200_OK
  assert _agent['name'] == expect_agent.name
  assert _agent['config'] == expect_agent.config
  assert _agent['agent_type']['id'] == expect_agent.agent_type
  assert _embedding['name'] == expect_embedding.name
  assert _embedding['config'] == expect_embedding.config
  assert _embedding['distance_strategy']['id'] == expect_embedding.distance_strategy
  assert _embedding['emb_type']['id'] == expect_embedding.emb_type
  assert all([_target_tool['pk'] in expect_tool_pks for _target_tool in _tools])

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_thread_retrieve_method(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['thread']
  user, other = factories.UserFactory.create_batch(2)
  user_assistant = factories.AssistantFactory(user=user)
  other_assistant = factories.AssistantFactory(user=other)
  targets = factories.ThreadFactory.create_batch(4, assistant=user_assistant)
  _ = factories.ThreadFactory.create_batch(3, assistant=other_assistant)
  instance = targets[2]
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'get', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  expected_assistant = instance.assistant
  _assistant = response.data['assistant']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == instance.name
  assert _assistant['pk'] == expected_assistant.pk

# ==================
# = destroy method =
# ==================
@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('factory_class,model_class,target', [
  (factories.AgentFactory, models.Agent, 'agent'),
  (factories.EmbeddingFactory, models.Embedding, 'embedding'),
  (factories.ToolFactory, models.Tool, 'tool'),
  (factories.AssistantFactory, models.Assistant, 'assistant'),
], ids=['agent-destroy', 'embedding-destroy', 'tool-destroy', 'assistant-destroy'])
def test_drf_destroy_method(drf_retrieve_update_destroy_settings, factory_class, model_class, target):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks[target]
  user = factories.UserFactory()
  instance = factory_class(user=user)
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'delete', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  total = model_class.objects.all().count()

  assert response.status_code == status.HTTP_204_NO_CONTENT
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('factory_class,model_class,target', [
  (factories.DocumentFileFactory, models.DocumentFile, 'docfile'),
  (factories.ThreadFactory, models.Thread, 'thread'),
], ids=['documentfile-destroy', 'thread-destroy'])
def test_drf_docfile_and_thread_destroy_method(drf_retrieve_update_destroy_settings, factory_class, model_class, target):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks[target]
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(user=user)
  instance = factory_class(assistant=assistant)
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'delete', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')
  total = model_class.objects.all().count()

  assert response.status_code == status.HTTP_204_NO_CONTENT
  assert total == 0

# ==================
# = create methods =
# ==================
@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_agent_create(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['agent']
  user = factories.UserFactory()
  form_data = {
    'name': 'agent-test',
    'agent_type': AgentType.AZURE.value,
    'config': {
      'model': 'sample-agent-model',
      'temperature': 0.1,
      'stream': True,
      'max_retries': 3,
      'api_key': 'sample-agent-azure',
      'endpoint': 'http://agent-dummy.com',
      'version': 'test-agent-version',
      'deployment': 'http://agent-test-dev.com',
    },
  }
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Agent.objects.all().count()
  exact_config = form_data['config']
  estimated_type = response.data['agent_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_201_CREATED
  assert response.data['name'] == form_data['name']
  assert estimated_type['id'] == form_data['agent_type']
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_embedding_create(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['embedding']
  user = factories.UserFactory()
  form_data = {
    'name': 'embedding-test',
    'emb_type': AgentType.AZURE.value,
    'distance_strategy': models.Embedding.DistanceType.EUCLIDEAN,
    'config': {
      'model': 'sample-embedding-model',
      'max_retries': 4,
      'api_key': 'sample-embedding-azure',
      'endpoint': 'http://embedding-dummy.com',
      'version': 'test-embedding-version',
      'deployment': 'http://embedding-test-dev.com',
    },
  }
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Embedding.objects.all().count()
  exact_config = form_data['config']
  estimated_type = response.data['emb_type']
  estimated_strategy = response.data['distance_strategy']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_201_CREATED
  assert response.data['name'] == form_data['name']
  assert estimated_type['id'] == form_data['emb_type']
  assert estimated_strategy['id'] == form_data['distance_strategy']
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_tool_create(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['tool']
  user = factories.UserFactory()
  form_data = {
    'name': 'tool-test',
    'tool_type': ToolType.RETRIEVER.value,
    'config': {
      'k': 5,
    },
  }
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Tool.objects.all().count()
  exact_config = form_data['config']
  estimated_type = response.data['tool_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_201_CREATED
  assert response.data['name'] == form_data['name']
  assert estimated_type['id'] == form_data['tool_type']
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_valid_config(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['tool']
  user = factories.UserFactory()
  form_data = {
    'name': 'test-tool',
    'tool_type': ToolType.ARXIV.value,
  }
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  outputs = response.data
  instance = models.Tool.objects.get(pk=outputs['pk'])
  total = models.Tool.objects.all().count()

  assert response.status_code == status.HTTP_201_CREATED
  assert total == 1
  assert len(instance.config) == 0

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('target_type,model_class,form_data', [
  ('agent', models.Agent, {'config': {}}),
  ('agent', models.Agent, {'name': 'test-agent'}),
  ('embedding', models.Embedding, {'config': {}}),
  ('embedding', models.Embedding, {'name': 'test-embedding'}),
  ('tool', models.Tool, {'config': {}}),
  ('tool', models.Tool, {'name': 'test-tool'}),
  ('tool', models.Tool, {'name': 'test-tool', 'tool_type': ToolType.ARXIV.value, 'config': {'api_key': 'dummy-key'}}),
], ids=[
  'no-name-exists-at-agent',
  'no-config-exists-at-agent',
  'no-name-exists-at-embedding',
  'no-config-exists-at-embedding',
  'no-name-exists-at-tool',
  'no-config-exists-at-tool',
  'invalid-config-at-tool',
])
def test_drf_invalid_patterns(drf_list_create_settings, target_type, model_class, form_data):
  _callbacks = drf_list_create_settings
  callback = _callbacks[target_type]
  user = factories.UserFactory()
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = model_class.objects.all().count()

  assert response.status_code == status.HTTP_400_BAD_REQUEST
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_invalid_emb_type(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['embedding']
  user = factories.UserFactory()
  form_data = {
    'name': 'test',
    'config': {},
    'emb_type': AgentType.ANTHROPIC
  }
  refresh = RefreshToken.for_user(user)
  with pytest.raises(ValueError) as ex:
    _ = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Embedding.objects.all().count()

  assert 'not implemented' in str(ex.value)
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('num_tools,kwargs,expected', [
  (0, {}, {'system_message': 'You are a helpful assistant.', 'is_interrupt': False}),
  (1, {'system_message': 'You are a teacher.'}, {'system_message': 'You are a teacher.', 'is_interrupt': False}),
  (2, {'is_interrupt': True}, {'system_message': 'You are a helpful assistant.', 'is_interrupt': True}),
], ids=['one-tool-no-kwargs', 'two-tools-set-msg', 'three-tools-set-interrupt-option'])
def test_drf_assistant_create(drf_list_create_settings, num_tools, kwargs, expected):
  _callbacks = drf_list_create_settings
  callback = _callbacks['assistant']
  user = factories.UserFactory()
  agent = factories.AgentFactory(user=user)
  embedding = factories.EmbeddingFactory(user=user)
  tools = factories.ToolFactory.create_batch(num_tools, user=user)
  tool_pks = [tool.pk for tool in tools]
  form_data = {
    'name': 'assistant-test',
    'agent_pk': agent.pk,
    'embedding_pk': embedding.pk,
    'tool_pks': tool_pks,
  }
  form_data.update(kwargs)
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Assistant.objects.all().count()

  assert response.status_code == status.HTTP_201_CREATED
  assert response.data['name'] == form_data['name']
  assert response.data['system_message'] == expected['system_message']
  assert response.data['agent']['pk'] == agent.pk
  assert response.data['embedding']['pk'] == embedding.pk
  assert all([item['pk'] in tool_pks for item in response.data['tools']])
  assert response.data['is_interrupt'] == expected['is_interrupt']
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('pattern,invalid_key,name',[
  ({'agent': 'other', 'embedding': 'own', 'tool': 'own'}, 'agent_pk', 'invalid-agent'),
  ({'agent': 'own', 'embedding': 'other', 'tool': 'own'}, 'embedding_pk', 'invalid-embedding'),
  ({'agent': 'own', 'embedding': 'own', 'tool': 'other'}, 'tool_pks', 'invalid-tool'),
], ids=['invalid-agent', 'invalid-embedding', 'invalid-tool'])
def test_drf_invalid_assistant_creation(drf_list_create_settings, pattern, invalid_key, name):
  _callbacks = drf_list_create_settings
  callback = _callbacks['assistant']
  users, agents, embeddings, tools = {}, {}, {}, {}

  for key in ['own', 'other']:
    _user = factories.UserFactory()
    users[key] = _user
    agents[key] = factories.AgentFactory(user=_user)
    embeddings[key] = factories.EmbeddingFactory(user=_user)
    tools[key] = factories.ToolFactory(user=_user)

  form_data = {
    'name': name,
    'agent_pk': agents[pattern['agent']].pk,
    'embedding_pk': embeddings[pattern['embedding']].pk,
    'tool_pks': [tools[pattern['tool']].pk],
  }
  refresh = RefreshToken.for_user(users['own'])
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  errs = output.get(invalid_key, None)
  total = models.Assistant.objects.all().count()

  if invalid_key == 'tool_pks':
    no_data = form_data[invalid_key][0]
  else:
    no_data = form_data[invalid_key]

  assert response.status_code == status.HTTP_400_BAD_REQUEST
  assert errs is not None
  assert 'does_not_exist' == errs[0].code
  assert str(no_data) in str(errs[0])
  assert total == 0

@pytest.fixture
def get_specific_assistant():
  config = {
    'model': 'sample',
    'temperature': 0,
    'stream': True,
    'max_retries': 3,
    'api_key': 'open-ai-key',
    'endpoint': 'http://dummy-open-ai/endpoint',
  }

  def getter(user):
    assistant = factories.AssistantFactory(
      user=user,
      embedding=factories.EmbeddingFactory(
        config=config,
        emb_type=AgentType.OPENAI,
      )
    )

    return assistant

  return getter

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('num,ext', [
  (0, 'txt'),
  (1, 'txt'),
  (2, 'pdf'),
  (3, 'docx'),
  (4, 'html'),
], ids=lambda value: f'{value}')
def test_drf_docfile_creation(drf_list_create_settings, get_specific_assistant, mocker, num, ext):
  class DummyIngest:
    def __init__(self, *args, **kwargs):
      pass
    def convert_input2blob(self, *args, **kwargs):
      return object()
    def invoke(self, *args, **kwargs):
      docfile_id = kwargs.get('docfile_id')

      return [str(docfile_id)]

  module = import_module('chatbot.models.utils.ingest')
  _ = getattr(module, 'IngestBlobRunnable')
  mocker.patch('chatbot.models.rag.IngestBlobRunnable', new=DummyIngest)
  reload(module)

  getter = get_specific_assistant
  _callbacks = drf_list_create_settings
  callback = _callbacks['docfile']
  user = factories.UserFactory()
  assistant = getter(user)
  file_fields = [SimpleUploadedFile(f'test.{ext}', f'This is a {idx+1}th-sample'.encode('utf-8')) for idx in range(num)]
  form_data = {
    'assistant_pk': assistant.pk,
    'upload_files': file_fields,
  }
  refresh = RefreshToken.for_user(user)
  response = callback(
    'post',
    HTTP_AUTHORIZATION=f'JWT {refresh.access_token}',
    data=encode_multipart(data=form_data, boundary=BOUNDARY),
    content_type=MULTIPART_CONTENT,
  )
  total = models.DocumentFile.objects.all().count()

  assert response.status_code == status.HTTP_202_ACCEPTED
  assert total == num

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('num,ext', [
  (2, 'csv'),
  (2, 'xlsx'),
  (2, 'dat'),
], ids=lambda value: f'{value}')
def test_drf_invalid_pattern_of_docfile_creation(drf_list_create_settings, get_specific_assistant, mocker, num, ext):
  class DummyIngest:
    def __init__(self, *args, **kwargs):
      pass
    def convert_input2blob(self, *args, **kwargs):
      return object()
    def invoke(self, *args, **kwargs):
      docfile_id = kwargs.get('docfile_id')

      return [str(docfile_id)]

  module = import_module('chatbot.models.utils.ingest')
  _ = getattr(module, 'IngestBlobRunnable')
  mocker.patch('chatbot.models.rag.IngestBlobRunnable', new=DummyIngest)
  reload(module)

  getter = get_specific_assistant
  _callbacks = drf_list_create_settings
  callback = _callbacks['docfile']
  user = factories.UserFactory()
  assistant = getter(user)
  file_fields = [SimpleUploadedFile(f'test.{ext}', f'This is a {idx+1}th-sample'.encode('utf-8')) for idx in range(num)]
  form_data = {
    'assistant_pk': assistant.pk,
    'upload_files': file_fields,
  }
  refresh = RefreshToken.for_user(user)
  response = callback(
    'post',
    HTTP_AUTHORIZATION=f'JWT {refresh.access_token}',
    data=encode_multipart(data=form_data, boundary=BOUNDARY),
    content_type=MULTIPART_CONTENT,
  )
  total = models.DocumentFile.objects.all().count()

  assert response.status_code == status.HTTP_400_BAD_REQUEST
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_thread_creation(drf_list_create_settings):
  _callbacks = drf_list_create_settings
  callback = _callbacks['thread']
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(user=user)
  docfiles = factories.DocumentFileFactory.create_batch(3, assistant=assistant)
  docfile_pks = [docfiles[0].pk, docfiles[-1].pk]
  form_data = {
    'assistant_pk': assistant.pk,
    'name': 'test-thread',
    'docfile_pks': docfile_pks,
  }
  refresh = RefreshToken.for_user(user)
  response = callback('post', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  total = models.Thread.objects.all().count()

  assert response.status_code == status.HTTP_201_CREATED
  assert output['name'] == form_data['name']
  assert output['assistant']['pk'] == assistant.pk
  assert all([item['pk'] in docfile_pks for item in output['docfiles']])
  assert total == 1

# ==================
# = update methods =
# ==================
@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_agent_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['agent']
  user = factories.UserFactory()
  type_id = AgentType.AZURE.value
  instance = factories.AgentFactory(
    user=user,
    name='test-azure',
    agent_type=type_id,
    config={
      'model': 'azure',
      'max_retries': 10,
    },
  )
  form_data = {
    'name': 'replaced-azure',
    'config': {
      'model': 'replaced-model',
      'max_retries': 3,
      'api_key': 'replace-azure',
    }
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'put', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Agent.objects.all().count()
  exact_config = form_data['config']
  estimated_type_id = response.data['agent_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == form_data['name']
  assert estimated_type_id['id'] == type_id
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_embedding_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['embedding']
  user = factories.UserFactory()
  type_id = AgentType.AZURE.value
  instance = factories.EmbeddingFactory(
    user=user,
    name='test-azure',
    emb_type=type_id,
    config={
      'model': 'azure',
      'max_retries': 10,
    },
  )
  form_data = {
    'name': 'replaced-azure',
    'config': {
      'model': 'replaced-model',
      'max_retries': 3,
      'api_key': 'replace-azure',
    }
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'put', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Embedding.objects.all().count()
  exact_config = form_data['config']
  estimated_type_id = response.data['emb_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == form_data['name']
  assert estimated_type_id['id'] == type_id
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_tool_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['tool']
  user = factories.UserFactory()
  type_id = ToolType.WIKIPEDIA.value
  instance = factories.ToolFactory(
    user=user,
    name='test-wiki',
    tool_type=type_id,
  )
  form_data = {
    'name': 'rename-wikipedia',
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'put', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Tool.objects.all().count()
  estimated_type_id = response.data['tool_type']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == form_data['name']
  assert estimated_type_id['id'] == type_id
  assert len(response.data['config']) == 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_assistant_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['assistant']
  user = factories.UserFactory()
  _tool_records = factories.ToolFactory.create_batch(2, user=user)
  instance = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  instance.tools.add(*_tool_records)
  other_agent = factories.AgentFactory(user=user)
  other_embedding = factories.EmbeddingFactory(user=user)
  other_tool_pks = [tool.pk for tool in factories.ToolFactory.create_batch(3, user=user)]
  form_data = {
    'name': 'replaced-assistant',
    'agent_pk': other_agent.pk,
    'embedding_pk': other_embedding.pk,
    'tool_pks': other_tool_pks,
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'put', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Assistant.objects.all().count()

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == form_data['name']
  assert response.data['agent']['pk'] == form_data['agent_pk']
  assert response.data['embedding']['pk'] == form_data['embedding_pk']
  assert all([item['pk'] in other_tool_pks for item in response.data['tools']])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_thread_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['thread']
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(user=user)
  docfiles = factories.DocumentFileFactory.create_batch(3, assistant=assistant)
  instance = factories.ThreadFactory(assistant=assistant, docfiles=(docfiles[0], docfiles[1]))
  docfile_pks = [docfiles[-1].pk]
  form_data = {
    'assistant_pk': assistant.pk,
    'name': 'replaced-thread',
    'docfile_pks': docfile_pks,
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'put', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  total = models.Thread.objects.all().count()

  assert response.status_code == status.HTTP_200_OK
  assert output['assistant']['pk'] == assistant.pk
  assert output['name'] == form_data['name']
  assert all([item['pk'] in docfile_pks for item in output['docfiles']])
  assert total == 1

# ==========================
# = partial update methods =
# ==========================
@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_agent_partial_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['agent']
  user = factories.UserFactory()
  name = 'test-azure'
  type_id = AgentType.AZURE.value
  instance = factories.AgentFactory(
    user=user,
    name=name,
    agent_type=type_id,
  )
  form_data = {
    'config': {
      'model': 'replaced-name',
      'api_key': 'replaced-api-key',
    }
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Agent.objects.all().count()
  exact_config = form_data['config']
  estimated_type_id = response.data['agent_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == name
  assert estimated_type_id['id'] == type_id
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_embedding_partial_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['embedding']
  user = factories.UserFactory()
  name = 'test-azure'
  type_id = AgentType.AZURE.value
  instance = factories.EmbeddingFactory(
    user=user,
    name=name,
    emb_type=type_id,
  )
  form_data = {
    'config': {
      'model': 'replaced-name',
      'api_key': 'replaced-api-key',
    }
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Embedding.objects.all().count()
  exact_config = form_data['config']
  estimated_type_id = response.data['emb_type']
  estimated_config = response.data['config']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == name
  assert estimated_type_id['id'] == type_id
  assert all([estimated_config[key] == exact_config[key] for key in exact_config.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_tool_partial_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['tool']
  user = factories.UserFactory()
  name = 'test-arxiv'
  type_id = ToolType.ARXIV.value
  instance = factories.ToolFactory(
    user=user,
    name=name,
    tool_type=type_id,
  )
  form_data = {}
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Tool.objects.all().count()
  estimated_type_id = response.data['tool_type']

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == name
  assert estimated_type_id['id'] == type_id
  assert len(response.data['config']) == 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('pattern', [
  {'name': True,  'agent': True,  'embedding': True,  'tools': False},
  {'name': True,  'agent': True,  'embedding': False, 'tools': True},
  {'name': True,  'agent': True,  'embedding': False, 'tools': False},
  {'name': True,  'agent': False, 'embedding': True,  'tools': True},
  {'name': True,  'agent': False, 'embedding': True,  'tools': False},
  {'name': True,  'agent': False, 'embedding': False, 'tools': True},
  {'name': True,  'agent': False, 'embedding': False, 'tools': False},
  {'name': False, 'agent': True,  'embedding': True,  'tools': True},
  {'name': False, 'agent': True,  'embedding': True,  'tools': False},
  {'name': False, 'agent': True,  'embedding': False, 'tools': True},
  {'name': False, 'agent': True,  'embedding': False, 'tools': False},
  {'name': False, 'agent': False, 'embedding': True,  'tools': True},
  {'name': False, 'agent': False, 'embedding': True,  'tools': False},
  {'name': False, 'agent': False, 'embedding': False, 'tools': True},
], ids=lambda _dict: 'set-{}'.format('-'.join([key for key, val in _dict.items() if val])))
def test_drf_assistant_partial_update(drf_retrieve_update_destroy_settings, pattern):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['assistant']
  user = factories.UserFactory()
  _tool_records = factories.ToolFactory.create_batch(2, user=user)
  instance = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  instance.tools.add(*_tool_records)
  replacer = {
    'name': lambda: {'name': 'replaced-name'},
    'agent': lambda: {'agent_pk': factories.AgentFactory(user=user).pk},
    'embedding': lambda: {'embedding_pk': factories.EmbeddingFactory(user=user).pk},
    'tools': lambda: {'tool_pks': [tool.pk for tool in factories.ToolFactory.create_batch(3, user=user)]},
  }
  expected = {
    'name': instance.name,
    'agent_pk': instance.agent.pk,
    'embedding_pk': instance.embedding.pk,
    'tool_pks': [tool.pk for tool in instance.tools.all()],
  }
  form_data = {}

  for key, is_active in pattern.items():
    if is_active:
      _func = replacer[key]
      target = _func()
      form_data.update(target)
      expected.update(target)

  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  total = models.Assistant.objects.all().count()

  assert response.status_code == status.HTTP_200_OK
  assert response.data['name'] == expected['name']
  assert response.data['agent']['pk'] == expected['agent_pk']
  assert response.data['embedding']['pk'] == expected['embedding_pk']
  assert all([item['pk'] in expected['tool_pks'] for item in response.data['tools']])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('pattern', [
  {'name': False, 'docfiles': True},
  {'name': True,  'docfiles': False},
], ids=lambda _dict: 'set-{}'.format('-'.join([key for key, val in _dict.items() if val])))
def test_drf_thread_partial_update(drf_retrieve_update_destroy_settings, pattern):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['thread']
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(user=user)
  docfiles = factories.DocumentFileFactory.create_batch(3, assistant=assistant)
  instance = factories.ThreadFactory(assistant=assistant, docfiles=(docfiles[0], docfiles[1]))
  replacer = {
    'name': lambda: {'name': 'replaced-name'},
    'docfiles': lambda: {'docfile_pks': [docfiles[-1].pk]},
  }
  expected = {
    'name': instance.name,
    'assistant_pk': instance.assistant.pk,
    'docfile_pks': [docfile.pk for docfile in instance.docfiles.all()],
  }
  form_data = {}

  for key, is_active in pattern.items():
    if is_active:
      _func = replacer[key]
      target = _func()
      form_data.update(target)
      expected.update(target)

  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  total = models.Thread.objects.all().count()

  assert response.status_code == status.HTTP_200_OK
  assert output['name'] == expected['name']
  assert output['assistant']['pk'] == expected['assistant_pk']
  assert all([item['pk'] in expected['docfile_pks'] for item in output['docfiles']])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_invalid_assistant_of_partial_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['thread']
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(user=user)
  other = factories.AssistantFactory(user=user)
  docfiles = factories.DocumentFileFactory.create_batch(3, assistant=assistant)
  instance = factories.ThreadFactory(assistant=assistant, docfiles=tuple(docfiles))
  form_data = {
    'assistant_pk': other.pk,
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  errs = output.get('assistant_pk', None)
  total = models.Thread.objects.all().count()

  assert response.status_code == status.HTTP_400_BAD_REQUEST
  assert errs is not None
  assert 'Invalid primary key' in str(errs[0])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_invalid_docfiles_of_partial_update(drf_retrieve_update_destroy_settings):
  _callbacks = drf_retrieve_update_destroy_settings
  callback = _callbacks['thread']
  user, other = factories.UserFactory.create_batch(2)
  assistant = factories.AssistantFactory(user=user)
  docfiles = factories.DocumentFileFactory.create_batch(3, assistant=assistant)
  invalid_docs = factories.DocumentFileFactory.create_batch(2, assistant=factories.AssistantFactory(user=other))
  instance = factories.ThreadFactory(assistant=assistant, docfiles=tuple(docfiles))
  form_data = {
    'docfile_pks': [_docfile.pk for _docfile in invalid_docs],
  }
  refresh = RefreshToken.for_user(user)
  response = callback(instance.pk, 'patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data=json.dumps(form_data), format='json')
  output = response.data
  errs = output.get('docfile_pks', None)
  total = models.Thread.objects.all().count()

  assert response.status_code == status.HTTP_400_BAD_REQUEST
  assert errs is not None
  assert 'Invalid docfiles exist' in str(errs[0])
  assert total == 1