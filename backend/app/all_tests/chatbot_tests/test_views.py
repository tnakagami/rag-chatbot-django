import pytest
import json
from rest_framework import status
from django.urls import reverse
from chatbot.models.agents import AgentType, ToolType
from urllib.parse import urlencode
from chatbot.models import (
  Agent,
  Embedding,
  Tool,
  Assistant,
  DocumentFile,
  Thread,
)
from . import factories

def ids(value):
  return f'{value}'

@pytest.fixture
def login_process(client, django_db_blocker):
  with django_db_blocker.unblock():
    user = factories.UserFactory()
    client.force_login(user)

  return client, user

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_jwt_get_access(login_process):
  client, _ = login_process
  url = reverse('chatbot:token')
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'token' in output.keys()

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_invalid_access_for_jwt(client):
  url = reverse('chatbot:token')
  response = client.get(url)
  expected = '{}?next={}'.format(reverse('account:login'), url)

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == expected

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_index_get_access(login_process):
  client, _ = login_process
  url = reverse('chatbot:index')
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_get_task_list(login_process):
  client, user = login_process
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  for task in factories.TaskResultFactory.create_batch(3):
    assistant.set_task_result(task.task_id, user.pk)
  url = reverse('chatbot:tasks')
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_settings_get_access(login_process):
  client, user = login_process
  agents = factories.AgentFactory.create_batch(3, user=user)
  embs = factories.EmbeddingFactory.create_batch(4, user=user)
  tools = factories.ToolFactory.create_batch(7, user=user)
  url = reverse('chatbot:settings')
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
@pytest.mark.parametrize('target', ['agent', 'embedding'])
def test_check_set_no_type_id_using_view(login_process, target):
  client, _ = login_process
  url = reverse(f'chatbot:collect_{target}_config')
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'config_html_form' in output.keys()
  assert 'config_exists' in output.keys()
  assert not output['config_exists']

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_collect_agent_config_using_view(login_process, get_target_agent_type):
  client, _ = login_process
  _, type_id = get_target_agent_type
  exact_fields = AgentType.get_llm_fields(type_id, {}, is_embedded=False)
  expected_keys = [item.get_key() for item in exact_fields]
  url = '{}?{}'.format(reverse('chatbot:collect_agent_config'), urlencode({'type_id': type_id}))
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'config_html_form' in output.keys()
  assert 'config_exists' in output.keys()
  assert all([key in output['config_html_form'] for key in expected_keys])
  assert output['config_exists']

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_collect_embedding_config_using_view(login_process, get_target_embedding_type):
  client, _ = login_process
  _, type_id = get_target_embedding_type
  exact_fields = AgentType.get_llm_fields(type_id, {}, is_embedded=True)
  expected_keys = [item.get_key() for item in exact_fields]
  url = '{}?{}'.format(reverse('chatbot:collect_embedding_config'), urlencode({'type_id': type_id}))
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'config_html_form' in output.keys()
  assert 'config_exists' in output.keys()
  assert all([key in output['config_html_form'] for key in expected_keys])
  assert output['config_exists']

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_collect_tool_config_with_its_config_using_view(login_process, get_target_tool_type_with_config):
  client, _ = login_process
  _, type_id = get_target_tool_type_with_config
  exact_fields = ToolType.get_config_field(type_id, {})
  expected_keys = [item.get_key() for item in exact_fields]
  url = '{}?{}'.format(reverse('chatbot:collect_tool_config'), urlencode({'type_id': type_id}))
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'config_html_form' in output.keys()
  assert 'config_exists' in output.keys()
  assert all([key in output['config_html_form'] for key in expected_keys])
  assert output['config_exists']

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_collect_tool_config_without_its_config_using_view(login_process, get_target_tool_type_without_config):
  client, _ = login_process
  _, type_id = get_target_tool_type_without_config
  exact_fields = ToolType.get_config_field(type_id, {})
  expected_keys = [item.get_key() for item in exact_fields]
  url = '{}?{}'.format(reverse('chatbot:collect_tool_config'), urlencode({'type_id': type_id}))
  response = client.get(url)
  output = response.json()

  assert response.status_code == status.HTTP_200_OK
  assert 'config_html_form' in output.keys()
  assert 'config_exists' in output.keys()
  assert all([key in output['config_html_form'] for key in expected_keys])
  assert not output['config_exists']

# =========
# = Agent =
# =========
@pytest.fixture
def get_agent_kwargs():
  link = 'agent'
  form_data = {
    'name': 'sample-agent',
    'agent_type': AgentType.AZURE.value,
    'config': json.dumps({'model': 'sample-agent'}),
  }
  model_class = Agent
  factory_class = factories.AgentFactory

  return link, form_data, model_class, factory_class

@pytest.fixture
def get_embedding_kwargs():
  link = 'embedding'
  form_data = {
    'name': 'sample-embedding',
    'emb_type': AgentType.BEDROCK.value,
    'distance_strategy': Embedding.DistanceType.COSINE.value,
    'config': json.dumps({'model': 'sample-embedding'}),
  }
  model_class = Embedding
  factory_class = factories.EmbeddingFactory

  return link, form_data, model_class, factory_class

@pytest.fixture
def get_tool_kwargs():
  link = 'tool'
  form_data = {
    'name': 'sample-tool',
    'tool_type': ToolType.RETRIEVER.value,
    'config': json.dumps({'k': 5}),
  }
  model_class = Tool
  factory_class = factories.ToolFactory

  return link, form_data, model_class, factory_class

@pytest.fixture(params=['agent', 'embedding', 'tool'])
def get_agent_embedding_tool_kwargs(request):
  if request.param == 'agent':
    link, form_data, model_class, factory_class = request.getfixturevalue('get_agent_kwargs')
  elif request.param == 'embedding':
    link, form_data, model_class, factory_class = request.getfixturevalue('get_embedding_kwargs')
  elif request.param == 'tool':
    link, form_data, model_class, factory_class = request.getfixturevalue('get_tool_kwargs')

  return link, form_data, model_class, factory_class

# =========================
# = Valid tests of config =
# =========================
@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
@pytest.mark.parametrize('link', ['agent', 'embedding', 'tool'], ids=lambda val: f'create-{val}-of-get-method')
def test_check_get_method_of_create_using_view(login_process, link):
  client, user = login_process
  url = reverse(f'chatbot:create_{link}')
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_create_using_view(login_process, get_agent_embedding_tool_kwargs):
  client, user = login_process
  link, form_data, model_class, _ = get_agent_embedding_tool_kwargs
  url = reverse(f'chatbot:create_{link}')
  response = client.post(url, data=urlencode(form_data), content_type='application/x-www-form-urlencoded')
  total = model_class.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:settings')
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_updation_using_view(login_process, get_agent_embedding_tool_kwargs):
  client, user = login_process
  link, form_data, model_class, factory_class = get_agent_embedding_tool_kwargs
  target = factory_class(user=user)
  pk = target.pk
  url = reverse(f'chatbot:update_{link}', kwargs={'pk': pk})
  response = client.post(url, data=urlencode(form_data), content_type='application/x-www-form-urlencoded')
  instance = model_class.objects.get_or_none(pk=pk)
  total = model_class.objects.all().count()
  _config = json.loads(form_data.pop('config'))

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:settings')
  assert all([getattr(instance, key) == value for key, value in form_data.items()])
  assert all([instance.config[key] == value for key, value in _config.items()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_deletion_using_view(login_process, get_agent_embedding_tool_kwargs):
  client, user = login_process
  link, form_data, model_class, factory_class = get_agent_embedding_tool_kwargs
  target = factory_class(user=user)
  pk = target.pk
  url = reverse(f'chatbot:delete_{link}', kwargs={'pk': pk})
  response = client.post(url, data=urlencode(form_data), content_type='application/x-www-form-urlencoded')
  total = model_class.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:settings')
  assert total == 0

# ==========================
# = Inalid tests of config =
# ==========================
@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_updation_using_view(login_process, get_agent_embedding_tool_kwargs):
  client, _ = login_process
  other = factories.UserFactory()
  link, form_data, model_class, factory_class = get_agent_embedding_tool_kwargs
  target = factory_class(user=other)
  pk = target.pk
  url = reverse(f'chatbot:update_{link}', kwargs={'pk': pk})
  response = client.post(url, data=urlencode(form_data), content_type='application/x-www-form-urlencoded')
  instance = model_class.objects.get_or_none(pk=pk)
  total = model_class.objects.all().count()
  _ = form_data.pop('config')

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert all([getattr(instance, key) == getattr(target, key) for key in form_data.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_deletion_using_view(login_process, get_agent_embedding_tool_kwargs):
  client, _ = login_process
  other = factories.UserFactory()
  link, form_data, model_class, factory_class = get_agent_embedding_tool_kwargs
  target = factory_class(user=other)
  pk = target.pk
  url = reverse(f'chatbot:delete_{link}', kwargs={'pk': pk})
  response = client.post(url, data=urlencode(form_data), content_type='application/x-www-form-urlencoded')
  instance = model_class.objects.get_or_none(pk=pk)
  total = model_class.objects.all().count()
  _ = form_data.pop('config')

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert all([getattr(instance, key) == getattr(target, key) for key in form_data.keys()])
  assert total == 1

# =============
# = Assistant =
# =============
@pytest.fixture
def get_assistant_kwargs(login_process, django_db_blocker):
  with django_db_blocker.unblock():
    client, user = login_process
    agent = factories.AgentFactory(user=user)
    embedding = factories.EmbeddingFactory(user=user)
    tools = factories.ToolFactory.create_batch(2, user=user)
    form_data = {
      'name': 'sample-assistant',
      'agent': agent.pk,
      'embedding': embedding.pk,
      'tools': [target.pk for target in tools],
    }

  return client, user, form_data

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_get_method_of_assistant_creation_using_view(login_process):
  client, _ = login_process
  url = reverse('chatbot:create_assistant')
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_assistant_creation_using_view(get_assistant_kwargs):
  client, user, form_data = get_assistant_kwargs
  url = reverse('chatbot:create_assistant')
  response = client.post(url, data=form_data)
  total = Assistant.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:index')
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_assistant_updation_using_view(get_assistant_kwargs):
  client, user, form_data = get_assistant_kwargs
  target = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  pk = target.pk
  url = reverse('chatbot:update_assistant', kwargs={'pk': pk})
  response = client.post(url, data=form_data)
  instance = Assistant.objects.get_or_none(pk=pk)
  total = Assistant.objects.all().count()
  expected_tools = form_data.pop('tools')
  out_tools = instance.tools.all().values_list('pk', flat=True)

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:index')
  assert instance.name == form_data['name']
  assert instance.agent.pk == form_data['agent']
  assert instance.embedding.pk == form_data['embedding']
  assert all([pk in expected_tools for pk in out_tools])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_get_method_of_specific_assistant_using_view(login_process):
  client, user = login_process
  target = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  pk = target.pk
  url = reverse('chatbot:detail_assistant', kwargs={'pk': pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_assistant_deletion_using_view(get_assistant_kwargs):
  client, user, _ = get_assistant_kwargs
  target = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  pk = target.pk
  url = reverse('chatbot:delete_assistant', kwargs={'pk': pk})
  response = client.post(url)
  total = Assistant.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:index')
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
@pytest.mark.parametrize('pattern', [
  {'agent': 'other', 'embedding': 'owner', 'tools': 'owner'},
  {'agent': 'owner', 'embedding': 'other', 'tools': 'owner'},
  {'agent': 'owner', 'embedding': 'owner', 'tools': 'other'},
], ids=['invalid-agent', 'invalid-embedding', 'invalid-tools'])
def test_check_invalid_post_request_of_assistant_creation_using_view(login_process, pattern):
  client, user = login_process
  other = factories.UserFactory()
  kwargs = {}
  # Create instances
  for owner, key in zip([user, other], ['owner', 'other']):
    agent = agent=factories.AgentFactory(user=owner)
    embedding = factories.EmbeddingFactory(user=owner)
    tools = factories.ToolFactory.create_batch(2, user=owner)

    kwargs[key] = {
      'agent': agent.pk,
      'embedding': embedding.pk,
      'tools': [target.pk for target in tools],
    }
  # Set form data
  form_data = {
    key: kwargs[target][key]
    for key, target in pattern.items()
  }
  form_data.update({'name': 'sample-assistant'})
  url = reverse('chatbot:create_assistant')
  response = client.post(url, data=form_data)
  total = Assistant.objects.all().count()

  assert response.status_code == status.HTTP_200_OK
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_assistant_updation_using_view(get_assistant_kwargs):
  client, user, form_data = get_assistant_kwargs
  other = factories.UserFactory()
  target = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
    tools=tuple(factories.ToolFactory.create_batch(1, user=other)),
  )
  pk = target.pk
  url = reverse('chatbot:update_assistant', kwargs={'pk': pk})
  response = client.post(url, data=form_data)
  instance = Assistant.objects.get_or_none(pk=pk)
  total = Assistant.objects.all().count()

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert all([getattr(instance, key) == getattr(target, key) for key in form_data.keys()])
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_get_request_of_specific_assistant_using_view(login_process):
  client, _ = login_process
  other = factories.UserFactory()
  target = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
    tools=tuple(factories.ToolFactory.create_batch(1, user=other)),
  )
  pk = target.pk
  url = reverse('chatbot:detail_assistant', kwargs={'pk': pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_assistant_deletion_using_view(get_assistant_kwargs):
  client, user, _ = get_assistant_kwargs
  other = factories.UserFactory()
  target = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
    tools=tuple(factories.ToolFactory.create_batch(1, user=other)),
  )
  pk = target.pk
  url = reverse('chatbot:delete_assistant', kwargs={'pk': pk})
  response = client.post(url)
  total = Assistant.objects.all().count()

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert total == 1

# ================
# = DocumentFile =
# ================
@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_get_method_of_docfile_creation_using_view(login_process):
  client, user = login_process
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  url = reverse('chatbot:create_docfile', kwargs={'assistant_pk': assistant.pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_docfile_creation_using_view(login_process):
  client, user = login_process
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  url = reverse('chatbot:create_docfile', kwargs={'assistant_pk': assistant.pk})
  response = client.post(url)
  total = DocumentFile.objects.all().count()

  assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_docfile_deletion_using_view(login_process):
  client, user = login_process
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
    tools=tuple(factories.ToolFactory.create_batch(1, user=user)),
  )
  target = factories.DocumentFileFactory(assistant=assistant)
  pk = target.pk
  url = reverse('chatbot:delete_docfile', kwargs={'pk': pk})
  response = client.post(url)
  total = DocumentFile.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:index')
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_docfile_deletion_using_view(login_process):
  client, user = login_process
  other = factories.UserFactory()
  assistant = factories.AssistantFactory(
    user=other,
    agent=factories.AgentFactory(user=other),
    embedding=factories.EmbeddingFactory(user=other),
    tools=tuple(factories.ToolFactory.create_batch(1, user=other)),
  )
  target = factories.DocumentFileFactory(assistant=assistant)
  url = reverse('chatbot:delete_docfile', kwargs={'pk': target.pk})
  response = client.post(url)
  total = DocumentFile.objects.all().count()

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert total == 1

# ==========
# = Thread =
# ==========
@pytest.fixture
def get_thread_kwargs(login_process):
  client, user = login_process
  other = factories.UserFactory()
  kwargs = {}

  for owner, key in zip([user, other], ['owner', 'other']):
    agent = agent=factories.AgentFactory(user=owner)
    embedding = factories.EmbeddingFactory(user=owner)
    tools = factories.ToolFactory.create_batch(2, user=owner)
    assistant = factories.AssistantFactory(
      user=owner,
      agent=agent,
      embedding=embedding,
      tools=tuple(tools),
    )

    kwargs[key] = {
      'user': owner,
      'agent': agent,
      'embedding': embedding,
      'tools': tools,
      'assistant': assistant,
      'docfiles': factories.DocumentFileFactory.create_batch(3, assistant=assistant),
    }

  return client, kwargs

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_get_method_of_thread_creation_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['owner']['assistant']
  url = reverse('chatbot:create_thread', kwargs={'assistant_pk': assistant.pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_thread_creation_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['owner']['assistant']
  docfiles = kwargs['owner']['docfiles']
  form_data = {
    'name': 'sample-thread',
    'docfiles': [_docfile.pk for _docfile in docfiles],
  }
  pk = assistant.pk
  url = reverse('chatbot:create_thread', kwargs={'assistant_pk': pk})
  response = client.post(url, data=form_data)
  total = Thread.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:detail_assistant', kwargs={'pk': pk})
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_get_method_of_specific_thread_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['owner']['assistant']
  target = factories.ThreadFactory(assistant=assistant)
  pk = target.pk
  url = reverse('chatbot:detail_thread', kwargs={'pk': pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_200_OK

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_thread_updation_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['owner']['assistant']
  docfiles = kwargs['owner']['docfiles']
  target = factories.ThreadFactory(assistant=assistant)
  pk = target.pk
  form_data = {
    'name': 'replaced-thread',
    'docfiles': [_docfile.pk for _docfile in docfiles],
  }
  url = reverse('chatbot:update_thread', kwargs={'pk': pk})
  response = client.post(url, data=form_data)
  instance = Thread.objects.get_or_none(pk=pk)
  total = Thread.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert instance.name == form_data['name']
  assert all([_docfile.pk in form_data['docfiles'] for _docfile in instance.docfiles.all()])
  assert response['Location'] == reverse('chatbot:detail_assistant', kwargs={'pk': assistant.pk})
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_post_method_of_thread_deletion_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['owner']['assistant']
  docfiles = kwargs['owner']['docfiles']
  target = factories.ThreadFactory(assistant=assistant, docfiles=tuple(docfiles))
  pk = target.pk
  url = reverse('chatbot:delete_thread', kwargs={'pk': pk})
  response = client.post(url)
  total = Thread.objects.all().count()

  assert response.status_code == status.HTTP_302_FOUND
  assert response['Location'] == reverse('chatbot:detail_assistant', kwargs={'pk': assistant.pk})
  assert total == 0

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_get_request_of_specific_thread_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['other']['assistant']
  target = factories.ThreadFactory(assistant=assistant)
  pk = target.pk
  url = reverse('chatbot:detail_thread', kwargs={'pk': pk})
  response = client.get(url)

  assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_thread_updation_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['other']['assistant']
  target = factories.ThreadFactory(assistant=assistant)
  pk = target.pk
  form_data = {
    'name': 'replaced-thread',
  }
  url = reverse('chatbot:update_thread', kwargs={'pk': pk})
  response = client.post(url, data=form_data)

  assert response.status_code == status.HTTP_403_FORBIDDEN

@pytest.mark.chatbot
@pytest.mark.view
@pytest.mark.django_db
def test_check_invalid_post_request_of_thread_deletion_using_view(get_thread_kwargs):
  client, kwargs = get_thread_kwargs
  assistant = kwargs['other']['assistant']
  docfiles = kwargs['other']['docfiles']
  target = factories.ThreadFactory(assistant=assistant, docfiles=tuple(docfiles))
  pk = target.pk
  url = reverse('chatbot:delete_thread', kwargs={'pk': pk})
  response = client.post(url)
  total = Thread.objects.all().count()

  assert response.status_code == status.HTTP_403_FORBIDDEN
  assert total == 1