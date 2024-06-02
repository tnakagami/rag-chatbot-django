import pytest
from chatbot import forms
# For test
from chatbot import models
from chatbot.models.agents import AgentType, ToolType
from . import factories

_max_agent_id = max([value for value, _ in AgentType.choices])
_max_tool_id = max([value for value, _ in ToolType.choices])

def ids(value):
  return f'{value}'

@pytest.fixture
def get_form_kwargs():
  user = factories.UserFactory()
  url = 'http://dummy.com'
  kwargs = {
    'user': user,
    'config_form_url': url, 
  }

  return kwargs

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_base_model_form(get_form_kwargs):
  class DummyModelForm(forms._BaseModelForm):
    class Meta:
      model = models.Agent
      fields = ('name', 'agent_type', 'config')

  val = AgentType.OPENAI.value
  kwargs = get_form_kwargs
  user = kwargs['user']
  agent = factories.AgentFactory(user=user, agent_type=val, config={})
  params = {
    'name': 'test',
    'agent_type': val,
  }
  form = DummyModelForm(data=params, user=user)
  instance = form.customize(agent)

  assert isinstance(instance, type(agent))

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_base_config_form():
  val = AgentType.OPENAI.value
  form = forms._BaseConfigForm(type_id=val)

  assert not form.config_exists

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_valid_agent_type_for_agent_form(get_form_kwargs, get_target_agent_type):
  kwargs = get_form_kwargs
  _, val = get_target_agent_type
  params = {
    'name': 'test',
    'agent_type': val,
  }
  form = forms.AgentForm(data=params, **kwargs)
  instance = form.save()
  total = models.Agent.objects.all().count()

  assert form.is_valid()
  assert len(instance.config) > 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_customize_method_of_agent_form(get_form_kwargs, get_target_agent_type):
  kwargs = get_form_kwargs
  _, val = get_target_agent_type
  agent = factories.AgentFactory(user=kwargs['user'], agent_type=val, config={})
  params = {
    'name': 'test',
    'agent_type': val,
  }
  form = forms.AgentForm(data=params, **kwargs)
  instance = form.customize(agent)

  assert len(instance.config) > 0

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('form_param',[
  {},
  {'name': 'test', 'agent_type': -1},
  {'name': 'test', 'agent_type': _max_agent_id + 1},
  {'name': 'a'*256},
], ids=['is-empty', 'invald-agent_type', 'missmatch-agent_type', 'too-long-name'])
def test_check_invalid_parameters_of_agent_form(get_form_kwargs, form_param):
  kwargs = get_form_kwargs
  form = forms.AgentForm(data=form_param, **kwargs)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
def test_check_agent_config_form(get_target_agent_type):
  _, val = get_target_agent_type
  form = forms.AgentConfigForm(type_id=val)

  assert form.config_exists

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.parametrize('type_id,config,expect', [
  (None, None, False),
  (None, {}, False),
], ids=['invalid-typeID-config', 'invalid-typeID'])
def test_check_invalid_arguments_of_agent_config_form(type_id, config, expect):
  form = forms.AgentConfigForm(type_id=type_id, config=config)

  assert form.config_exists == expect

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_valid_embedding_form(get_form_kwargs, get_target_embedding_type):
  kwargs = get_form_kwargs
  _, val = get_target_embedding_type
  params = {
    'name': 'test',
    'distance_strategy': models.Embedding.DistanceType.EUCLIDEAN,
    'emb_type': val,
  }
  form = forms.EmbeddingForm(data=params, **kwargs)
  instance = form.save()
  total = models.Embedding.objects.all().count()

  assert form.is_valid()
  assert len(instance.config) > 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_customize_method_of_embedding_form(get_form_kwargs, get_target_embedding_type):
  kwargs = get_form_kwargs
  _, val = get_target_embedding_type
  embedding = factories.EmbeddingFactory(user=kwargs['user'], emb_type=val, config={})
  params = {
    'name': 'test',
    'distance_strategy': models.Embedding.DistanceType.EUCLIDEAN,
    'emb_type': val,
  }
  form = forms.EmbeddingForm(data=params, **kwargs)
  instance = form.customize(embedding)

  assert len(instance.config) > 0

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('form_param', [
  {},
  {'name': 'test', 'emb_type': -1},
  {'name': 'test', 'emb_type': _max_agent_id + 1},
  {'name': 'a'*256},
], ids=['is-empty', 'invald-emb_type', 'missmatch-emb_type', 'too-long-name'])
def test_check_invalid_parameters_of_embedding_form(get_form_kwargs, form_param):
  kwargs = get_form_kwargs
  form = forms.EmbeddingForm(data=form_param, **kwargs)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_invalid_function_of_anthropic(get_form_kwargs):
  kwargs = get_form_kwargs
  form_param = {
    'name': 'test',
    'emb_type': AgentType.ANTHROPIC.value,
  }
  form = forms.EmbeddingForm(data=form_param, **kwargs)

  with pytest.raises(ValueError) as ex:
    _ = form.save()
  
  assert 'Embedding could not be created' in str(ex.value)

@pytest.mark.chatbot
@pytest.mark.form
def test_check_embedding_config_form(get_target_embedding_type):
  _, val = get_target_embedding_type
  form = forms.EmbeddingConfigForm(type_id=val)

  assert form.config_exists

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.parametrize('type_id,config,expect', [
  (None, None, False),
  (None, {}, False),
], ids=['invalid-typeID-config', 'invalid-typeID'])
def test_check_invalid_arguments_of_embedding_config_form(type_id, config, expect):
  form = forms.EmbeddingConfigForm(type_id=type_id, config=config)

  assert form.config_exists == expect

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_valid_tool_type_with_config_for_tool_form(get_form_kwargs, get_target_tool_type_with_config):
  kwargs = get_form_kwargs
  _, val = get_target_tool_type_with_config
  params = {
    'name': 'test',
    'tool_type': val,
  }
  form = forms.ToolForm(data=params, **kwargs)
  instance = form.save()
  total = models.Tool.objects.all().count()

  assert form.is_valid()
  assert len(instance.config) > 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_valid_tool_type_without_config_for_tool_form(get_form_kwargs, get_target_tool_type_without_config):
  kwargs = get_form_kwargs
  _, val = get_target_tool_type_without_config
  params = {
    'name': 'test',
    'tool_type': val,
  }
  form = forms.ToolForm(data=params, **kwargs)
  instance = form.save()
  total = models.Tool.objects.all().count()

  assert form.is_valid()
  assert len(instance.config) == 0
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_customize_method_with_config_of_tool_form(get_form_kwargs, get_target_tool_type_with_config):
  kwargs = get_form_kwargs
  _, val = get_target_tool_type_with_config
  _tool = factories.ToolFactory(user=kwargs['user'], tool_type=val, config={})
  params = {
    'name': 'test',
    'tool_type': val,
  }
  form = forms.ToolForm(data=params, **kwargs)
  instance = form.customize(_tool)

  assert len(instance.config) > 0

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_customize_method_without_config_of_tool_form(get_form_kwargs, get_target_tool_type_without_config):
  kwargs = get_form_kwargs
  _, val = get_target_tool_type_without_config
  _tool = factories.ToolFactory(user=kwargs['user'], tool_type=val, config={})
  params = {
    'name': 'test',
    'tool_type': val,
  }
  form = forms.ToolForm(data=params, **kwargs)
  instance = form.customize(_tool)

  assert len(instance.config) == 0
  assert instance.config == _tool.config

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('form_param', [
  {},
  {'name': 'test', 'tool_type': -1},
  {'name': 'test', 'tool_type': _max_tool_id + 1},
  {'name': 'a'*256},
], ids=['is-empty', 'invald-tool_type', 'missmatch-tool_type', 'too-long-name'])
def test_check_invalid_parameters_of_tool_form(get_form_kwargs, form_param):
  kwargs = get_form_kwargs
  form = forms.ToolForm(data=form_param, **kwargs)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
def test_check_tool_config_form_with_config(get_target_tool_type_with_config):
  _, val = get_target_tool_type_with_config
  form = forms.ToolConfigForm(type_id=val)

  assert form.config_exists

@pytest.mark.chatbot
@pytest.mark.form
def test_check_tool_config_form_without_config(get_target_tool_type_without_config):
  _, val = get_target_tool_type_without_config
  form = forms.ToolConfigForm(type_id=val)

  assert not form.config_exists

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('input_param,tool_count',[
  ({'name': 'test', 'system_message': 'sample', 'is_interrupt': True}, 2),
  ({'name': 'test', 'is_interrupt': False}, 0),
  ({'name': 'a'*255}, 1),
], ids=['valid-pattern', 'is-default-pattern', 'max-length-pattern'])
def test_check_valid_input_for_assistant_form(input_param, tool_count):
  user = factories.UserFactory()
  agent = factories.AgentFactory(user=user)
  embedding = factories.EmbeddingFactory(user=user)
  tools = [_tool.pk for _tool in factories.ToolFactory.create_batch(tool_count, user=user)]

  form_param = {
    'agent': agent.pk,
    'embedding': embedding.pk,
    'tools': tools,
  }
  form_param.update(input_param)
  form = forms.AssistantForm(data=form_param, user=user)
  is_valid = form.is_valid()
  instance = form.save()
  total = models.Assistant.objects.all().count()

  assert is_valid
  assert total == 1
  assert instance.is_interrupt == input_param.get('is_interrupt', False)

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('input_param,item_list',[
  ({'system_message': 'sample'}, ['agent', 'embedding']),
  ({'name': 'a'*256, 'system_message': 'sample'}, ['agent', 'embedding']),
  ({'name': 'test'}, ['embedding']),
  ({'name': 'test'}, ['agent']),
], ids=['name-is-empty', 'invalid-name-length', 'agent-is-empty', 'embedding-is-empty'])
def test_check_invalid_input_for_assistant_form(input_param, item_list):
  callback = {
    'agent': lambda _user: factories.AgentFactory(user=_user),
    'embedding': lambda _user: factories.EmbeddingFactory(user=_user),
  }
  user = factories.UserFactory()
  form_param = {}

  for key in item_list:
    func = callback[key]
    form_param.update({key: func(user)})
  form_param.update(input_param)
  form = forms.AssistantForm(data=form_param, user=user)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_invalid_tool_pks_for_assistant_form():
  user = factories.UserFactory()
  tools = factories.ToolFactory.create_batch(2, user=user)
  max_pk = models.Tool.objects.order_by('-pk').first().pk
  form_param = {
    'name': 'test',
    'agent': factories.AgentFactory(user=user),
    'embedding': factories.EmbeddingFactory(user=user),
    'tools': [max_pk + 1],
  }
  form = forms.AssistantForm(data=form_param, user=user)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('pattern,name',[
  ({'agent': 'other', 'embedding': 'own',   'tool': 'own'},   'invalid-agent'),
  ({'agent': 'own',   'embedding': 'other', 'tool': 'own'},   'invalid-embedding'),
  ({'agent': 'own',   'embedding': 'own',   'tool': 'other'}, 'invalid-tool'),
], ids=['invalid-agent', 'invalid-embedding', 'invalid-tool'])
def test_check_not_owner_input_for_assistant_form(pattern, name):
  users, agents, embeddings, tools = {}, {}, {}, {}

  for key in ['own', 'other']:
    _user = factories.UserFactory()
    users[key] = _user
    agents[key] = factories.AgentFactory(user=_user)
    embeddings[key] = factories.EmbeddingFactory(user=_user)
    tools[key] = factories.ToolFactory.create_batch(2, user=_user)

  form_param = {
    'agent': agents[pattern['agent']].pk,
    'embedding': embeddings[pattern['embedding']].pk,
    'tools': [item.pk for item in tools[pattern['tool']]],
  }
  form = forms.AssistantForm(data=form_param, user=users['own'])

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('name,docfile_count',[
  ('test', 2),
  ('test', 1),
  ('a'*255, 0),
], ids=['valid-pattern-docfile-is-multiple', 'docfile-is-single', 'max-length-pattern'])
def test_check_valid_input_for_thread_form(name, docfile_count):
  assistant = factories.AssistantFactory()
  docfiles = [_docfile.pk for _docfile in factories.DocumentFileFactory.create_batch(docfile_count, assistant=assistant)]
  form_param = {
    'name': name,
    'docfiles': docfiles,
  }
  form = forms.ThreadForm(data=form_param, assistant=assistant)
  is_valid = form.is_valid()
  form.save()
  total = models.Thread.objects.all().count()

  assert is_valid
  assert total == 1

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
@pytest.mark.parametrize('form_param',[
  {'name': 'a'*256},
  {},
], ids=['invalid-name-length', 'name-is-empty'])
def test_check_invalid_input_for_thread_form(form_param):
  assistant = factories.AssistantFactory()
  form = forms.ThreadForm(data=form_param, assistant=assistant)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_invalid_docfiles_for_thread_form():
  assistant = factories.AssistantFactory()
  other = factories.AssistantFactory()
  docfiles =   factories.DocumentFileFactory.create_batch(2, assistant=assistant) \
             + factories.DocumentFileFactory.create_batch(1, assistant=factories.AssistantFactory())
  form_param = {
    'name': 'test',
    'docfiles': [_docfile.pk for _docfile in docfiles],
  }
  form = forms.ThreadForm(data=form_param, assistant=assistant)

  assert not form.is_valid()

@pytest.mark.chatbot
@pytest.mark.form
@pytest.mark.django_db
def test_check_no_assistants_exist_for_thread_form():
  form_param = {
    'name': 'test',
  }
  form = forms.ThreadForm(data=form_param, assistant=None)

  assert not form.is_valid()