import pytest
from chatbot.models.utils import llms
from dataclasses import asdict
# Classes for comparing instance
import httpx
## OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
## Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
## Anthropic and Fireworks
import anthropic
from chatbot.models.utils._customLLMWrapper import CustomChatAnthropic, CustomChatFireworks, CustomFireworksEmbeddings
from fireworks.client.chat_completion import ChatCompletionV2
## Bedrock
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
## Ollma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
## Google
from typing import Mapping
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings

@pytest.fixture
def fileds_checker_wrapper(check_llm_fields, check_embedding_fields):
  _base_llm_checker = check_llm_fields
  _base_embedding_checker = check_embedding_fields

  def wrapper(callback):
    def llm_checker(llm_fields, expected):
      return all([
        _base_llm_checker(llm_fields, expected),
        callback(llm_fields, expected),
      ])

    def embedding_checker(embedinng_fields, expected):
      return all([
        _base_embedding_checker(embedinng_fields, expected),
        callback(embedinng_fields, expected),
      ])

    return llm_checker, embedding_checker

  return wrapper

# ============
# = _BaseLLM =
# ============
@pytest.fixture
def get_basellm_arg():
  kwargs = {
    'model': 'sample',
    'temperature': 2,
    'stream': True,
    'max_retries': 3,
  }
  expected = {key: val for key, val in kwargs.items()}

  return kwargs, expected

@pytest.fixture
def get_basellm_arg_with_proxy(get_basellm_arg):
  specific = {
    'proxy': 'http://proxy.com:12345/hoge',
  }
  kwargs, expected = get_basellm_arg
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected

@pytest.fixture(params=['with_proxy', 'without_proxy'])
def basellm_context(request):
  if request.param == 'with_proxy':
    kwargs, expected = request.getfixturevalue('get_basellm_arg_with_proxy')
    sync_client = httpx.Client
    async_client = httpx.AsyncClient
  elif request.param == 'without_proxy':
    kwargs, expected = request.getfixturevalue('get_basellm_arg')
    sync_client = type(None)
    async_client = type(None)

  return kwargs, expected, sync_client, async_client

@pytest.mark.chatbot
@pytest.mark.util
def test_check_default_basellm_arg():
  expected = {
    'model': '',
    'temperature': 0,
    'stream': False,
    'max_retries': 10,
  }
  instance = llms._BaseLLM()

  assert instance.model == expected['model']
  assert instance.temperature == expected['temperature']
  assert instance.stream == expected['stream']
  assert instance.max_retries == expected['max_retries']
  assert instance.proxy is None

@pytest.mark.chatbot
@pytest.mark.util
def test_check_basellm_args(get_basellm_arg_with_proxy):
  kwargs, expected = get_basellm_arg_with_proxy
  instance = llms._BaseLLM(**kwargs)

  assert instance.model == expected['model']
  assert instance.temperature == expected['temperature']
  assert instance.stream == expected['stream']
  assert instance.max_retries == expected['max_retries']
  assert instance.proxy is not None

@pytest.mark.chatbot
@pytest.mark.util
def test_check_basellm_methods(get_basellm_arg_with_proxy, check_llm_fields, check_embedding_fields):
  kwargs, expected = get_basellm_arg_with_proxy
  ignore_keys = ['temperature', 'stream']
  instance = llms._BaseLLM(**kwargs)
  llm_fields = instance.get_fields(instance, is_embedded=False)
  embedinng_fields = instance.get_fields(instance, is_embedded=True)

  with pytest.raises(NotImplementedError):
    instance.get_llm()
  assert check_llm_fields(llm_fields, expected)
  assert check_embedding_fields(embedinng_fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.parametrize('ignore_keys', [
  ['model'],
  ['model', 'temperature'],
  ['model', 'temperature', 'stream'],
  ['model', 'temperature', 'stream', 'max_retries'],
  ['model', 'temperature', 'stream', 'max_retries', 'proxy'],
])
def test_check_basellm_delete_keys(get_basellm_arg_with_proxy, ignore_keys):
  kwargs, expected = get_basellm_arg_with_proxy
  instance = llms._BaseLLM(**kwargs)
  rests = instance.delete_keys(asdict(instance), ignore_keys)

  assert all([ignore_key not in rests.keys() for ignore_key in ignore_keys])
  for key in rests.keys():
    assert rests[key] == expected[key]

# =============
# = OpenAILLM =
# =============
@pytest.fixture
def get_openai_context(fileds_checker_wrapper, basellm_context):
  specific = {
    'api_key': 'open-ai-key',
    'endpoint': 'http://dummy-open-ai/endpoint',
  }
  def _check_specific(fields, expected):
    return all([
      fields['api_key'] == expected['api_key'],
      fields['endpoint'] == expected['endpoint'],
    ])
  # Setup
  wrapper = fileds_checker_wrapper
  kwargs, expected, sync_client, async_client = basellm_context
  llm_checker, embedding_checker = wrapper(_check_specific)
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, llm_checker, embedding_checker, sync_client, async_client

@pytest.mark.chatbot
@pytest.mark.util
def test_check_openai_chatbot(get_openai_context):
  kwargs, expected, llm_checker, _, http_client, http_async_client = get_openai_context
  is_embedded = False
  instance = llms.OpenAILLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, ChatOpenAI)
  assert llm.model_name == expected['model']
  assert llm.openai_api_key.get_secret_value() == expected['api_key']
  assert llm.openai_api_base == expected['endpoint']
  assert llm.temperature == expected['temperature']
  assert llm.streaming == expected['stream']
  assert llm.max_retries == expected['max_retries']
  assert isinstance(llm.http_client, http_client)
  assert isinstance(llm.http_async_client, http_async_client)
  assert llm_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_openai_embedding(get_openai_context):
  kwargs, expected, _, embedding_checker, http_client, http_async_client = get_openai_context
  is_embedded = True
  instance = llms.OpenAILLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, OpenAIEmbeddings)
  assert llm.model == expected['model']
  assert llm.openai_api_key.get_secret_value() == expected['api_key']
  assert llm.openai_api_base == expected['endpoint']
  assert llm.max_retries == expected['max_retries']
  assert isinstance(llm.http_client, http_client)
  assert isinstance(llm.http_async_client, http_async_client)
  assert embedding_checker(fields, expected)

# ==================
# = AzureOpenAILLM =
# ==================
@pytest.fixture
def get_azure_openai_context(fileds_checker_wrapper, basellm_context):
  specific = {
    'api_key': 'azure-open-ai-key',
    'endpoint': 'http://dummy-azure-open-ai/endpoint',
    'version': 'dummy-version',
    'deployment': 'http://dummy-azure-open-ai/deployment',
  }
  def _check_specific(fields, expected):
    return all([
      fields['api_key'] == expected['api_key'],
      fields['endpoint'] == expected['endpoint'],
      fields['version'] == expected['version'],
      fields['deployment'] == expected['deployment'],
    ])
  # Setup
  wrapper = fileds_checker_wrapper
  kwargs, expected, sync_client, async_client = basellm_context
  llm_checker, embedding_checker = wrapper(_check_specific)
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, llm_checker, embedding_checker, sync_client, async_client

@pytest.mark.chatbot
@pytest.mark.util
def test_check_azure_openai_chatbot(get_azure_openai_context):
  kwargs, expected, llm_checker, _, http_client, http_async_client = get_azure_openai_context
  is_embedded = False
  instance = llms.AzureOpenAILLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, AzureChatOpenAI)
  assert llm.model_name == expected['model']
  assert llm.openai_api_key.get_secret_value() == expected['api_key']
  assert llm.azure_endpoint == expected['endpoint']
  assert llm.openai_api_version == expected['version']
  assert llm.deployment_name == expected['deployment']
  assert llm.temperature == expected['temperature']
  assert llm.streaming == expected['stream']
  assert llm.max_retries == expected['max_retries']
  assert isinstance(llm.http_client, http_client)
  assert isinstance(llm.http_async_client, http_async_client)
  assert llm_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_azure_openai_embedding(get_azure_openai_context):
  kwargs, expected, _, embedding_checker, http_client, http_async_client = get_azure_openai_context
  is_embedded = True
  instance = llms.AzureOpenAILLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, AzureOpenAIEmbeddings)
  assert llm.model == expected['model']
  assert llm.openai_api_key.get_secret_value() == expected['api_key']
  assert llm.azure_endpoint == expected['endpoint']
  assert llm.openai_api_version == expected['version']
  assert llm.deployment == expected['deployment']
  assert llm.max_retries == expected['max_retries']
  assert isinstance(llm.http_client, http_client)
  assert isinstance(llm.http_async_client, http_async_client)
  assert embedding_checker(fields, expected)

# ================
# = AnthropicLLM =
# ================
@pytest.fixture
def get_anthropic_context(fileds_checker_wrapper, basellm_context):
  specific = {
    'api_key': 'anthropic-key',
    'endpoint': 'http://dummy-anthropic/endpoint',
  }
  def _check_specific(fields, expected):
    return all([
      fields['api_key'] == expected['api_key'],
      fields['endpoint'] == expected['endpoint'],
    ])
  # Setup
  wrapper = fileds_checker_wrapper
  kwargs, expected, _, _ = basellm_context
  llm_checker, embedding_checker = wrapper(_check_specific)
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, llm_checker, embedding_checker, anthropic.Client, anthropic.AsyncClient

@pytest.mark.chatbot
@pytest.mark.util
def test_check_anthropic_chatbot(get_anthropic_context):
  kwargs, expected, llm_checker, _, http_client, http_async_client = get_anthropic_context
  is_embedded = False
  instance = llms.AnthropicLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, CustomChatAnthropic)
  assert llm.model == expected['model']
  assert llm.anthropic_api_key.get_secret_value() == expected['api_key']
  assert llm.anthropic_api_url == expected['endpoint']
  assert llm.temperature == expected['temperature']
  assert llm.streaming == expected['stream']
  assert llm.max_retries == expected['max_retries']
  assert isinstance(llm._client, http_client)
  assert isinstance(llm._async_client, http_async_client)
  assert llm_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_anthropic_embedding(get_anthropic_context):
  kwargs, _, _, _, _, _ = get_anthropic_context
  is_embedded = True
  instance = llms.AnthropicLLM(**kwargs)

  with pytest.raises(ValueError) as ex_llm:
    _ = instance.get_llm(is_embedded=is_embedded)
  with pytest.raises(ValueError) as ex_fields:
    _ = instance.get_fields(is_embedded=is_embedded)

  assert 'Embedding model' in str(ex_llm.value)
  assert 'Embedding model' in str(ex_fields.value)

# ==============
# = BedrockLLM =
# ==============
@pytest.fixture
def get_bedrock_context(basellm_context, mocker):
  specific = {
    'service_name': 'bedrock-runtime',
    'region_name': 'dummy-bedrock-region',
    'version': 'dummy-bedrock-version',
    'endpoint': 'http://dummy-bedrock/endpoint',
    'access_key': 'dummy-access-key',
    'secret_key': 'dummy-secret-key',
  }
  def llm_checker(fields, expected):
    use_proxy = 'proxy' in expected.keys()

    return all([
      'max_retries' not in fields.keys(),
      fields['model'] == expected['model'],
      fields['temperature'] == expected['temperature'],
      fields['stream'] == expected['stream'],
      fields['proxy'] == expected['proxy'] if use_proxy else fields['proxy'] is None,
      fields['service_name'] == expected['service_name'],
      fields['region_name'] == expected['region_name'],
      fields['version'] == expected['version'],
      fields['endpoint'] == expected['endpoint'],
      fields['access_key'] == expected['access_key'],
      fields['secret_key'] == expected['secret_key'],
    ])
  def embedding_checker(fields, expected):
    use_proxy = 'proxy' in expected.keys()
    ignore_keys = ['temperature', 'stream', 'max_retries']

    return all([
      all([ignore_key not in fields.keys() for ignore_key in ignore_keys]),
      fields['model'] == expected['model'],
      fields['proxy'] == expected['proxy'] if use_proxy else fields['proxy'] is None,
      fields['service_name'] == expected['service_name'],
      fields['region_name'] == expected['region_name'],
      fields['version'] == expected['version'],
      fields['endpoint'] == expected['endpoint'],
      fields['access_key'] == expected['access_key'],
      fields['secret_key'] == expected['secret_key'],
    ])
  def _judge_proxy(config, expected):
    use_proxy = 'proxy' in expected.keys()

    if use_proxy:
      return all([
        config.proxies['http'] == expected['proxy'],
        config.proxies['https'] == expected['proxy'],
      ])
    else:
      return True
  # Setup
  kwargs, expected, _, _ = basellm_context
  kwargs.update(specific)
  expected.update(specific)
  # Define mock
  dummy_model = {
    'version': '0.0.1',
    'metadata': {
      'apiVersion': kwargs['version'],
      'endpointPrefix': 'bedrock-runtime',
      'jsonVersion': '1.1',
      'protocol': 'rest-json',
      'serviceFullName': 'Amazon Bedrock Runtime',
      'serviceId': 'Bedrock Runtime',
      'signatureVersion': 'v4',
      'signingName': 'bedrock',
      'uid': 'bedrock-runtime-uid',
    }
  }
  _const_param_getter = lambda: ({
    'type': 'String',
    'buildIn': 'dummy',
    'default': 'sample',
    'required': False,
    'deprecated': False,
  })
  dummy_rule = {
    'version': '0.0.1',
    'parameters': {
      'Region': _const_param_getter(),
      'UseDualStack': _const_param_getter(),
      'UseFIPS': _const_param_getter(),
      'Endpoint': _const_param_getter(),
    },
    'rules': []
  }
  mocker.patch('botocore.loaders.Loader.load_service_model', side_effect=[dummy_model, dummy_rule])

  return kwargs, expected, _judge_proxy, llm_checker, embedding_checker

@pytest.mark.chatbot
@pytest.mark.util
def test_check_bedrock_chatbot(get_bedrock_context):
  kwargs, expected, judge_proxy, llm_checker, _ = get_bedrock_context
  is_embedded = False
  instance = llms.BedrockLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, ChatBedrock)
  assert llm.model_id == expected['model']
  assert llm.model_kwargs == {'temperature': expected['temperature']}
  assert llm.streaming == expected['stream']
  assert llm_checker(fields, expected)
  assert judge_proxy(llm.client._client_config, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_bedrock_embedding(get_bedrock_context):
  kwargs, expected, judge_proxy, _, embedding_checker = get_bedrock_context
  is_embedded = True
  instance = llms.BedrockLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, BedrockEmbeddings)
  assert llm.model_id == expected['model']
  assert embedding_checker(fields, expected)
  assert judge_proxy(llm.client._client_config, expected)

# ================
# = FireworksLLM =
# ================
@pytest.fixture
def get_fireworks_context(basellm_context):
  specific = {
    'api_key': 'dummy-firework-api-key',
    'endpoint': 'http://dummy-fireworks/endpoint',
  }
  def llm_checker(fields, expected):
    use_proxy = 'proxy' in expected.keys()

    return all([
      fields['model'] == expected['model'],
      fields['temperature'] == expected['temperature'],
      fields['stream'] == expected['stream'],
      fields['max_retries'] == expected['max_retries'],
      fields['proxy'] == expected['proxy'] if use_proxy else fields['proxy'] is None,
      fields['api_key'] == expected['api_key'],
      fields['endpoint'] == expected['endpoint'],
    ])
  def embedding_checker(fields, expected):
    ignore_keys = ['temperature', 'stream', 'max_retries']
    use_proxy = 'proxy' in expected.keys()

    return all([
      all([ignore_key not in fields.keys() for ignore_key in ignore_keys]),
      fields['model'] == expected['model'],
      fields['proxy'] == expected['proxy'] if use_proxy else fields['proxy'] is None,
      fields['api_key'] == expected['api_key'],
      fields['endpoint'] == expected['endpoint'],
    ])
  # Setup
  kwargs, expected, _, _ = basellm_context
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, llm_checker, embedding_checker

@pytest.mark.chatbot
@pytest.mark.util
def test_check_fireworks_chatbot(get_fireworks_context):
  kwargs, expected, llm_checker, _ = get_fireworks_context
  is_embedded = False
  instance = llms.FireworksLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, CustomChatFireworks)
  assert llm.model_name == expected['model']
  assert llm.fireworks_api_key.get_secret_value() == expected['api_key']
  assert llm.fireworks_api_base == expected['endpoint']
  assert llm.temperature == expected['temperature']
  assert llm.streaming == expected['stream']
  assert llm.model_kwargs == {'max_retries': expected['max_retries']}
  assert isinstance(llm.client, ChatCompletionV2)
  assert isinstance(llm.async_client, ChatCompletionV2)
  assert llm_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_fireworks_embedding(get_fireworks_context):
  kwargs, expected, _, embedding_checker = get_fireworks_context
  is_embedded = True
  instance = llms.FireworksLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, CustomFireworksEmbeddings)
  assert llm.model == expected['model']
  assert llm.fireworks_api_key.get_secret_value() == expected['api_key']
  assert llm.base_url == expected['endpoint']
  assert isinstance(llm._client._client, httpx.Client)
  assert embedding_checker(fields, expected)

# =============
# = OllamaLLM =
# =============
@pytest.fixture
def get_ollama_context(basellm_context):
  specific = {
    'endpoint': 'http://dummy-ollama/endpoint',
  }
  def _check_specific(fields, expected):
    ignore_keys = ['stream', 'max_retries']

    return all([
      all([ignore_key not in fields.keys() for ignore_key in ignore_keys]),
      fields['model'] == expected['model'],
      fields['temperature'] == expected['temperature'],
      fields['endpoint'] == expected['endpoint'],
    ])
  # Setup
  kwargs, expected, _, _ = basellm_context
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, _check_specific

@pytest.mark.chatbot
@pytest.mark.util
def test_check_ollama_chatbot(get_ollama_context):
  kwargs, expected, checker = get_ollama_context
  is_embedded = False
  instance = llms.OllamaLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, ChatOllama)
  assert llm.model == expected['model']
  assert llm.base_url == expected['endpoint']
  assert llm.temperature == expected['temperature']
  assert checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_ollama_embedding(get_ollama_context):
  kwargs, expected, checker = get_ollama_context
  is_embedded = True
  instance = llms.OllamaLLM(**kwargs)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, OllamaEmbeddings)
  assert llm.model == expected['model']
  assert llm.base_url == expected['endpoint']
  assert llm.temperature == expected['temperature']
  assert checker(fields, expected)

# =============
# = GeminiLLM =
# =============
@pytest.fixture
def get_gemini_context(get_basellm_arg, mocker):
  specific = {
    'location': 'us-central1',
  }
  def llm_field_checker(fields, expected):
    return all([
      fields['model'] == expected['model'],
      fields['temperature'] == expected['temperature'],
      fields['max_retries'] == expected['max_retries'],
      fields['stream'] == expected['stream'],
      fields['location'] == expected['location'],
    ])

  def embedding_field_checker(fields, expected):
    ignore_keys = ['temperature', 'stream']

    return all([
      all([ignore_key not in fields.keys() for ignore_key in ignore_keys]),
      fields['model'] == expected['model'],
      fields['max_retries'] == expected['max_retries'],
      fields['location'] == expected['location'],
    ])
  # mock
  class DummyCredentials:
    def __init__(self, info):
      self.info = info
      self.project = 'dummy-project'
  class DummyPretrainedModel:
    def __init__(self, *args, **kwargs):
      self._endpoint_name = 'dummy-endpoint'
    def get_embeddings(self, *args, **kwargs):
      return None
  mocker.patch('google.oauth2.service_account.Credentials.from_service_account_info', lambda info, **kwargs: DummyCredentials(info))
  mocker.patch('vertexai.language_models.ChatModel.from_pretrained', new=DummyPretrainedModel)
  mocker.patch('vertexai.preview.language_models.ChatModel.from_pretrained', new=DummyPretrainedModel)
  mocker.patch('vertexai.language_models.TextEmbeddingModel.from_pretrained', new=DummyPretrainedModel)
  mocker.patch('vertexai.vision_models.MultiModalEmbeddingModel.from_pretrained', new=DummyPretrainedModel)
  # Setup
  kwargs, expected = get_basellm_arg
  kwargs.update(specific)
  expected.update(specific)

  return kwargs, expected, llm_field_checker, embedding_field_checker

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.parametrize('model_name', ['gemini', 'chat-bison'])
def test_check_gemini_chatbot(get_gemini_context, model_name):
  specific = {'model': model_name}
  service_account = {'username': 'hogehoge', 'email': 'hoge@example.com'}
  kwargs, expected, llm_checker, _ = get_gemini_context
  kwargs.update(specific)
  expected.update(specific)
  is_embedded = False
  instance = llms.GeminiLLM(**kwargs, service_account=service_account)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, ChatVertexAI)
  assert llm.model_name == expected['model']
  assert llm.temperature == expected['temperature']
  assert llm.max_retries == expected['max_retries']
  assert llm.streaming == expected['stream']
  assert llm_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.parametrize('model_name', ['textembedding-gecko@003', 'multimodalembedding'])
def test_check_gemini_embedding(get_gemini_context, model_name):
  specific = {'model': model_name}
  service_account = {'username': 'hogehoge', 'email': 'hoge@example.com'}
  kwargs, expected, _, embedding_checker = get_gemini_context
  kwargs.update(specific)
  expected.update(specific)
  is_embedded = True
  instance = llms.GeminiLLM(**kwargs, service_account=service_account)
  llm = instance.get_llm(is_embedded=is_embedded)
  fields = instance.get_fields(is_embedded=is_embedded)

  assert isinstance(llm, VertexAIEmbeddings)
  assert llm.model_name == expected['model']
  assert llm.max_retries == expected['max_retries']
  assert embedding_checker(fields, expected)

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.parametrize('is_embedded', [False, True])
def test_invalid_service_account_gemini(get_gemini_context, is_embedded):
  kwargs = get_gemini_context[0]
  instance = llms.GeminiLLM(**kwargs, service_account=None)

  with pytest.raises(ValueError) as ex:
    _ = instance.get_llm(is_embedded=is_embedded)

  assert 'service_account must be set' in str(ex.value)
