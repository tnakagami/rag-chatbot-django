import pytest
from chatbot.models.utils import (
  _client,
  _customFireworks,
  _customLLMWrapper,
  _customRetriever,
)
# Classes for comparing instance
from pydantic.v1.error_wrappers import ValidationError
from openai import OpenAIError
import httpx
import fireworks
from fireworks.client import Chat
from fireworks.client.api_client import FireworksClient as FireworksClientV1
from fireworks.client.chat_completion import ChatCompletionV2
from fireworks.client.completion import CompletionV2
from fireworks.client.embedding import EmbeddingV1
from fireworks.client.rerank import RerankV1
from fireworks.client.image import ImageInference

def _client_non_proxy_checker(client_without_proxy):
  transport = list(client_without_proxy._mounts.values())

  return len(transport) == 0

def _client_proxy_checker(client_with_proxy, expected):
  transport = list(client_with_proxy._mounts.values())[0]
  _url = transport._pool._proxy_url

  return all([
    _url.scheme == expected['scheme'],
    _url.host == expected['host'],
    _url.port == expected['port'],
    _url.target == expected['target'],
  ])

@pytest.fixture
def get_firework_args():
  kwargs = {
    'api_key': 'api_key',
    'base_url': 'http://example.com/base',
    'timeout': 543,
  }

  return kwargs

@pytest.fixture
def get_chat_anthropic_args():
  kwargs = {
    'model': 'anthropic-model',
    'api_key': 'api-key',
    'anthropic_api_url': 'http://example.com/base',
    'max_retries': 10,
  }

  return kwargs

@pytest.fixture
def get_chat_fireworks_args():
  kwargs = {
    'model': 'fireworks-model',
    'api_key': 'api-key',
    'base_url': 'http://example.com/base',
  }
  model_kwargs = {
    'max_retries': 10,
  }

  return kwargs, model_kwargs

@pytest.fixture
def mock_post_request_of_kay_retriever(mocker):
  class DummyResponse:
    def __init__(self, status_code, success=None, error=None):
      self.status_code = status_code
      self.success = success
      self.error = error

    def json(self):
      return {
        'success': self.success,
        'error': self.error,
      }

  def get_response(url, json, headers=None):
    query = json['query']

    if 'valid' in query:
      response = DummyResponse(200, success=True, error=None)
    elif 'is-success' in query:
      response = DummyResponse(200, success='true', error=None)
    elif 'is-not-success' in query:
      response = DummyResponse(200, success=False, error='Not success')
    elif 'bad-request' in query:
      response = DummyResponse(400, success=False, error='Bad Request')
    elif 'not-auth' in query:
      response = DummyResponse(401, success=False, error='Not auth')
    else:
      response = DummyResponse(500, success=False, error='Internal Server Error')

    return response

  mocker.patch('chatbot.models.utils._customRetriever.requests.post', get_response)

  return mocker

@pytest.fixture
def get_kay_retriever_args():
  kwargs = {
    'api_key': 'api-key',
    'dataset_id': 'company',
    'data_types': ['type1', 'type2'],
  }

  return kwargs

@pytest.fixture
def mock_call_kay_is_success(get_kay_retriever_args, mocker):
  kwargs = get_kay_retriever_args
  mocker.patch('chatbot.models.utils._customRetriever._KayRetriever._call_kay', return_value={'success': True, 'contexts': 'valid-answer'})

  return mocker, kwargs

@pytest.fixture
def mock_call_kay_is_failed(get_kay_retriever_args, mocker):
  kwargs = get_kay_retriever_args
  mocker.patch('chatbot.models.utils._customRetriever._KayRetriever._call_kay', return_value={'success': False})

  return mocker, kwargs




@pytest.mark.private
@pytest.mark.parametrize([
  'proxy',
  'is_async',
  'expected_class',
  'expected_scheme',
  'expected_host',
  'expected_port',
  'expected_target',
], [
  ('http://example.com:12345', False, httpx.Client, b'http', b'example.com', 12345, b'/'),
  ('https://example.com/valid', False, httpx.Client, b'https', b'example.com', None, b'/valid'),
  ('http://example.co.jp:23456/ok', True, httpx.AsyncClient, b'http', b'example.co.jp', 23456, b'/ok'),
  ('https://example.co.jp', True, httpx.AsyncClient, b'https', b'example.co.jp', None, b'/'),
])
def test_check_valid_client(
  proxy,
  is_async,
  expected_class,
  expected_scheme,
  expected_host,
  expected_port,
  expected_target,
):
  instance = _client.get_client(proxy=proxy, is_async=is_async)
  expected_proxy = {
    'scheme': expected_scheme,
    'host': expected_host,
    'port': expected_port,
    'target': expected_target,
  }

  assert isinstance(instance, expected_class)
  assert _client_proxy_checker(instance, expected_proxy)

@pytest.mark.private
@pytest.mark.parametrize([
  'proxy',
  'is_async'
], [
  ('example.com:8000', False),
  ('example.com:8000', True),
  ('http://:8000', False),
  ('http://:8000', True),
  ('http://', False),
  ('http://', True),
  ('', False),
  ('', True),
  (None, False),
  (None, True),
])
def test_check_invalid_client(proxy, is_async):
  instance = _client.get_client(proxy=proxy, is_async=is_async)

  assert instance is None

@pytest.mark.private
@pytest.mark.parametrize('cls', [_customFireworks.CustomFireworks, _customFireworks.CustomAsyncFireworks])
def test_check_class_members_of_custom_fireworks(get_firework_args, cls):
  kwargs = get_firework_args
  instance = cls(**kwargs)

  assert isinstance(instance, cls)
  assert isinstance(instance._client_v1, FireworksClientV1)
  assert isinstance(instance._image_client_v1, ImageInference)
  assert isinstance(instance.completion, CompletionV2)
  assert isinstance(instance.chat, Chat)
  assert isinstance(instance.embeddings, EmbeddingV1)
  assert isinstance(instance.rerank, RerankV1)

@pytest.mark.private
def test_enter_and_exit_for_custom_fireworks_sync_client(get_firework_args):
  kwargs = get_firework_args

  try:
    with _customFireworks.CustomFireworks(**kwargs) as instance:
      pass
  except Exception:
    pytest.fail('Raise Exception when the instance of CustomFireworks is executed by `with` statement.')

@pytest.mark.private
@pytest.mark.asyncio
async def test_enter_and_exit_for_custom_fireworks_async_client(get_firework_args):
  kwargs = get_firework_args

  try:
    async with _customFireworks.CustomAsyncFireworks(**kwargs) as instance:
      pass
  except Exception:
    pytest.fail('Raise Exception when the instance of CustomAsyncFireworks is executed by `with` statement.')

@pytest.mark.private
@pytest.mark.parametrize('cls', [
  _customFireworks.CustomFireworks,
  _customFireworks.CustomAsyncFireworks,
])
def test_check_firewarks_input_args_without_proxy(get_firework_args, cls):
  kwargs = get_firework_args
  instance = cls(**kwargs)
  client_v1 = instance._client_v1

  assert client_v1.api_key == kwargs['api_key']
  assert client_v1.base_url == kwargs['base_url']
  assert client_v1.request_timeout == kwargs['timeout']
  assert _client_non_proxy_checker(client_v1._client)
  assert _client_non_proxy_checker(client_v1._async_client)

@pytest.mark.private
@pytest.mark.parametrize([
  'cls',
  'proxy_url',
  'expected_scheme',
  'expected_host',
  'expected_port',
  'expected_target',
], [
  (_customFireworks.CustomFireworks, 'http://proxy.com:12345/bar', b'http', b'proxy.com', 12345, b'/bar'),
  (_customFireworks.CustomAsyncFireworks, 'https://proxy.co.jp:23456/foo', b'https', b'proxy.co.jp', 23456, b'/foo'),
])
def test_check_firewarks_input_args_with_proxy(
  get_firework_args,
  cls,
  proxy_url,
  expected_scheme,
  expected_host,
  expected_port,
  expected_target,
):
  kwargs = get_firework_args
  instance = _customFireworks.CustomFireworks(**kwargs, proxies=proxy_url)
  client_v1 = instance._client_v1
  expected_proxy = {
    'scheme': expected_scheme,
    'host': expected_host,
    'port': expected_port,
    'target': expected_target,
  }

  assert client_v1.api_key == kwargs['api_key']
  assert client_v1.base_url == kwargs['base_url']
  assert client_v1.request_timeout == kwargs['timeout']
  assert _client_proxy_checker(client_v1._client, expected_proxy)
  assert _client_proxy_checker(client_v1._async_client, expected_proxy)

@pytest.mark.private
@pytest.mark.parametrize([
  'proxy_url',
  'expected_scheme',
  'expected_host',
  'expected_port',
  'expected_target',
], [
  (None, b'', b'', None, b'/'),
  ('http://proxy.com:8000/valid', b'http', b'proxy.com', 8000, b'/valid'),
])
def test_check_custom_chat_anthropic(
  get_chat_anthropic_args,
  proxy_url,
  expected_scheme,
  expected_host,
  expected_port,
  expected_target,
):
  kwargs = get_chat_anthropic_args
  llm = _customLLMWrapper.CustomChatAnthropic(**kwargs, proxy_url=proxy_url)
  expected_proxy = {
    'scheme': expected_scheme,
    'host': expected_host,
    'port': expected_port,
    'target': expected_target,
  }
  sync_client = llm._client._client
  async_client = llm._async_client._client

  assert isinstance(llm, _customLLMWrapper.CustomChatAnthropic)
  assert llm.model == kwargs['model']
  assert llm.anthropic_api_key.get_secret_value() == kwargs['api_key']
  assert llm.anthropic_api_url == kwargs['anthropic_api_url']
  assert llm.max_retries == kwargs['max_retries']
  assert llm.proxy_url is None if proxy_url is None else llm.proxy_url == proxy_url
  assert _client_non_proxy_checker(sync_client) if proxy_url is None else _client_proxy_checker(sync_client, expected_proxy)
  assert _client_non_proxy_checker(async_client) if proxy_url is None else _client_proxy_checker(async_client, expected_proxy)

@pytest.mark.private
@pytest.mark.parametrize([
  'model_name',
  'api_key',
  'base_url',
  'raise_class',
  'error_target'
], [
  (None, 'api-key', 'http://example.com/base', ValidationError, 'model_name'),
  ('anthropic-model', None, 'http://example.com/base', None, 'api_key'),
  ('anthropic-model', 'api-key', None, None, 'anthropic_api_url'),
])
def test_invalid_args_of_custom_chat_anthropic(
  model_name,
  api_key,
  base_url,
  raise_class,
  error_target,
):
  params = {
    'model': model_name,
    'api_key': api_key,
    'anthropic_api_url': base_url,
  }
  kwargs = {key: val for key, val in params.items() if val is not None}

  if raise_class is None:
    try:
      _ = _customLLMWrapper.CustomChatAnthropic(**kwargs)
    except Exception:
      pytest.fail(f'Raise Exception when the argument `{error_target}` is not set')
  else:
    with pytest.raises(raise_class) as ex:
      _ = _customLLMWrapper.CustomChatAnthropic(**kwargs)

    assert error_target in str(ex.value)

@pytest.mark.private
@pytest.mark.parametrize('proxy_url', [None, 'http://proxy.com:8000/valid'])
def test_check_custom_chat_fireworks(get_chat_fireworks_args, proxy_url):
  kwargs, model_kwargs = get_chat_fireworks_args
  llm = _customLLMWrapper.CustomChatFireworks(**kwargs, model_kwargs=model_kwargs, proxy_url=proxy_url)
  sync_client = llm.client
  async_client = llm.async_client

  assert llm.model_name == kwargs['model']
  assert llm.fireworks_api_key.get_secret_value() == kwargs['api_key']
  assert llm.fireworks_api_base == kwargs['base_url']
  assert llm.model_kwargs == model_kwargs
  assert llm.proxy_url is None if proxy_url is None else llm.proxy_url == proxy_url
  assert isinstance(sync_client, ChatCompletionV2)
  assert isinstance(async_client, ChatCompletionV2)

@pytest.mark.private
def test_invalid_api_key_for_custom_chat_fireworks(get_chat_fireworks_args):
  kwargs, _ = get_chat_fireworks_args
  del kwargs['api_key']

  with pytest.raises(ValidationError) as ex:
    _ = _customLLMWrapper.CustomChatFireworks(**kwargs)

  assert 'fireworks_api_key' in str(ex.value)

@pytest.mark.private
@pytest.mark.parametrize([
  'model_name',
  'api_key',
  'endpoint',
  'proxy_url',
  'expected_model',
  'expected_key',
  'expected_url',
  'expected_scheme',
  'expected_host',
  'expected_port',
  'expected_target',
], [
  (None, 'api-key', None, None,
   'nomic-ai/nomic-embed-text-v1.5', 'api-key', 'https://api.fireworks.ai/inference/v1', b'', b'', None, b'/'),
  ('fireworks-embeddings-model', 'api-key', 'http://example.com/foo', 'http://proxy.com:8000/valid', 
   'fireworks-embeddings-model', 'api-key', 'http://example.com/foo', b'http', b'proxy.com', 8000, b'/valid'),
])
def test_check_custom_fireworks_embeddings(
  model_name,
  api_key,
  endpoint,
  proxy_url,
  expected_model,
  expected_key,
  expected_url,
  expected_scheme,
  expected_host,
  expected_port,
  expected_target,
):
  params = {
    'model': model_name,
    'fireworks_api_key': api_key,
    'base_url': endpoint,
    'http_client': _client.get_client(proxy_url, is_async=False),
  }
  kwargs = {key: val for key, val in params.items() if val is not None}
  llm = _customLLMWrapper.CustomFireworksEmbeddings(**kwargs)
  sync_client = llm._client._client
  expected_proxy = {
    'scheme': expected_scheme,
    'host': expected_host,
    'port': expected_port,
    'target': expected_target,
  }

  assert llm.model == expected_model
  assert llm.fireworks_api_key.get_secret_value() == expected_key
  assert llm.base_url == expected_url
  assert _client_non_proxy_checker(sync_client) if proxy_url is None else _client_proxy_checker(sync_client, expected_proxy)

@pytest.mark.private
def test_invalid_api_key_for_custom_fireworks_embeddings():
  with pytest.raises(OpenAIError) as ex:
    _ = _customLLMWrapper.CustomFireworksEmbeddings()
  err_msg = str(ex.value)

  assert 'api_key' in err_msg and 'OPENAI_API_KEY' in err_msg

@pytest.mark.private
def test_check_custom_kay_retriever(get_kay_retriever_args):
  kwargs = get_kay_retriever_args
  instance = _customRetriever._KayRetriever(**kwargs)

  assert instance.api_key == kwargs['api_key']
  assert instance.dataset_id == kwargs['dataset_id']
  assert instance.data_types == kwargs['data_types']

@pytest.mark.private
def test_invalid_dataset_id_for_custom_kay_retriever(get_kay_retriever_args):
  kwargs = get_kay_retriever_args
  kwargs['dataset_id'] = 'invalid-dataset'

  with pytest.raises(_customRetriever.InvalidInput) as ex:
    _ = _customRetriever._KayRetriever(**kwargs)

  assert 'Invalid dataset_id' in str(ex.value)

@pytest.mark.private
@pytest.mark.parametrize('query', ['valid', 'is-success'])
def test_check_call_kay_of_custom_kay_retriever(mock_post_request_of_kay_retriever, get_kay_retriever_args, query):
  _ = mock_post_request_of_kay_retriever
  kwargs = get_kay_retriever_args
  instance = _customRetriever._KayRetriever(**kwargs)
  response = instance._call_kay(query)

  assert response['success'] == True

@pytest.mark.private
@pytest.mark.parametrize('query,raise_class,err', [
  ('is-not-success', _customRetriever.ServerError, 'Server error: Not success'),
  ('bad-request', _customRetriever.ServerError, 'Bad Request for'),
  ('not-auth', _customRetriever.APIKeyError, 'Invalid API Key'),
  ('other-error', _customRetriever.ServerError, 'Server error: 500'),
])
def test_invalid_call_kay_response_of_custom_kay_retriever(mock_post_request_of_kay_retriever, get_kay_retriever_args, query, raise_class, err):
  _ = mock_post_request_of_kay_retriever
  kwargs = get_kay_retriever_args
  instance = _customRetriever._KayRetriever(**kwargs)

  with pytest.raises(raise_class) as ex:
    _ = instance._call_kay(query)

  assert err in str(ex.value)

@pytest.mark.private
@pytest.mark.parametrize([
  'prompt',
  'num_context',
  'instruction',
  'expected_retrieval_config'
], [
  ('sample', 5, 'ok', {'num_context': 5, 'instruction': 'ok'}),
  ('sample', 5, None, {'num_context': 5}),
])
def test_check_call_kay_args_from_query_method_of_custom_kay_retriever(
  get_kay_retriever_args,
  mocker,
  prompt,
  num_context,
  instruction,
  expected_retrieval_config,
):
  def dummy_response_of_call_kay(_self, query, dataset_config, retrieval_config):
    response = {
      'success': True,
      'contexts': {
        'query': query,
        'dataset_config': dataset_config,
        'retrieval_config': retrieval_config,
      }
    }

    return response
  kwargs = get_kay_retriever_args
  mocker.patch('chatbot.models.utils._customRetriever._KayRetriever._call_kay', dummy_response_of_call_kay)
  instance = _customRetriever._KayRetriever(**kwargs)
  contexts = instance.query(prompt, num_context=num_context, instruction=instruction)
  expected_dataset_config = {
    'dataset_id': kwargs['dataset_id'],
    'data_types': kwargs['data_types'],
  }

  assert contexts['query'] == prompt
  assert contexts['dataset_config'] == expected_dataset_config
  assert contexts['retrieval_config'] == expected_retrieval_config

@pytest.mark.private
def test_check_query_is_success_of_custom_kay_retriever(mock_call_kay_is_success):
  _, kwargs = mock_call_kay_is_success
  instance = _customRetriever._KayRetriever(**kwargs)

  try:
    _ = instance.query('sample', instruction='ok')
  except Exception:
    pytest.fail(f'Raise Exception when the function query is called')

@pytest.mark.private
def test_check_query_is_not_success_of_custom_kay_retriever(mock_call_kay_is_failed):
  _, kwargs = mock_call_kay_is_failed
  instance = _customRetriever._KayRetriever(**kwargs)

  with pytest.raises(_customRetriever.ServerError) as ex:
    _ = instance.query('sample')

  assert 'Unknown Error' in str(ex.value)

@pytest.mark.private
def test_check_custom_kayai_retriever(get_kay_retriever_args):
  kwargs = get_kay_retriever_args
  num_contexts = 10
  instance = _customRetriever.CustomKayAiRetriever.create(**kwargs, num_contexts=num_contexts)

  assert isinstance(instance.client, _customRetriever._KayRetriever)
  assert instance.num_contexts == num_contexts