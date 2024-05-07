from typing import Dict, Optional, Union, Any
from langchain.pydantic_v1 import root_validator
# For anthropic
import anthropic
from langchain_anthropic import ChatAnthropic
# For fireworks
from ._customFireworks import CustomFireworks, CustomAsyncFireworks
from langchain_fireworks import ChatFireworks
from langchain_fireworks.embeddings import FireworksEmbeddings
import openai

def _get_wrapper(values: Dict, key: str, default=None):
  out = values.get(key, None) or default

  return out

class CustomChatAnthropic(ChatAnthropic):
  proxy_url: Optional[str] = None

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    _values = super().validate_environment(values)
    client_params = {
      'api_key':         _values['anthropic_api_key'].get_secret_value(),
      'base_url':        _values['anthropic_api_url'],
      'max_retries':     _values['max_retries'],
      'default_headers': _values.get('default_headers'),
    }
    proxy_url = _get_wrapper(values, 'proxy_url')
    _values['_client'] = anthropic.Client(**client_params, proxies=proxy_url)
    _values['_async_client'] = anthropic.AsyncClient(**client_params, proxies=proxy_url)

    return _values

class CustomChatFireworks(ChatFireworks):
  proxy_url: Optional[str] = None

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    _values = super().validate_environment(values)
    api_key = _get_wrapper(_values, 'fireworks_api_key')

    if api_key:
      api_key = api_key.get_secret_value()

    client_params = {
      'api_key':  api_key,
      'base_url': _values['fireworks_api_base'],
      'timeout':  _values['request_timeout'],
    }
    proxy_url = _get_wrapper(values, 'proxy_url')
    _values['client'] = CustomFireworks(**client_params, proxies=proxy_url).chat.completions
    _values['async_client'] = CustomAsyncFireworks(**client_params, proxies=proxy_url).chat.completions

    return _values

class CustomFireworksEmbeddings(FireworksEmbeddings):
  http_client: Union[Any, None] = None
  base_url: Union[str, None] = None

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    _values = super().validate_environment(values)
    api_key = _get_wrapper(_values, 'fireworks_api_key')
    _values['base_url'] = _get_wrapper(values, 'base_url', default='https://api.fireworks.ai/inference/v1')
    _values['model'] = _get_wrapper(values, 'model', default='nomic-ai/nomic-embed-text-v1.5')

    if api_key:
      api_key = api_key.get_secret_value()

    client_params = {
      'api_key':  api_key,
      'base_url': _values['base_url'],
      'http_client': _get_wrapper(values, 'http_client')
    }
    _values['_client'] = openai.OpenAI(**client_params)

    return _values