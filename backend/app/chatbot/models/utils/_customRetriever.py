import os
import requests
import openai
from typing import List, Dict, Union, Any
from langchain.pydantic_v1 import root_validator, Field
from langchain_community.utils.openai import is_openai_v1
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.retrievers.kay import KayAiRetriever
from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun

class CustomDallEAPIWrapper(DallEAPIWrapper):
  http_async_client: Union[Any, None] = None

  @root_validator()
  def validate_environment(cls, values: Dict) -> Dict:
    _sync_client = values.pop('http_client', None)
    async_client = values.pop('http_async_client', None)
    # Set None to `http_client` variable to create default client and async_client
    values['http_client'] = None
    _values = super().validate_environment(values)

    if is_openai_v1():
      client_params = {
        'api_key': _values['openai_api_key'],
        'organization': _values['openai_organization'],
        'base_url': _values['openai_api_base'],
        'timeout': _values['request_timeout'],
        'max_retries': _values['max_retries'],
        'default_headers': _values['default_headers'],
        'default_query': _values['default_query'],
      }
      _values['client'] = openai.OpenAI(**client_params, http_client=_sync_client).images
      _values['async_client'] = openai.AsyncOpenAI(**client_params, http_client=async_client).images

    return _values

class InvalidInput(Exception):
    pass

class APIKeyError(Exception):
    pass

class ServerError(Exception):
    pass

class _KayRetriever:
  def __init__(self, api_key=None, dataset_id='company', data_types=None):
    self._validate_dataset_id(dataset_id)
    self.api_key = api_key or os.getenv('KAY_API_KEY')
    self.dataset_id = dataset_id
    self.data_types = data_types

  def _validate_dataset_id(self, dataset_id):
    available_dataset_id = ['company']

    if dataset_id not in available_dataset_id:
      raise InvalidInput(
        f'Invalid dataset_id: Has to be one of the following {available_dataset_id}'
      )

  def query(self, query, num_context=6, instruction=None) -> list:
    dataset_config = {
      'dataset_id': self.dataset_id, 
      'data_types': self.data_types,
    }
    retrieval_config = {
      'num_context': num_context,
    }

    if instruction:
      retrieval_config['instruction'] = instruction
    # Request-Response
    embed_store_response = self._call_kay(query, dataset_config, retrieval_config)

    if embed_store_response.get('success') == True:
      contexts = embed_store_response.get('contexts')

      return contexts
    else:
      raise ServerError(embed_store_response.get('error', 'Unknown Error'))

  def _call_kay(self, prompt, dataset_config=None, retrieval_config=None):
    url = 'https://api.kay.ai/retrieve'
    headers = {'API-Key': self.api_key}
    payload = {
      'query': prompt,
      'dataset_config': dataset_config,
      'retrieval_config': retrieval_config,
    }
    response = requests.post(url, headers=headers, json=payload)
    status_code = response.status_code
    # Parsing the response and handling errors
    if status_code == 200:
      _json = response.json()

      if _json['success'] in {True, 'true'}:
        _json['success'] = True

        return _json
      else:
        err = _json['error']

        raise ServerError(f'Server error: {err}')
    elif status_code == 400:
      raise ServerError(f'Bad Request for {url}')
    elif status_code == 401:
      raise APIKeyError( 'Invalid API Key')
    else:
      raise ServerError(f'Server error: {status_code}')

class CustomKayAiRetriever(KayAiRetriever):
  @classmethod
  def create(
      cls,
      dataset_id: str,
      data_types: List[str],
      num_contexts: int = 6,
      api_key: str = '',
  ):
    client = _KayRetriever(
      api_key=api_key,
      dataset_id=dataset_id,
      data_types=data_types,
    )

    return cls(client=client, num_contexts=num_contexts)