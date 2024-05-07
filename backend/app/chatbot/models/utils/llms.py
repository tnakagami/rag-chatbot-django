from typing import Dict, List, Union, Any, Mapping
from dataclasses import dataclass, asdict
from google.oauth2.service_account import Credentials
from ._client import get_client
# Import langchain libraries of Chatbot
import boto3
from botocore.config import Config
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from ._customLLMWrapper import CustomChatAnthropic, CustomChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models.ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
# Import langchain libraries of Embedding
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from ._customLLMWrapper import CustomFireworksEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

@dataclass
class _BaseLLM:
  model: str = ''
  temperature: int = 0
  stream: bool = False
  max_retries: int = 10
  proxy: Union[Any, None] = None

  def get_llm(self, is_embedded=False):
    raise NotImplementedError

  def delete_keys(self, target: Dict, ignore_keys: List[str]):
    for key in ignore_keys:
      del target[key]

    return target

  def get_fields(self, instance, is_embedded=False):
    ignore_fields_for_embedding = ['temperature', 'stream']
    target = asdict(instance)

    if is_embedded:
      target = self.delete_keys(target, ignore_fields_for_embedding)

    return target

@dataclass
class OpenAILLM(_BaseLLM):
  api_key: str = ''
  endpoint: str = ''

  def get_llm(self, is_embedded=False):
    http_client = get_client(proxy=self.proxy, is_async=False)
    http_async_client = get_client(proxy=self.proxy, is_async=True)

    if is_embedded:
      llm = OpenAIEmbeddings(
        http_client=http_client,
        http_async_client=http_async_client,
        model=self.model,
        api_key=self.api_key,
        base_url=self.endpoint,
        max_retries=self.max_retries,
      )
    else:
      llm = ChatOpenAI(
        http_client=http_client,
        http_async_client=http_async_client,
        model=self.model,
        api_key=self.api_key,
        base_url=self.endpoint,
        temperature=self.temperature,
        max_retries=self.max_retries,
        streaming=self.stream,
      )

    return llm

  def get_fields(self, is_embedded=False):
    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class AzureOpenAILLM(_BaseLLM):
  api_key: str = ''
  endpoint: str = ''
  version: str = ''
  deployment: str = ''

  def get_llm(self, is_embedded=False):
    http_client = get_client(proxy=self.proxy, is_async=False)
    http_async_client = get_client(proxy=self.proxy, is_async=True)

    if is_embedded:
      llm = AzureOpenAIEmbeddings(
        http_client=http_client,
        http_async_client=http_async_client,
        model=self.model,
        api_key=self.api_key,
        azure_endpoint=self.endpoint,
        api_version=self.version,
        azure_deployment=self.deployment,
        max_retries=self.max_retries,
      )
    else:
      llm = AzureChatOpenAI(
        http_client=http_client,
        http_async_client=http_async_client,
        model=self.model,
        api_key=self.api_key,
        azure_endpoint=self.endpoint,
        api_version=self.version,
        azure_deployment=self.deployment,
        temperature=self.temperature,
        max_retries=self.max_retries,
        streaming=self.stream,
      )

    return llm

  def get_fields(self, is_embedded=False):
    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class AnthropicLLM(_BaseLLM):
  api_key: str = ''
  endpoint: str = ''

  def get_llm(self, is_embedded=False):
    if is_embedded:
      raise ValueError(f'[{self.__class__.__name__}] Embedding model is not implemented')
    else:
      llm = CustomChatAnthropic(
        model=self.model,
        api_key=self.api_key,
        anthropic_api_url=self.endpoint,
        proxy_url=self.proxy,
        temperature=self.temperature,
        max_retries=self.max_retries,
        streaming=self.stream,
      )

    return llm

  def get_fields(self, is_embedded=False):
    if is_embedded:
      raise ValueError(f'[{self.__class__.__name__}] Embedding model is not implemented')

    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class BedrockLLM(_BaseLLM):
  service_name: str = 'bedrock-runtime'
  region_name: str = ''
  version: str = ''
  endpoint: str = ''
  access_key: str = ''
  secret_key: str = ''

  def get_llm(self, is_embedded=False):
    proxy_url = self.proxy

    if proxy_url:
      config = Config(
        region_name=self.region_name or None,
        proxies={'http': proxy_url, 'https': proxy_url}
      )
    else:
      config = None

    client = boto3.client(
      self.service_name,
      region_name=self.region_name or None,
      api_version=self.version or None,
      endpoint_url=self.endpoint or None,
      aws_access_key_id=self.access_key or None,
      aws_secret_access_key=self.secret_key or None,
      config=config,
    )

    if is_embedded:
      llm = BedrockEmbeddings(
        model_id=self.model,
        client=client,
      )
    else:
      model_kwargs = {
        'temperature': self.temperature,
      }
      llm = ChatBedrock(
        model_id=self.model,
        streaming=self.stream,
        client=client,
        model_kwargs=model_kwargs,
      )

    return llm

  def get_fields(self, is_embedded=False):
    target = super().get_fields(self, is_embedded=is_embedded)
    target = self.delete_keys(target, ['max_retries'])

    return target

@dataclass
class FireworksLLM(_BaseLLM):
  api_key: str = ''
  endpoint: str = ''

  def get_llm(self, is_embedded=False):
    if is_embedded:
      llm = CustomFireworksEmbeddings(
        http_client=get_client(proxy=self.proxy, is_async=False),
        model=self.model,
        fireworks_api_key=self.api_key,
        base_url=self.endpoint,
      )
    else:
      model_kwargs = {'max_retries': self.max_retries}
      llm = CustomChatFireworks(
        model=self.model,
        api_key=self.api_key,
        base_url=self.endpoint,
        temperature=self.temperature,
        streaming=self.stream,
        model_kwargs=model_kwargs,
        proxy_url=self.proxy,
      )

    return llm

  def get_fields(self, is_embedded=False):
    target = super().get_fields(self, is_embedded=is_embedded)

    if is_embedded:
      target = self.delete_keys(target, ['max_retries'])

    return target

@dataclass
class OllamaLLM(_BaseLLM):
  model: str = 'llama2'
  endpoint: str = ''

  def get_llm(self, is_embedded=False):
    if is_embedded:
      llm = OllamaEmbeddings(
        model=self.model,
        base_url=self.endpoint,
        temperature=self.temperature,
      )
    else:
      llm = ChatOllama(
        model=self.model,
        base_url=self.endpoint,
        temperature=self.temperature,
      )

    return llm

  def get_fields(self, is_embedded=False):
    target = super().get_fields(self, is_embedded=False)
    target = self.delete_keys(target, ['max_retries', 'stream'])

    return target

@dataclass
class GeminiLLM(_BaseLLM):
  model: str = 'gemini'
  service_account: Mapping[str, str] = None
  location: str = 'us-central1'

  def get_llm(self, is_embedded=False):
    if self.service_account is None:
      raise ValueError(f'[{self.__class__.__name__}] service_account must be set. Please check your configure.')
    # Collect credentials from service account information
    credentials = Credentials.from_service_account_info(self.service_account)
    project = credentials.project

    if is_embedded:
      llm = VertexAIEmbeddings(
        project=project,
        credentials=credentials,
        model_name=self.model,
        max_retries=self.max_retries,
        location=self.location,
      )
    else:
      llm = ChatVertexAI(
        project=project,
        credentials=credentials,
        model=self.model,
        temperature=self.temperature,
        max_retries=self.max_retries,
        streaming=self.stream,
        location=self.location,
      )

    return llm

  def get_fields(self, is_embedded=False):
    return super().get_fields(self, is_embedded=is_embedded)