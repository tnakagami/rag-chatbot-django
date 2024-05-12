from typing import Dict, List, Union, Any, Mapping
from dataclasses import dataclass, field, fields
from google.oauth2.service_account import Credentials
from ._local import get_client, LocalField
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
# For Django
from django.utils.translation import gettext_lazy

@dataclass
class _BaseLLM:
  model: str = field(default='', metadata={'type': str, 'label': gettext_lazy('model name')})
  temperature: int = field(default=0, metadata={'type': int, 'label': gettext_lazy('temperature')})
  stream: bool = field(default=False, metadata={'type': bool, 'label': gettext_lazy('enable streaming')})
  max_retries: int = field(default=10, metadata={'type': int, 'label': gettext_lazy('max retries')})
  proxy: Union[str, None] = field(default=None, metadata={'type': str, 'label': gettext_lazy('proxy url')})

  def get_llm(self, is_embedded: bool = False):
    raise NotImplementedError

  def delete_fields(self, targets: List[LocalField], ignores: List[str]) -> List[LocalField]:
    return list(filter(lambda item: item.name not in ignores, targets))

  def get_fields(self, instance: object, is_embedded: bool = False) -> List[LocalField]:
    ignore_fields_for_embedding = ['temperature', 'stream']
    targets = [
      LocalField(
        name=element.name,
        value=getattr(instance, element.name, element.default),
        default=element.default,
        data_type=element.metadata['type'],
        label=str(element.metadata['label']),
      )
      for element in fields(instance)
    ]

    if is_embedded:
      targets = self.delete_fields(targets, ignore_fields_for_embedding)

    return targets

@dataclass
class OpenAILLM(_BaseLLM):
  api_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('api key')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})

  def get_llm(self, is_embedded: bool = False) -> Union[OpenAIEmbeddings, ChatOpenAI]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class AzureOpenAILLM(_BaseLLM):
  api_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('api key')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})
  version: str = field(default='', metadata={'type': str, 'label': gettext_lazy('version')})
  deployment: str = field(default='', metadata={'type': str, 'label': gettext_lazy('deployment')})

  def get_llm(self, is_embedded: bool = False) -> Union[AzureOpenAIEmbeddings, AzureChatOpenAI]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class AnthropicLLM(_BaseLLM):
  api_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('api key')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})

  def get_llm(self, is_embedded: bool = False) -> CustomChatAnthropic:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    if is_embedded:
      raise ValueError(f'[{self.__class__.__name__}] Embedding model is not implemented')

    return super().get_fields(self, is_embedded=is_embedded)

@dataclass
class BedrockLLM(_BaseLLM):
  service_name: str = field(default='bedrock-runtime', metadata={'type': str, 'label': gettext_lazy('service name')})
  region_name: str = field(default='', metadata={'type': str, 'label': gettext_lazy('region name')})
  version: str = field(default='', metadata={'type': str, 'label': gettext_lazy('version')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})
  access_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('access key')})
  secret_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('secret key')})

  def get_llm(self, is_embedded: bool = False) -> Union[BedrockEmbeddings, ChatBedrock]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    targets = super().get_fields(self, is_embedded=is_embedded)
    targets = self.delete_fields(targets, ['max_retries'])

    return targets

@dataclass
class FireworksLLM(_BaseLLM):
  api_key: str = field(default='', metadata={'type': str, 'label': gettext_lazy('api key')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})

  def get_llm(self, is_embedded: bool = False) -> Union[CustomFireworksEmbeddings, CustomChatFireworks]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    targets = super().get_fields(self, is_embedded=is_embedded)

    if is_embedded:
      targets = self.delete_fields(targets, ['max_retries'])

    return targets

@dataclass
class OllamaLLM(_BaseLLM):
  model: str = field(default='llama2', metadata={'type': str, 'label': gettext_lazy('model name')})
  endpoint: str = field(default='', metadata={'type': str, 'label': gettext_lazy('endpoint')})

  def get_llm(self, is_embedded=False) -> Union[OllamaEmbeddings, ChatOllama]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    targets = super().get_fields(self, is_embedded=False)
    targets = self.delete_fields(targets, ['max_retries', 'stream'])

    return targets

@dataclass
class GeminiLLM(_BaseLLM):
  model: str = field(default='gemini', metadata={'type': str, 'label': gettext_lazy('model name')})
  service_account: Mapping[str, str] = field(default=None, metadata={'type': dict, 'label': gettext_lazy('service account')})
  location: str = field(default='us-central1', metadata={'type': str, 'label': gettext_lazy('location')})

  def get_llm(self, is_embedded=False) -> Union[VertexAIEmbeddings, ChatVertexAI]:
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

  def get_fields(self, is_embedded: bool = False) -> List[LocalField]:
    return super().get_fields(self, is_embedded=is_embedded)