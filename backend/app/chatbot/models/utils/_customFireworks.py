from typing import Union
import httpx
import fireworks
from fireworks.client import Chat
from fireworks.client.api_client import FireworksClient as FireworksClientV1
from fireworks.client.completion import CompletionV2
from fireworks.client.embedding import EmbeddingV1
from fireworks.client.rerank import RerankV1
from fireworks.client.image import ImageInference

class _BaseFireworks:
  _client_v1: FireworksClientV1
  _image_client_v1: ImageInference
  completion: CompletionV2
  chat: Chat
  embeddings: EmbeddingV1
  rerank: RerankV1

  def __init__(
    self,
    *,
    api_key: Union[str, None] = None,
    base_url: Union[str, httpx.URL, None] = None,
    timeout: int = 600,
    account: str = 'fireworks',
    **kwargs,
  ) -> None:
    self._client_v1 = FireworksClientV1(
      api_key=api_key, 
      base_url=base_url, 
      request_timeout=timeout, 
      **kwargs,
    )
    self._image_client_v1 = ImageInference(
      model=None,
      api_key=api_key,
      base_url=base_url,
      request_timeout=timeout,
      account=account,
    )
    self.completion = CompletionV2(self._client_v1)
    self.chat = Chat(self._client_v1)
    self.embeddings = EmbeddingV1(self._client_v1)
    self.rerank = RerankV1(self._client_v1)

class CustomFireworks(_BaseFireworks):
  def __enter__(self):
    return self

  def __exit__(self, *exc) -> None:
    self.close()

  def close(self) -> None:
    self._client_v1.close()
    self._image_client_v1.close()

class CustomAsyncFireworks(_BaseFireworks):
  async def __aenter__(self):
    return self

  async def __aexit__(self, *exc) -> None:
    await self.aclose()

  async def aclose(self) -> None:
    await self._client_v1.aclose()
    await self._image_client_v1.aclose()