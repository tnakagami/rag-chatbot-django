from typing import Optional, Dict, Union, Any
from dataclasses import dataclass, make_dataclass, field
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain.schema.embeddings import Embeddings
from langchain_robocorp import ActionServerToolkit
from langchain_community.tools.arxiv.tool import ArxivQueryRun, ArxivInput
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.connery import ConneryService
from langchain_community.agent_toolkits.connery import ConneryToolkit
from ._customRetriever import CustomDallEAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun, DDGInput
from ._customRetriever import CustomKayAiRetriever
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_community.tools.tavily_search import (
    TavilyAnswer as _TavilyAnswer,
    TavilySearchResults,
)
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.retrievers.you import YouRetriever
from langchain_community.retrievers.wikipedia import WikipediaRetriever
# For Django
from django.utils.translation import gettext_lazy
from django.db.models import Manager as DjangoManager
from ._local import get_client, LocalField
from .vectorstore import CustomVectorStore, DistanceStrategy

@dataclass
class _RetrievalConfig:
  assistant_id: int
  manager: DjangoManager
  strategy: DistanceStrategy
  embeddings: Embeddings
  search_kwargs: Union[Dict[str, Any], LocalField] = field(default_factory=dict)

  def __post_init__(self) -> None:
    target = LocalField(
      name='k',
      value=self.search_kwargs.get('k', 4),
      default=4,
      data_type=int,
      label=str(gettext_lazy('Number of output documents when searching documents by similarity')),
    )
    target.value = target.data
    self.search_kwargs = target

class _ToolConfig(BaseModel):
  Ellipsis

class _ApiKeyConfig(_ToolConfig):
  api_key: str = Field('', description=gettext_lazy('api key'))

class _ActionServerConfig(_ApiKeyConfig):
  url: str = Field('', description=gettext_lazy('url'))

class _ConneryConfig(_ApiKeyConfig):
  url: str = Field('', description=gettext_lazy('url'))

class _DallEConfig(_ApiKeyConfig):
  model: str = Field('', description=gettext_lazy('model name'))
  endpoint: str = Field('', description=gettext_lazy('endpoint'))
  proxy: Optional[Union[Any, None]] = Field(None, description=gettext_lazy('proxy url'))

class _BaseTool(BaseModel):
  name: str = ''
  description: str = ''
  multi_use: Optional[bool] = False
  config: Optional[Union[_ToolConfig, None]] = None

  def __init__(self, config: Union[Dict, None] = None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def get_config_fields(self) -> list[LocalField]:
    if self.config is not None:
      schema = self.config.schema()

      targets = [
        LocalField(
          name=name,
          value=getattr(self.config, name, info.get('default', None)),
          default=info.get('default', None),
          data_type=str,
          label=str(info['description']),
        )
        for name, info in schema['properties'].items()
      ]
    else:
      targets = []

    return targets

  def get_tools(self):
    raise NotImplementedError

class RetrievalTool(_BaseTool):
  name: str = Field(gettext_lazy('Retrieval'), const=True)
  description: str = Field(gettext_lazy('Look up information in uploaded files.'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _RetrievalConfig(**config)

  def get_config_fields(self) -> list[LocalField]:
    targets = [self.config.search_kwargs]

    return targets

  def get_tools(self) -> Tool:
    kwargs = self.config.search_kwargs.asdict()
    kwargs.update({'assistant_id': self.config.assistant_id})
    vectorstore = CustomVectorStore(
      manager=self.config.manager,
      strategy=self.config.strategy,
      embedding_function=self.config.embeddings,
    )
    tool = create_retriever_tool(
      retriever=vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs=kwargs,
      ),
      name=str(self.name),
      description=str(self.description)
    )

    return tool

class ActionServerTool(_BaseTool):
  name: str = Field(gettext_lazy('Action Server by Robocorp'), const=True)
  description: str = Field(gettext_lazy('Run AI actions with Roborop Action Server'), const=True)
  multi_use: bool = Field(True, const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ActionServerConfig(**config)

  def get_tools(self) -> list[Tool]:
    toolkit = ActionServerToolkit(
      api_key=str(self.config.api_key),
      url=str(self.config.url),
    )
    tools = toolkit.get_tools()

    return tools

class ArxivTool(_BaseTool):
  name: str = Field(gettext_lazy('Arxiv'), const=True)
  description: str = Field(gettext_lazy('Searches Arxiv'), const=True)

  def get_tools(self) -> Union[Tool, ArxivQueryRun]:
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(), args_schema=ArxivInput)

class ConneryTool(_BaseTool):
  name: str = Field(gettext_lazy('AI Action Runner by Connery'), const=True)
  description: str = Field(gettext_lazy('Connect OpenGPTs to the real world with Connery'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ConneryConfig(**config)

  def get_tools(self) -> list[Tool]:
    service = ConneryService(
      api_key=str(self.config.api_key),
      runner_url=str(self.config.url),
    )
    toolkit = ConneryToolkit.create_instance(service)
    tools = toolkit.get_tools()

    return tools

class DallETool(_BaseTool):
  name: str = Field(gettext_lazy('Image Generator (Dall-E)'), const=True)
  description: str = Field(gettext_lazy("Generates images from a text description using OpenAI's DALL-E model."), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _DallEConfig(**config)

  def get_tools(self) -> Tool:
    wrapper = CustomDallEAPIWrapper(
      http_client=get_client(self.config.proxy, is_async=False),
      http_async_client=get_client(self.config.proxy, is_async=True),
      model=str(self.config.model),
      api_key=str(self.config.api_key),
      base_url=str(self.config.endpoint),
      max_retries=10,
      size='1024x1024',
      quality='hd',
    )
    tool = Tool(
      name=str(self.name),
      func=wrapper.run,
      description=str(gettext_lazy(
        'Useful for when you need to generate images from a text description. Input should be an image description.',
      )),
    )

    return tool

class DDGSearchTool(_BaseTool):
  name: str = Field(gettext_lazy('DuckDuckGo Search'), const=True)
  description: str = Field(gettext_lazy('Searches the web with DuckDuckGo'), const=True)

  def get_tools(self) -> Union[Tool, DuckDuckGoSearchRun]:
    return DuckDuckGoSearchRun(args_schema=DDGInput)

class SecFilingsTool(_BaseTool):
  name: str = Field(gettext_lazy('SEC Filings (Kay.ai)'), const=True)
  description: str = Field(gettext_lazy('Searches through SEC filings using Kay.ai'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ApiKeyConfig(**config)

  def get_tools(self) -> Tool:
    retriever = CustomKayAiRetriever.create(
      api_key=str(self.config.api_key),
      dataset_id='company',
      data_types=['10-K', '10-Q'],
      num_contexts=3
    )
    tool = create_retriever_tool(
      retriever=retriever,
      name=str(self.name),
      description=str(self.description),
    )

    return tool

class PressReleasesTool(_BaseTool):
  name: str = Field(gettext_lazy('Press Releases (Kay.ai)'), const=True)
  description: str = Field(gettext_lazy('Searches through press releases using using Kay.ai'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ApiKeyConfig(**config)

  def get_tools(self) -> Tool:
    retriever = CustomKayAiRetriever.create(
      api_key=str(self.config.api_key),
      dataset_id='company',
      data_types=['PressRelease'],
      num_contexts=6
    )
    tool = create_retriever_tool(
      retriever=retriever,
      name=str(self.name),
      description=str(self.description),
    )

    return tool

class PubMedTool(_BaseTool):
  name: str = Field(gettext_lazy('PubMed'), const=True)
  description: str = Field(gettext_lazy('Searches PubMed'), const=True)

  def get_tools(self) -> Tool:
    return create_retriever_tool(
      retriever=PubMedRetriever(),
      name=str(self.name),
      description=str(self.description),
    )

class TavilySearchTool(_BaseTool):
  name: str = Field(gettext_lazy('Tavily Search (with evidence)'), const=True)
  description: str = Field(gettext_lazy('Uses the Tavilysearch engine. Includes sources in the response.'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ApiKeyConfig(**config)

  def get_tools(self) -> Union[Tool, TavilySearchResults]:
    return TavilySearchResults(
      api_wrapper=TavilySearchAPIWrapper(tavily_api_key=str(self.config.api_key)),
      name=str(self.name),
    )

class TavilyAnswerTool(_BaseTool):
  name: str = Field(gettext_lazy('Tavily Search (only answer)'), const=True)
  description: str = Field(gettext_lazy('Uses the Tavilysearch engine. This returns only the answer, no supporting evidence.'), const=True)

  def __init__(self, config: Dict, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ApiKeyConfig(**config)

  def get_tools(self) -> Union[Tool, _TavilyAnswer]:
    return _TavilyAnswer(
      api_wrapper=TavilySearchAPIWrapper(tavily_api_key=str(self.config.api_key)),
      name=str(self.name),
    )

class YouSearchTool(_BaseTool):
  name: str = Field(gettext_lazy('You.com Search'), const=True)
  description: str = Field(gettext_lazy('Uses You.com search, optimized responses for LLMs.'), const=True)

  def __init__(self, config: Union[Dict, None] = None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.config = _ApiKeyConfig(**config)

  def get_tools(self) -> Tool:
    retriever = YouRetriever(
      n_hits=3,
      n_snippets_per_hit=3,
      ydc_api_key=str(self.config.api_key),
    )
    tool = create_retriever_tool(
      retriever=retriever,
      name=str(self.name),
      description=str(self.description),
    )

    return tool

class WikipediaTool(_BaseTool):
  name: str = Field(gettext_lazy('Wikipedia Search'), const=True)
  description: str = Field(gettext_lazy('Searches Wikipedia'), const=True)

  def get_tools(self) -> Tool:
    return create_retriever_tool(
      retriever=WikipediaRetriever(),
      name=str(self.name),
      description=str(self.description),
    )
