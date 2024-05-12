from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Any
from collections.abc import Callable
from functools import wraps
from django.db import models
from django.utils.translation import gettext_lazy
from django.core.exceptions import ValidationError
from django.db.models import Manager as DjangoManager
from langchain.schema.embeddings import Embeddings
from langgraph.checkpoint import BaseCheckpointSaver
from .utils._local import LocalField
from .utils import (
  OpenAILLM,
  AzureOpenAILLM,
  AnthropicLLM,
  BedrockLLM,
  FireworksLLM,
  OllamaLLM,
  GeminiLLM,
)
from .utils import ToolExecutor, XmlExecutor, BaseTool
from .utils import (
  RetrievalTool,
  ActionServerTool,
  ArxivTool,
  ConneryTool,
  DallETool,
  DDGSearchTool,
  SecFilingsTool,
  PressReleasesTool,
  PubMedTool,
  TavilySearchTool,
  TavilyAnswerTool,
  YouSearchTool,
  WikipediaTool,
)

VALID_LLMS = Union[
  OpenAILLM,
  AzureOpenAILLM,
  AnthropicLLM,
  BedrockLLM,
  FireworksLLM,
  OllamaLLM,
  GeminiLLM,
]
VALID_TOOLS = Union[
  RetrievalTool,
  ActionServerTool,
  ArxivTool,
  ConneryTool,
  DallETool,
  DDGSearchTool,
  SecFilingsTool,
  PressReleasesTool,
  PubMedTool,
  TavilySearchTool,
  TavilyAnswerTool,
  YouSearchTool,
  WikipediaTool,
]

@dataclass
class AgentArgs:
  tools: List[BaseTool]
  checkpoint: BaseCheckpointSaver
  system_message: str = ''
  is_interrupt: bool = False

@dataclass
class ToolArgs:
  assistant_id: int
  manager: DjangoManager
  embedding: Any

class AgentType(models.IntegerChoices):
  OPENAI    = 1, gettext_lazy('Open AI')
  AZURE     = 2, gettext_lazy('Azure')
  ANTHROPIC = 3, gettext_lazy('Anthropic (Claude 2)')
  BEDROCK   = 4, gettext_lazy('Amazon Bedrock')
  FIREWORKS = 5, gettext_lazy('Fireworks (Mixtral)')
  OLLAMA    = 6, gettext_lazy('Ollama')
  GEMINI    = 7, gettext_lazy('GEMINI')

  def __str__(self) -> str:
    return f'{self.label}'

  @property
  def _llm_type(self) -> VALID_LLMS:
    # Patterns of LLM's chatbot
    lookup = {
      AgentType.OPENAI:    OpenAILLM,
      AgentType.AZURE:     AzureOpenAILLM,
      AgentType.ANTHROPIC: AnthropicLLM,
      AgentType.BEDROCK:   BedrockLLM,
      AgentType.FIREWORKS: FireworksLLM,
      AgentType.OLLAMA:    OllamaLLM,
      AgentType.GEMINI:    GeminiLLM,
    }

    return lookup[self]

  @property
  def _executor_type(self) -> Union[ToolExecutor, XmlExecutor]:
    # Patterns of executor
    lookup = {
      AgentType.OPENAI:    ToolExecutor,
      AgentType.AZURE:     ToolExecutor,
      AgentType.ANTHROPIC: ToolExecutor,
      AgentType.BEDROCK:   XmlExecutor,
      AgentType.FIREWORKS: ToolExecutor,
      AgentType.OLLAMA:    ToolExecutor,
      AgentType.GEMINI:    ToolExecutor,
    }

    return lookup[self]

  @staticmethod
  def get_embedding_choices() -> tuple[int, str]:
    # Filtering choices
    _invalids = [AgentType.ANTHROPIC]
    invalid_vals = [member.value for member in _invalids]
    _choices = [(value, label) for value, label in AgentType.choices if value not in invalid_vals]

    return _choices

  @staticmethod
  def get_embedding_validator() -> Callable[[int], int]:
    valids = [value for value, _ in AgentType.get_embedding_choices()]
    invalids = [(value, label) for value, label in AgentType.choices if value not in valids]

    @wraps(AgentType.get_embedding_validator)
    def validator(value):
      matched = [(item, label) for item, label in invalids if item == value]

      if len(matched) > 0:
        _, label = matched[0]

        raise ValidationError(
          gettext_lazy(f'{label} is the invalid AgentType'),
          params={'value': value}
        )

      return value

    return validator

  @classmethod
  def get_llm_fields(cls, agent_id: int, config: Dict, is_embedded: bool = False) -> List[LocalField]:
    _self = cls(agent_id)
    instance = _self._llm_type(**config)
    fields = instance.get_fields(is_embedded=is_embedded)

    return fields

  @classmethod
  def get_executor(
    cls,
    agent_id: int,
    config: Dict,
    args: AgentArgs,
  ) -> Union[ToolExecutor, XmlExecutor]:
    _self = cls(agent_id)
    instance = _self._llm_type(**config)
    llm = instance.get_llm(is_embedded=False)
    instance = _self._executor_type(llm, args.tools, args.is_interrupt, args.checkpoint)
    executor = instance.get_app(args.system_message)

    return executor

  @classmethod
  def get_embedding(
    cls,
    agent_id: int,
    config: Dict,
  ) -> Any:
    _self = cls(agent_id)
    instance = _self._llm_type(**config)
    embedding = instance.get_llm(is_embedded=True)

    return embedding

class ToolType(models.IntegerChoices):
  RETRIEVER          =  1, gettext_lazy('Retriever')
  ACTION_SERVER      =  2, gettext_lazy('Action Server')
  ARXIV              =  3, gettext_lazy('Arxiv')
  CONNERY_ACTION     =  4, gettext_lazy('Connery Action')
  DALLE_TOOL         =  5, gettext_lazy('Dall-E Tool')
  DDG_SEARCH         =  6, gettext_lazy('DuckDuckGo Search')
  KAY_SEC_FILINGS    =  7, gettext_lazy('SEC Filings (Kay AI)')
  KAY_PRESS_RELEASES =  8, gettext_lazy('Press Releases (Kay AI)')
  PUBMED             =  9, gettext_lazy('PubMed')
  TAVILY_SEARCH      = 10, gettext_lazy('Tavily Search')
  TAVILY_ANSWER      = 11, gettext_lazy('Tavily Answer')
  YOU_SEARCH         = 12, gettext_lazy('You Search')
  WIKIPEDIA          = 13, gettext_lazy('Wikipedia')

  def __str__(self) -> str:
    return f'{self.label}'

  @property
  def _tool_type(self) -> VALID_TOOLS:
    # Patterns of tool
    lookup = {
      ToolType.RETRIEVER:          RetrievalTool,
      ToolType.ACTION_SERVER:      ActionServerTool,
      ToolType.ARXIV:              ArxivTool,
      ToolType.CONNERY_ACTION:     ConneryTool,
      ToolType.DALLE_TOOL:         DallETool,
      ToolType.DDG_SEARCH:         DDGSearchTool,
      ToolType.KAY_SEC_FILINGS:    SecFilingsTool,
      ToolType.KAY_PRESS_RELEASES: PressReleasesTool,
      ToolType.PUBMED:             PubMedTool,
      ToolType.TAVILY_SEARCH:      TavilySearchTool,
      ToolType.TAVILY_ANSWER:      TavilyAnswerTool,
      ToolType.YOU_SEARCH:         YouSearchTool,
      ToolType.WIKIPEDIA:          WikipediaTool,
    }

    return lookup[self]

  @classmethod
  def get_config_field(
    cls,
    tool_id: int,
    config: Dict,
  ) -> List[LocalField]:
    _self = cls(tool_id)

    if _self == ToolType.RETRIEVER:
      retriever_config = {
        'assistant_id': 0,
        'manager': None,
        'strategy': None,
        'embeddings': None,
        'search_kwargs': config,
      }
      instance = _self._tool_type(retriever_config)
    else:
      instance = _self._tool_type(config)
    fields = instance.get_config_fields()

    return fields

  @classmethod
  def get_tool(
    cls,
    tool_id: int,
    config: Dict,
    args: ToolArgs,
  ) -> List[BaseTool]:
    _self = cls(tool_id)

    if _self == ToolType.RETRIEVER:
      retriever_config = {
        'assistant_id': args.assistant_id,
        'manager': args.manager,
        'strategy': args.embedding.get_distance_strategy(),
        'embeddings': args.embedding.get_embedding(),
        'search_kwargs': config,
      }
      instance = _self._tool_type(retriever_config)
    else:
      instance = _self._tool_type(config)
    tools = instance.get_tools()

    return tools
