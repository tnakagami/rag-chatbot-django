from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Union, Any
from functools import wraps
from django.db import models
from django.utils.translation import gettext_lazy
from django.core.exceptions import ValidationError
from django.db.models import Manager as DjangoManager
from langchain.schema.embeddings import Embeddings
from langgraph.checkpoint import BaseCheckpointSaver
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

  def __str__(self):
    return f'{self.label}'

  @property
  def _llm_type(self):
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
  def _executor_type(self):
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

  @classmethod
  def get_embedding_choices(cls):
    # Filtering choices
    _invalids = [cls.ANTHROPIC]
    invalid_vals = [member.value for member in _invalids]
    _choices = [(value, label) for value, label in cls.choices if value not in invalid_vals]

    return _choices

  @classmethod
  def get_embedding_validator(cls):
    valids = [value for value, _ in cls.get_embedding_choices()]
    invalids = [(value, label) for value, label in cls.choices if value not in valids]

    @wraps(cls.get_embedding_validator)
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
  def get_llm_fields(cls, agent_id: int, is_embedded=False):
    _self = cls(agent_id)
    instance = _self._llm_type()
    fields = instance.get_fields(is_embedded=is_embedded)

    return fields

  @classmethod
  def get_executor(
    cls,
    agent_id: int,
    config: Dict,
    args: AgentArgs,
  ):
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
  ):
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

  def __str__(self):
    return f'{self.label}'

  @property
  def _tool_type(self):
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
  ):
    _self = cls(tool_id)
    instance = _self._tool_type(config)
    fields = instance.get_config_fields()

    return fields

  @classmethod
  def get_tool(
    cls,
    tool_id: int,
    config: Dict,
    args: ToolArgs,
  ):
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
