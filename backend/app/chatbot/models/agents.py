import pickle
import dataclasses
from typing import Dict, Tuple, List
from django.db import models
from django.utils.translation import gettext_lazy
from django.core.exceptions import ValidationError
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
  TavilySearchTool,
  TavilyAnswerTool,
  YouSearchTool,
  WikipediaTool,
)

class GAIType(models.IntegerChoices):
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
      GAIType.OPENAI:    OpenAILLM,
      GAIType.AZURE:     AzureOpenAILLM,
      GAIType.ANTHROPIC: AnthropicLLM,
      GAIType.BEDROCK:   BedrockLLM,
      GAIType.FIREWORKS: FireworksLLM,
      GAIType.OLLAMA:    OllamaLLM,
      GAIType.GEMINI:    GeminiLLM,
    }

    return lookup[self]

  @property
  def _executer_type(self):
    # Patterns of executer
    lookup = {
      GAIType.OPENAI:    ToolExecutor,
      GAIType.AZURE:     ToolExecutor,
      GAIType.ANTHROPIC: ToolExecutor,
      GAIType.BEDROCK:   XmlExecutor,
      GAIType.FIREWORKS: ToolExecutor,
      GAIType.OLLAMA:    ToolExecutor,
      GAIType.GEMINI:    ToolExecutor,
    }

    return lookup[self]

  @classmethod
  def _get_invalid_types(cls):
    return [cls.ANTHROPIC]

  @classmethod
  def get_embedding_choices(cls):
    # Filtering choices
    invalid_vals = [member.value for member in cls._get_invalid_types()]
    _choices = [(value, label) for value, label in cls.choices if value not in invalid_vals]

    return _choices

  @classmethod
  def get_embedding_validator(cls):
    invalids = cls.get_embedding_choices()

    def validator(value):
      matched = [(item, label) for item, label in invalids if item != value]

      if matched:
        _, label = matched[0]
        err = f'{label} is the invalid GAIType'

        raise ValidationError(
          gettext_lazy(err),
          params={'value': value}
        )

    return validator

  @classmethod
  def get_llm_fields(cls, gai_id: int, is_embedded=False):
    instance = gai_id._llm_type()
    fields = instance.get_fields(is_embedded=is_embedded)

    return fields

  @classmethod
  def get_executer(
    cls,
    gai_id: int,
    config: Dict,
    system_message: str,
    tools: List[BaseTool],
    is_interrupt: bool = False,
    *argv: Tuple,
    **kwargs: Dict,
  ):
    _self = cls(gai_id)
    instance = _self._llm_type(**config)
    llm = instance.get_llm(is_embedded=False)
    instance = _self._executer_type(llm, tools, is_interrupt)
    executer = instance.get_app(system_message)

    return executer

  @classmethod
  def get_embedding(
    cls,
    gai_id: int,
    *argv: Tuple,
    **kwargs: Dict,
  ):
    _self = cls(gai_id)
    instance = _self._llm_type(**config)
    embedding = instance.get_llm(is_embedded=True)

    return embedding

class ToolType(models.IntegerChoices):
  from langchain_core.vectorstores import VectorStore
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

  @property
  def tool_type(self):
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
  def get_tool(
    cls,
    tool_id: int,
    config: Dict,
    vector_store: VectorStore,
    *argv: Tuple,
    **kwargs: Dict,
  ):
    _self = cls(tool_id)
    tool = None

    return tool
