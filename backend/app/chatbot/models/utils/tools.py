from typing import Optional, Dict
from typing_extensions import TypedDict as ToolConfig
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import Tool
from langchain_robocorp import ActionServerToolkit
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.connery import ConneryService
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.retrievers.kay import KayAiRetriever
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_community.tools.tavily_search import (
    TavilyAnswer as _TavilyAnswer,
    TavilySearchResults,
)
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.retrievers.you import YouRetriever
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from django.utils.translation import gettext_lazy

class _BaseTool(BaseModel):
  name: Optional[str]
  description: Optional[str]
  multi_use: Optional[bool] = False

  def get_tool(self):
    raise NotImplemented

class RetrievalTool(_BaseTool):
  name: str = Field(gettext_lazy('Retrieval'), const=True)
  description: str = Field(gettext_lazy('Look up information in uploaded files.', const=True))

class ActionServerTool(_BaseTool):
  name: str = Field(gettext_lazy('Action Server by Robocorp', const=True))
  description: str = Field(gettext_lazy('Run AI actions with Roborop Action Server'), const=True)
  multi_use: bool = Field(True, const=True)
  url: str = Field(gettext_lazy('Action Server url', const=False))
  api_key: str = Field(gettext_lazy('API key', const=False))

class ArxivTool(_BaseTool):
  name: str = Field(gettext_lazy('Arxiv'), const=True)
  description: str = Field(gettext_lazy('Searches Arxiv'), const=True)

class ConneryTool(_BaseTool):
  name: str = Field(gettext_lazy('AI Action Runner by Connery'), const=True)
  description: str = Field(gettext_lazy('Connect OpenGPTs to the real world with Connery'), const=True)
  api_key: str = Field(gettext_lazy('API key', const=False))

class DallETool(_BaseTool):
  name: str = Field(gettext_lazy('Image Generator (Dall-E)'), const=True)
  description: str = Field(gettext_lazy("Generates images from a text description using OpenAI's DALL-E model."), const=True)
  # Use OpenAI keys

class DDGSearchTool(_BaseTool):
  name: str = Field(gettext_lazy('DuckDuckGo Search'), const=True)
  description: str = Field(gettext_lazy('Searches the web with DuckDuckGo'), const=True)

class SecFilingsTool(_BaseTool):
  name: str = Field(gettext_lazy('SEC Filings (Kay.ai)'), const=True)
  description: str = Field(gettext_lazy('Searches through SEC filings using Kay.ai'), const=True)
  # Use KAI_API_KEY and need to custom python script

class PressReleasesTool(_BaseTool):
  name: str = Field(gettext_lazy('Press Releases (Kay.ai)'), const=True)
  description: str = Field(gettext_lazy('Searches through press releases using using Kay.ai'), const=True)
  # Use KAI_API_KEY and need to custom python script

class PubMedTool(_BaseTool):
  name: str = Field(gettext_lazy('PubMed'), const=True)
  description: str = Field(gettext_lazy('Searches PubMed'), const=True)

class TavilySearchTool(_BaseTool):
  name: str = Field(gettext_lazy('Tavily Search (with evidence)'), const=True)
  description: str = Field(gettext_lazy('Uses the Tavilysearch engine. Includes sources in the response.'), const=True)
  api_key: str = Field(gettext_lazy('API key', const=False))

class TavilyAnswerTool(_BaseTool):
  name: str = Field(gettext_lazy('Tavily Search (only answer)'), const=True)
  description: str = Field(gettext_lazy('Uses the Tavilysearch engine. This returns only the answer, no supporting evidence.'), const=True)
  api_key: str = Field(gettext_lazy('API key', const=False))

class YouSearchTool(_BaseTool):
  name: str = Field(gettext_lazy('You.com Search'), const=True)
  description: str = Field(gettext_lazy('Uses You.com search, optimized responses for LLMs.'), const=True)
  api_key: str = Field(gettext_lazy('API key', const=False))

class WikipediaTool(_BaseTool):
  name: str = Field(gettext_lazy('Wikipedia Search'), const=True)
  description: str = Field(gettext_lazy('Searches Wikipedia'), const=True)