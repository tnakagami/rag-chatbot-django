from typing import Dict, List, Union
from django.db import models
from django.utils.translation import gettext_lazy
from django.contrib.auth import get_user_model
from .agents import AgentArgs, ToolArgs, AgentType, ToolType

User = get_user_model()

class BaseConfig(models.Model):
  user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    blank=True,
    help_text=gettext_lazy('Config owner'),
  )
  name = models.CharField(
    max_length=255,
    help_text=gettext_lazy('Config name'),
  )
  config = models.JSONField(
    blank=True,
    null=True,
    help_text=gettext_lazy('Config'),
  )

class Agent(BaseConfig):
  agent_type = models.IntegerField(
    choices=AgentType.choices,
    default=AgentType.OPENAI,
    help_text=gettext_lazy('Agent type'),
  )

  def __str__(self):
    return f'{self.name} ({self.chatbot})'

  def get_executor(self, args: AgentArgs):
    return AgentType.get_executor(self.agent_type, self.config, args)

class Embedding(BaseConfig):
  emb = models.IntegerField(
    choices=AgentType.get_embedding_choices(),
    default=AgentType.OPENAI,
    validators=[AgentType.get_embedding_validator()],
    help_text=gettext_lazy('Embedding type'),
  )

  def __str__(self):
    return f'{self.name} ({self.emb})'

  def get_embedding(self):
    embedding = AgentType.get_embedding(self.emb, self.config)

    return embedding

class Tool(BaseConfig):
  tool_type = models.IntegerField(
    choices=ToolType.choices,
    default=ToolType.RETRIEVER,
    help_text=gettext_lazy('Tool type'),
  )

  def __str__(self):
    return f'{self.name} ({self.tool})'

  def get_tool(self, args: ToolArgs):
    return ToolType.get_tool(self.tool_type, self.config, args)