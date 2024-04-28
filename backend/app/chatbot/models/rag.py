from typing import Dict, List, Union
from django.db import models
from django.utils.translation import gettext_lazy
from django.contrib.auth import get_user_model
from .agents import AgentArgs, ToolArgs, AgentType, ToolType

User = get_user_model()

class _BaseConfig(models.Model):
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

class Chatbot(_BaseConfig):
  chatbot = models.IntegerField(
    choices=AgentType.choices,
    default=AgentType.OPENAI,
    help_text=gettext_lazy('Chatbot type'),
  )

  def __str__(self):
    return f'{self.name} ({self.chatbot})'

  def get_executor(self, args: AgentArgs):
    return AgentType.get_executor(self.chatbot, self.config, args)

class Embedding(_BaseConfig):
  emb = models.IntegerField(
    choices=AgentType.get_embedding_choices(),
    default=AgentType.OPENAI,
    validators=[AgentType.get_embedding_validator()],
    help_text=gettext_lazy('Embedding type'),
  )

  def __str__(self):
    return f'{self.name} ({self.emb})'

  def get_embedding(self):
    embedding = AgentType.get_embedding(self.emb)

    return embedding

class Tool(_BaseConfig):
  tool = models.IntegerField(
    choices=ToolType.choices,
    default=ToolType.RETRIEVER,
    help_text=gettext_lazy('Tool type'),
  )

  def __str__(self):
    return f'{self.name} ({self.tool})'

  def get_tool(self, args: ToolArgs):
    return ToolType.get_tool(self.tool, self.config, args)