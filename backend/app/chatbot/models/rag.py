from typing import List, Union
from django.db import models
from django.utils.translation import gettext_lazy
from django.contrib.auth import get_user_model
from .agents import GAIType, ToolType, BaseTool

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
  DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'
  chatbot = models.IntegerField(
    choices=GAIType.choices,
    default=GAIType.OPENAI,
    help_text=gettext_lazy('Chatbot type'),
  )

  def __str__(self):
    return f'{self.name} ({self.chatbot})'

  def get_executer(
    self, 
    system_message: str = Chatbot.DEFAULT_SYSTEM_MESSAGE, 
    tools: List[BaseTool], 
    is_interrupt=False
  ):
    executer = GAIType.get_executer(self.chatbot, self.config, system_message, tools, is_interrupt)

    return executer

class Embedding(_BaseConfig):
  emb = models.IntegerField(
    choices=GAIType.get_embedding_choices(),
    default=GAIType.OPENAI,
    validators=[GAIType.get_embedding_validator()],
    help_text=gettext_lazy('Embedding type'),
  )

  def __str__(self):
    return f'{self.name} ({self.emb})'

  def get_embedding(self):
    embedding = GAIType.get_embedding(self.emb)

    return embedding

class Tool(_BaseConfig):
  tool = models.IntegerField(
    choices=ToolType.choices,
    default=ToolType.RETRIEVER,
    help_text=gettext_lazy('Tool type'),
  )

  def __str__(self):
    return f'{self.name} ({self.tool})'

  def get_tool(self, vector_store=None):
    tool = ToolType.get_tool(self.tool, self.config, vector_store=vector_store)

    return tool