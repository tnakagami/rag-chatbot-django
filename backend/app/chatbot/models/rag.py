from typing import Dict, List, Union, Tuple
from django.db import models
from django.utils.translation import gettext_lazy
from django.contrib.auth import get_user_model
from pgvector.django import VectorField, L2Distance, CosineDistance, MaxInnerProduct
from picklefield.fields import PickledObjectField
from .agents import AgentArgs, ToolArgs, AgentType, ToolType
from .utils.checkpoint import DjangoPostgresCheckpoint
from .utils.vectorstore import DistanceStrategy

User = get_user_model()

class BaseManager(models.Manager):
  def get_or_none(self, **kwargs):
    try:
      return self.get_queryet().get(**kwargs)
    except self.model.DoesNotExist:
      return None

class BaseConfig(models.Model):
  name = models.CharField(
    gettext_lazy('Config name'),
    max_length=255,
    help_text=gettext_lazy('255 characters or fewer.'),
  )
  config = models.JSONField(
    gettext_lazy('Config'),
    blank=True,
    null=True,
    help_text=gettext_lazy('Required: JSON format'),
  )

  objects = BaseManager()

class Agent(BaseConfig):
  user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    blank=True,
    related_name='agent_configs',
    verbose_name=gettext_lazy('Agent config owner'),
  )
  agent_type = models.IntegerField(
    gettext_lazy('Agent type'),
    choices=AgentType.choices,
    default=AgentType.OPENAI,
  )

  def __str__(self):
    return f'{self.name} ({self.chatbot})'

  def get_executor(self, args: AgentArgs):
    return AgentType.get_executor(self.agent_type, self.config, args)

class Embedding(BaseConfig):
  class DistanceType(models.IntegerChoices):
    EUCLIDEAN         = 1, gettext_lazy('Euclidean')
    COSINE            = 2, gettext_lazy('Cosine')
    MAX_INNER_PRODUCT = 3, gettext_lazy('Max inner product')

    @property
    def _strategy(self):
      # Patterns of distance strategy
      lookup = {
        Embedding.DistanceType.EUCLIDEAN:         DistanceStrategy.EUCLIDEAN,
        Embedding.DistanceType.COSINE:            DistanceStrategy.COSINE,
        Embedding.DistanceType.MAX_INNER_PRODUCT: DistanceStrategy.MAX_INNER_PRODUCT,
      }

      return lookup[self]

  user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    blank=True,
    related_name='embedding_configs',
    verbose_name=gettext_lazy('Embedding config owner'),
  )
  distance_strategy = models.IntegerField(
    gettext_lazy('Distance strategy'),
    choices=DistanceType.choices,
    default=DistanceType.COSINE,
  )
  emb = models.IntegerField(
    gettext_lazy('Embedding type'),
    choices=AgentType.get_embedding_choices(),
    default=AgentType.OPENAI,
    validators=[AgentType.get_embedding_validator()],
  )

  def __str__(self):
    return f'{self.name} ({self.emb})'

  def get_embedding(self):
    embedding = AgentType.get_embedding(self.emb, self.config)

    return embedding

  def get_distance_strategy(self):
    return DistanceType(self.distance_strategy)._strategy

class Tool(BaseConfig):
  user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    blank=True,
    related_name='tool_configs',
    verbose_name=gettext_lazy('Tool config owner'),
  )
  tool_type = models.IntegerField(
    gettext_lazy('Tool type'),
    choices=ToolType.choices,
    default=ToolType.RETRIEVER,
  )

  def __str__(self):
    return f'{self.name} ({self.tool})'

  def get_tool(self, args: ToolArgs):
    return ToolType.get_tool(self.tool_type, self.config, args)

class Assistant(models.Model):
  user = models.ForeignKey(
    User,
    on_delete=models.CASCADE,
    blank=True,
    related_name='assistants',
    verbose_name=gettext_lazy('Assistant owner'),
  )
  name = models.CharField(
    gettext_lazy('Assistant name'),
    max_length=255,
    help_text=gettext_lazy('255 characters or fewer.'),
  )
  system_message = models.TextField(
    help_text=gettext_lazy('System message'),
    blank=True,
    default=gettext_lazy('You are a helpful assistant.'),
  )
  agent = models.ForeignKey(
    Agent,
    on_delete=models.CASCADE,
    blank=False,
    null=False,
    verbose_name=gettext_lazy('Agent used in RAG'),
  )
  embedding = models.ForeignKey(
    Embedding,
    on_delete=models.CASCADE,
    verbose_name=gettext_lazy('Embedding used in RAG'),
  )
  tools = models.ManyToManyField(
    Tool,
    related_name='tools',
    verbose_name=gettext_lazy('Tools used in RAG'),
  )
  is_interrupt = models.BooleanField(
    gettext_lazy('Interrupt flag in LangGraph workflow'),
    default=False,
    help_text=gettext_lazy('If True, Interrupt before the specific node (e.g. "action" node) when the workflow is stopped with human intervention.'),
  )

  objects = BaseManager()

  def __str__(self):
    return f'{self.name}'

  def get_assistant(self):
    tool_args = ToolArgs(
      assistant_id=self.pk,
      manager=EmbeddingStore.objects,
      embedding=self.embedding,
    )
    tools = []
    # Collect each tool instance
    for target in self.tools.all():
      _target_tools = target.get_tool(tool_args)

      if isinstance(_target_tools, list):
        tools.expand(_target_tools)
      else:
        tools.append(_target_tools)
    # Get the executor
    agent_args = AgentArgs(
      tools=tools,
      checkpoint=DjangoPostgresCheckpoint(manager=LangGraphCheckpoint.objects),
      system_message=self.system_message,
      is_interrupt=self.is_interrupt,
    )
    assistant = self.agent.get_executor(agent_args)

    return assistant

class Thread(models.Model):
  assistant = models.ForeignKey(
    Assistant,
    on_delete=models.CASCADE,
    blank=True,
    related_name='threads',
    verbose_name=gettext_lazy('Base assistant of thread'),
  )
  name = models.CharField(
    gettext_lazy('Thread name'),
    max_length=255,
    help_text=gettext_lazy('255 characters or fewer.'),
  )

  objects = BaseManager()

class EmbeddingStoreQuerySet(models.QuerySet):
  def similarity_search_with_distance_by_vector(self, embedded_query, assistant_id, distance_strategy):
    # In the case of L2Distance:
    #  distance(x_vec, y_vec) = np.linalg.norm(x_vec - y_vec, axis=0)
    # In the case of CosineDistance:
    #  distance(x_vec, y_vec) = np.dot(x_vec, y_vec) / np.linalg.norm(x_vec, axis=0) / np.linalg.norm(y_vec, axis=0)
    # In the case of MaxInnerProduct:
    #  distance(x_vec, y_vec) = np.dot(x_vec, y_vec)
    patterns = {
      DistanceStrategy.EUCLIDEAN:         L2Distance,
      DistanceStrategy.COSINE:            CosineDistance,
      DistanceStrategy.MAX_INNER_PRODUCT: MaxInnerProduct,
    }
    strategy = patterns.get(distance_strategy, CosineDistance)
    queryset = self.filter(assistant__pk=assistant_id) \
                   .annotate(distance=strategy('embedding', embedded_query)) \
                   .order_by('distance')

    return queryset

class EmbeddingStoreManager(BaseManager, models.Manager.from_queryset(EmbeddingStoreQuerySet)):
  def create(self, assistant_id, *args, **kwargs):
    assistant = Assistant.objects.get(pk=assistant_id)

    return super().create(assistant=assistant, *args, **kwargs)

class EmbeddingStore(models.Model):
  assistant = models.ForeignKey(
    Assistant,
    on_delete=models.CASCADE,
    related_name='embedding_stores',
    verbose_name=gettext_lazy('Owner'),
  )
  embedding = VectorField(
    gettext_lazy('Embedding vector'),
  )
  document = models.TextField(
    gettext_lazy('Document'),
    blank=True,
    null=True,
  )

  objects = EmbeddingStoreManager()

  def __str__(self):
    username = self.assistant.user.get_short_name()
    assistant_name = str(self.assistant)

    return f'{assistant_name}({username})'

class LangGraphCheckpointQuerySet(models.QuerySet):
  def collect_checkpoints(self, thread_id, thread_ts=None):
    if thread_ts:
      queryset = self.filter(thread__pk=thread_id, current_time=thread_ts)
    else:
      queryset = self.filter(thread__pk=thread_id)

    return queryset.order_by('-current_time')

class LangGraphCheckpointManager(BaseManager, models.Manager.from_queryset(LangGraphCheckpointQuerySet)):
  def update_or_create(self, thread_id, *args, **kwargs):
    thread = Thread.objects.get(pk=thread_id)

    return super().update_or_create(thread=thread, *args, **kwargs)

class LangGraphCheckpoint(models.Model):
  thread = models.ForeignKey(
    Thread,
    on_delete=models.CASCADE,
    related_name='checkpoints',
    verbose_name=gettext_lazy('LangGraph checkpoint'),
  )
  current_time = models.DateTimeField(
    gettext_lazy('Current checkpoint time'),
    help_text=gettext_lazy('Required: ISO format'),
  )
  previous_time = models.DateTimeField(
    gettext_lazy('Previous checkpoint time'),
    blank=True,
    null=True,
    help_text=gettext_lazy('Required: ISO format'),
  )
  checkpoint = PickledObjectField(
    gettext_lazy('Checkpoint'),
    help_text=gettext_lazy('Required: Pickle instance'),
  )

  objects = LangGraphCheckpointManager()