import logging
from typing import Dict, List, Union, Tuple
from zoneinfo import ZoneInfo
from celery import states
from django.conf import settings
from django.db import models, NotSupportedError
from django.utils.translation import gettext_lazy
from django.contrib.auth import get_user_model
from django_celery_results.models import TaskResult
from pgvector.django import VectorField, L2Distance, CosineDistance, MaxInnerProduct
from picklefield.fields import PickledObjectField
from .agents import AgentArgs, ToolArgs, AgentType, ToolType
from .utils.checkpoint import DjangoPostgresCheckpoint
from .utils.vectorstore import DistanceStrategy, CustomVectorStore
from .utils.ingest import IngestBlobRunnable

User = get_user_model()
g_logger = logging.getLogger(__name__)

def convert_timezone(target_datatime, is_string=False):
  timezone = ZoneInfo(settings.TIME_ZONE)
  output = target_datatime.astimezone(timezone)

  if is_string:
    output = output.strftime('%Y-%m-%d %H:%M:%S.%f')

  return output

class BaseManager(models.Manager):
  def get_or_none(self, **kwargs):
    try:
      return self.get_queryset().get(**kwargs)
    except self.model.DoesNotExist:
      return None

  async def aget_or_none(self, **kwargs):
    try:
      return await self.get_queryset().aget(**kwargs)
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
    default=dict,
    help_text=gettext_lazy('Required: JSON format'),
  )

  objects = BaseManager()

  def get_shortname(self, name):
    max_len = 32

    if len(name) > max_len:
      ret = '{}...'.format(name[:max_len])
    else:
      ret = name

    return ret

  def get_config(self):
    return self.config

  @classmethod
  def get_config_form_args(cls, instance=None, default_type=AgentType.OPENAI):
    if instance is None:
      type_id, config = default_type, {}
    else:
      type_id, config = instance.get_config()

    return type_id, config

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
    agent_type = AgentType(self.agent_type)

    return f'{self.name} ({agent_type})'

  def is_owner(self, user):
    return self.user.pk == user.pk

  def get_shortname(self):
    return super().get_shortname(str(self))

  def get_config(self):
    return self.agent_type, super().get_config()

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
  emb_type = models.IntegerField(
    gettext_lazy('Embedding type'),
    choices=AgentType.embedding_choices,
    default=AgentType.OPENAI,
  )

  def __str__(self):
    emb_type = AgentType(self.emb_type)

    return f'{self.name} ({emb_type})'

  def is_owner(self, user):
    return self.user.pk == user.pk

  def get_shortname(self):
    return super().get_shortname(str(self))

  def get_config(self):
    return self.emb_type, super().get_config()

  def get_embedding(self):
    embedding = AgentType.get_embedding(self.emb_type, self.config)

    return embedding

  def get_distance_strategy(self):
    return Embedding.DistanceType(self.distance_strategy)._strategy

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
    tool_type = ToolType(self.tool_type)

    return f'{self.name} ({tool_type})'

  def is_owner(self, user):
    return self.user.pk == user.pk

  def get_shortname(self):
    return super().get_shortname(str(self))

  def get_config(self):
    return self.tool_type, super().get_config()

  @classmethod
  def get_config_form_args(cls, instance=None):
    return super().get_config_form_args(instance=instance, default_type=ToolType.RETRIEVER)

  def get_tool(self, args: ToolArgs):
    return ToolType.get_tool(self.tool_type, self.config, args)

class AssistantManager(BaseManager):
  def collection_with_docfiles(self, **kwargs):
    return self.get_queryset().prefetch_related('docfiles').filter(**kwargs)

  def collect_own_tasks(self, **kwargs):
    user = kwargs.get('user', None)
    assistant = kwargs.get('assistant', None)

    if user is None and assistant is None:
      raise NotSupportedError
    # Collect user's tasks
    params = {
      'user': user,
      'assistant': assistant,
    }
    content = ','.join([f'{key}={instance.pk}' for key, instance in params.items() if instance is not None])
    queryset = TaskResult.objects.exclude(status=states.SUCCESS).filter(task_kwargs__icontains=content)

    return queryset

class Assistant(models.Model):
  class Meta:
    ordering = ['pk']

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
    blank=True,
    verbose_name=gettext_lazy('Tools used in RAG'),
  )
  is_interrupt = models.BooleanField(
    gettext_lazy('Interrupt flag in LangGraph workflow'),
    blank=True,
    default=False,
    help_text=gettext_lazy('If True, Interrupt before the specific node (e.g. "action" node) when the workflow is stopped with human intervention.'),
  )

  objects = AssistantManager()

  def __str__(self):
    return f'{self.name}'

  def is_owner(self, user):
    return self.user.pk == user.pk

  def get_executor(self, docfile_ids=None):
    if docfile_ids is None or not isinstance(docfile_ids, list):
      docfile_ids = []

    tool_args = ToolArgs(
      assistant_id=self.pk,
      manager=EmbeddingStore.objects,
      embedding=self.embedding,
      docfile_ids=docfile_ids,
    )
    tools = []
    # Collect each tool instance
    for target in self.tools.all():
      _target_tools = target.get_tool(tool_args)

      if isinstance(_target_tools, list):
        tools.extend(_target_tools)
      else:
        tools.append(_target_tools)
    # Get the executor
    agent_args = AgentArgs(
      tools=tools,
      checkpoint=DjangoPostgresCheckpoint(manager=LangGraphCheckpoint.objects),
      system_message=self.system_message,
      is_interrupt=self.is_interrupt,
    )
    executor = self.agent.get_executor(agent_args)

    return executor

  def set_task_result(self, task_id, user_pk):
    assistant_pk = self.pk
    task = TaskResult.objects.get_task(task_id)
    task.task_name = f'embedding process ({self})'
    task.task_kwargs = f"{{'info': 'user={user_pk},assistant={assistant_pk}'}}"
    task.save()

class DocumentFileManager(BaseManager):
  def collect_own_files(self, user):
    return self.get_queryset().filter(assistant__user=user, is_active=True)

  def active(self):
    return self.filter(is_active=True)

class DocumentFile(models.Model):
  MAX_FILENAME_LENGTH = 255
  class Meta:
    ordering = ['pk']

  assistant = models.ForeignKey(
    Assistant,
    on_delete=models.CASCADE,
    blank=True,
    related_name='docfiles',
    verbose_name=gettext_lazy('Base assistant of document files'),
  )
  name = models.CharField(
    gettext_lazy('Document name'),
    max_length=MAX_FILENAME_LENGTH,
    blank=True,
    help_text=gettext_lazy(f'{MAX_FILENAME_LENGTH} characters or fewer.'),
  )
  is_active = models.BooleanField(
    gettext_lazy('Document status'),
    blank=True,
    default=False,
    help_text=gettext_lazy('If True, the embedding process is finished.'),
  )

  objects = DocumentFileManager()

  def __str__(self):
    return f'{self.name} ({self.assistant})'

  def is_owner(self, user):
    return self.assistant.is_owner(user)

  @staticmethod
  def get_valid_extensions():
    return ['.pdf', '.txt', '.html', '.docx']

  @classmethod
  def from_files(cls, assistant, filefields):
    store = CustomVectorStore(
      manager=EmbeddingStore.objects,
      strategy=assistant.embedding.get_distance_strategy(),
      embedding_function=assistant.embedding.get_embedding(),
    )
    runnable = IngestBlobRunnable(store=store, record_id=assistant.pk)
    ids = []

    for field in filefields:
      instance = cls.objects.create(
        assistant=assistant,
        name=field.name,
        is_active=False,
      )

      try:
        if field.closed:
          field.open(field.mode)
        # Create blob data
        blob = runnable.convert_input2blob(field)
        # Create embedding vector from blob data
        current_ids = runnable.invoke(blob, docfile_id=instance.pk)
        # Update document status
        instance.is_active = True
        instance.save()
        ids.extend(current_ids)
      except Exception as ex:
        g_logger.warn(f'DocumentFile[from_files]{ex}')
        instance.delete()

    return ids

class ThreadManager(BaseManager):
  def collect_own_threads(self, user):
    return self.get_queryset().prefetch_related('docfiles').filter(assistant__user=user)

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
  docfiles = models.ManyToManyField(
    DocumentFile,
    related_name='docfiles',
    blank=True,
    verbose_name=gettext_lazy('Document files used in RAG'),
  )

  objects = ThreadManager()

  def __str__(self):
    return f'{self.name} ({self.assistant})'

  def is_owner(self, user):
    return self.assistant.is_owner(user)

  def get_executor(self):
    docfile_ids = self.docfiles.all().values_list('pk', flat=True)
    executor = self.assistant.get_executor(docfile_ids=docfile_ids)

    return executor

class EmbeddingStoreQuerySet(models.QuerySet):
  def similarity_search_with_distance_by_vector(self, embedded_query, distance_strategy, **kwargs):
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
    assistant_id = kwargs.get('assistant_id', None)
    docfile_ids = kwargs.get('docfile_ids', None)

    if assistant_id is not None:
      if docfile_ids is None or not isinstance(docfile_ids, list) or len(docfile_ids) == 0:
        params = {'assistant__pk': assistant_id}
      else:
        params = {'assistant__pk': assistant_id, 'docfile__pk__in': docfile_ids}

      queryset = self.filter(**params) \
                     .annotate(distance=strategy('embedding', embedded_query)) \
                     .order_by('distance')
    else:
      queryset = self.none()

    return queryset

class EmbeddingStoreManager(BaseManager, models.Manager.from_queryset(EmbeddingStoreQuerySet)):
  def create(self, *args, **kwargs):
    assistant_id = kwargs.get('assistant_id', None)
    docfile_id = kwargs.get('docfile_id', None)

    if assistant_id is None or docfile_id is None:
      raise NotSupportedError

    assistant = Assistant.objects.get(pk=assistant_id)
    docfile = DocumentFile.objects.get(pk=docfile_id)

    return super().create(assistant=assistant, docfile=docfile, *args, **kwargs)

class EmbeddingStore(models.Model):
  assistant = models.ForeignKey(
    Assistant,
    on_delete=models.CASCADE,
    related_name='embedding_stores',
    verbose_name=gettext_lazy('Base assistant of embedding store'),
  )
  docfile = models.ForeignKey(
    DocumentFile,
    on_delete=models.CASCADE,
    related_name='embedding_stores',
    verbose_name=gettext_lazy('Base document file of embedding store'),
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

    return f'{self.assistant} ({username})'

class LangGraphCheckpointQuerySet(models.QuerySet):
  def collect_checkpoints(self, thread_id, thread_ts=None):
    if thread_ts is not None:
      queryset = self.filter(thread__pk=thread_id, current_time=thread_ts)
    else:
      queryset = self.filter(thread__pk=thread_id)

    return queryset.order_by('-current_time')

class LangGraphCheckpointManager(BaseManager, models.Manager.from_queryset(LangGraphCheckpointQuerySet)):
  def update_or_create(self, thread_id, current_time, *args, **kwargs):
    instance = self.get_or_none(thread__pk=thread_id, current_time=current_time)
    # In the caes of inserting the new checkpoint
    if instance is None:
      thread = Thread.objects.get_or_none(pk=thread_id)

      if thread is not None:
        instance = self.create(thread=thread, current_time=current_time, *args, **kwargs)
        created = True
      else:
        instance, created = None, False
    # In the caes of updating the checkpoint
    else:
      checkpoint = kwargs.get('checkpoint', None)
      created = False

      if checkpoint is not None:
        instance.checkpoint = checkpoint
        instance.save(update_fields=['checkpoint'])
      else:
        instance = None

    return instance, created

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