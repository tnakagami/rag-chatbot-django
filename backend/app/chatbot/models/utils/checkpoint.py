import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Iterator, Optional, Union
from django.db.models import Manager as DjangoManager
from django.utils.timezone import localtime
from asgiref.sync import sync_to_async
from langchain_core.runnables.config import run_in_executor
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointAt,
    CheckpointThreadTs,
    CheckpointTuple,
    SerializerProtocol,
)

@dataclass
class _ThreadConfig:
  thread_id: int
  thread_ts: Union[datetime, None] = None

  def __init__(self, thread_id: int, thread_ts: Union[datetime, str, None] = None):
    self.thread_id = thread_id

    if thread_ts:
      self.thread_ts = datetime.fromisoformat(thread_ts) if isinstance(thread_ts, str) else thread_ts
    else:
      self.thread_ts = None

  def get_config_dict(self) -> Union[dict, None]:
    if self.thread_ts:
      config = {
        'configrable': {
          'thread_id': self.thread_id,
          'thread_ts': self.thread_ts.isoformat(),
        }
      }
    else:
      config = None

    return config

class DjangoPostgresCheckpoint(BaseCheckpointSaver):
  def __init__(
    self,
    *,
    serde: Optional[SerializerProtocol] = pickle,
    at: Optional[CheckpointAt] = CheckpointAt.END_OF_STEP,
    manager: DjangoManager
  ):
    super().__init__(serde=serde, at=at)
    self.manager = manager

  @property
  def config_specs(self) -> list[ConfigurableFieldSpec]:
    return [
      ConfigurableFieldSpec(
        id='thread_id',
        annotation=Optional[int],
        name='Thread ID',
        description=None,
        default=None,
        is_shared=True,
      ),
      CheckpointThreadTs,
    ]

  def _convert_django_timestamp(self, timestamp, is_localtime=False) -> Union[None, localtime, datetime]:
    if timestamp:
      out = localtime(timestamp) if is_localtime else timestamp
    else:
      out = None

    return out

  def _create_checkpoint_tuple(self, instance) -> Union[None, CheckpointTuple]:
    if instance:
      thread_id = instance.thread.pk
      current_time = self._convert_django_timestamp(instance.current_time, is_localtime=True)
      previous_time = self._convert_django_timestamp(instance.previous_time, is_localtime=True)
      current = _ThreadConfig(thread_id=thread_id, thread_ts=current_time).get_config_dict()
      previous = _ThreadConfig(thread_id=thread_id, thread_ts=previous_time).get_config_dict()
      result = CheckpointTuple(config=current, checkpoint=instance.checkpoint, parent_config=previous)
    else:
      result = None

    return result

  def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
    if value := self.get_tuple(config):
      return value.checkpoint

  def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
    runnable_config = config['configrable']
    target = _ThreadConfig(
      thread_id=runnable_config.get('thread_id'),
      thread_ts=runnable_config.get('thread_ts'),
    )
    queryset = self.manager.collect_checkpoints(target.thread_id, target.thread_ts)
    result = self._create_checkpoint_tuple(queryset.first())

    return result

  def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
    runnable_config = config['configrable']
    target = _ThreadConfig(thread_id=runnable_config.get('thread_id'))
    queryset = self.manager.collect_checkpoints(target.thread_id, target.thread_ts)

    for instance in queryset.iterator():
      result = self._create_checkpoint_tuple(instance)

      yield result

  def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
    runnable_config = config['configrable']
    thread_id = runnable_config.get('thread_id')
    current = _ThreadConfig(thread_id=thread_id, thread_ts=checkpoint['ts'])
    previous = _ThreadConfig(thread_id=thread_id, thread_ts=runnable_config.get('thread_ts'))
    self.manager.update_or_create(
      thread_id=current.thread_id,
      current_time=self._convert_django_timestamp(current.thread_ts, is_localtime=False),
      previous_time=self._convert_django_timestamp(previous.thread_ts, is_localtime=False),
      checkpoint=checkpoint,
    )
    result = current.get_config_dict()

    return RunnableConfig(result)

  async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
    if value := await self.aget_tuple(config):
      return value.checkpoint

  async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
    return await run_in_executor(None, self.get_tuple, config)

  async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
    runnable_config = config['configrable']
    target = _ThreadConfig(thread_id=runnable_config.get('thread_id'))
    queryset = await sync_to_async(self.manager.collect_checkpoints)(target.thread_id, target.thread_ts)

    async for instance in queryset.aiterator():
      result = await sync_to_async(self._create_checkpoint_tuple)(instance)

      yield result

  async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
    return await run_in_executor(None, self.put, config, checkpoint)
