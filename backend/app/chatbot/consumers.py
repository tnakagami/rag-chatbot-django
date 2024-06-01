import logging
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Union
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from langchain.pydantic_v1 import ValidationError
from langchain_core.messages import AnyMessage, BaseMessage, message_chunk_to_message
from .models import Thread

g_logger = logging.getLogger(__name__)
g_controllers = {}

class InputMessageValidationError(Exception):
  pass

class _ChatController:
  def __init__(self, app, pk):
    self.app = app
    self.thread_id = pk

  def is_stream(self):
    return app._is_streaming

  def _formatter(self, event_type: str, data: Any) -> Dict[str, Any]:
    return {'event': event_type, 'data': data}

  async def _astream_state(self, inputs: Union[Sequence[AnyMessage], Dict[str, Any]]):
    root_run_id: Optional[str] = None
    messages: dict[str, BaseMessage] = {}

    async for response in self.app.astream_events(inputs, {}, version='v1', stream_mode='values', exclude_tags=['nostream']):
      run_id = response['run_id']

      if response['event'] == 'on_chat_start' and not root_run_id:
        root_run_id = run_id
        yield run_id

      elif response['event'] == 'on_chat_stream' and root_run_id == run_id:
        new_messages: list[BaseMessage] = []

        # response['data']['chunk'] is Sequence[AnyMessage] or a Dict[str, Any]
        chunks: Union[Sequence[AnyMessage], Dict[str, Any]] = response['data']['chunk']

        if isinstance(chunks, dict):
          chunks = chunks['messages']

        for msg in chunks:
          msg_id = msg['id'] if isinstance(msg, dict) else msg.id

          if msg_id in messages and msg == messages[msg_id]:
            continue
          else:
            messages[msg_id] = msg
            new_messages += [msg]

        if new_messages:
          yield new_messages

      elif response['event'] == 'on_chat_stream':
        msg: BaseMessage = response['data']['chunk']
        msg_id = msg['id'] if isinstance(msg, dict) else msg.id

        if msg_id not in messages:
          messages[msg_id] = msg
        else:
          messages[msg_id] += msg

        yield [messages[msg_id]]

  def _validate(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]) -> None:
    try:
      self.app.get_input_schema().validate(message)
    except ValidationError as ex:
      err_msg: str = str(ex)
      g_logger.error(f'ChatController[validate]{err_msg}')
      raise InputMessageValidationError(err_msg)

  async def aget_thread_state(self) -> Dict[str, Any]:
    config = {
      'configurable': {
        'thread_id': self.thread_id,
      }
    }
    # state: the instance of langgraph.pregel.types.StateSnapshot class
    snapshot = await self.app.aget_state(config)
    state = {
      'values': snapshot.values,
      'next': snapshot.next,
    }

    return self._formatter('history', state)

  async def ainvoke(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]) -> Dict[str, Any]:
    try:
      self._validate(message)
      result = await self.app.ainvoke(message)
    except InputMessageValidationError as ex:
      result = self._formatter('error', [str(ex)])
    except Exception as ex:
      err_msg = str(ex)
      g_logger.error(f'ChatController[ainvoke]{err_msg}')
      result = self._formatter('error', [err_msg])

    return result

  async def astream(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]):
    try:
      self._validate(message)

      async for chunk in self._astream_state(message):
        if isinstance(chunk, str):
          # For example, the matched chunk is `run_id`.
          yield self._formatter('metadata', [chunk])
        else:
          yield self._formatter('data', [message_chunk_to_message(target) for target in chunk])
    except InputMessageValidationError as ex:
      yield self._formatter('error', [str(ex)])
    except Exception as ex:
      err_msg = str(ex)
      g_logger.error(f'ChatController[astream]{err_msg}')
      yield self._formatter('error', [err_msg])

    yield self._formatter('end', [])

# ============
# = Consumer =
# ============
class ThreadConsumer(AsyncJsonWebsocketConsumer):
  def __init__(self, *args, **kwargs):
    self.thread = None
    super().__init__(*ags, **kwargs)

  async def connect(self):
    try:
      user = self.scope['user']
      pk = int(self.scope['rul_route']['kwargs']['thread_pk'])
      self.group_name = f'thread{pk}'
      self.thread = await database_sync_to_async(Thread.objects.get_or_none)(pk=pk)
      is_owner = await database_sync_to_async(self.thread.is_owner)(user)

      if is_owner:
        await self.accept()
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        executor = await database_sync_to_async(self.thread.get_executor)()
        g_controllers[self.group_name] = _ChatController(app=executor, pk=pk)

    except Exception as ex:
      raise Exception(ex)

  async def disconnect(self, close_code):
    await self.channel_layer.group_discard(self.group_name, self.channel_name)
    await self.close()
    del g_controllers[self.group_name]

  async def get_history(self, controller, **kwargs):
    response = await controller.aget_thread_state()
    await self.callback(response)

  async def send_message(self, controller, message, **kwargs):
    if controller.is_stream():
      async for response in controller.astream(message):
        await self.callback(response)
    else:
      response = await controller.ainvoke(message)
      await self.callback(response)

  # Receive message from websocket
  async def reveive_json(self, content):
    command = content.pop('command')
    controller = g_controllers.get(self.group_name)
    # The function list to execute command
    handlers = {
      'get_history': self.get_history,
      'send_message': self.send_message,
    }

    try:
      await handlers[command](controller, **content)
    except Exception as ex:
      raise Exception(ex)

  async def callback(self, response):
    # Set this class's method to send the response
    response.update({'type': 'send_response'})
    await self.channel_layer.group_send(self.group_name, response)

  async def send_response(self, event):
    try:
      await self.send_json(content=event)
    except Exception as ex:
      g_logger.warn(f'Thread[send_response]{ex}')
      raise Exception(ex)