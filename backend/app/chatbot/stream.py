import json
import orjson
import functools
import logging
from typing import Any, Dict, Optional, Sequence, Union
from langchain.pydantic_v1 import ValidationError
from langchain_core.messages import AnyMessage, BaseMessage, message_chunk_to_message

g_logger = logging.getLogger(__name__)

class InputMessageValidationError(Exception):
  pass

class ChatbotController:
  def __init__(self, app, pk):
    def _default(obj):
      if hasattr(obj, 'dict') and callable(obj.dict):
        return obj.dict()
      raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    # Set class member's variable
    self.app = app
    self.thread_id = pk
    self.dumps = functools.partial(orjson.dumps, default=_default)

  def _is_stream(self) -> bool:
    return self.app._is_streaming

  def _converter(self, response: Any) -> str:
    serialized = json.dumps(response)

    return f'{serialized}\n\n'.encode('utf-8')

  def _formatter(self, event_type: str, data: Any) -> Dict[str, Any]:
    return {'event': event_type, 'data': data}

  def _get_error(self, err_msg: str) -> Any:
    return orjson.dumps({'status_code': 500, 'error': err_msg}).decode()

  async def _astream_state(self, inputs: Union[Sequence[AnyMessage], Dict[str, Any]]) -> Any:
    root_run_id: Optional[str] = None
    messages: dict[str, BaseMessage] = {}
    config = self._get_config()

    async for response in self.app.astream_events(inputs, config, version='v1', stream_mode='values', exclude_tags=['nostream']):
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
        msg_id: Optional[str] = msg['id'] if isinstance(msg, dict) else msg.id

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

  def _get_config(self) -> Dict[str, Any]:
    return {
      'configurable': {
        'thread_id': self.thread_id,
      }
    }

  def _generate_response(self, collected_messages: list[AnyMessage]):
    ids: list[str] = []
    data: list[str] = []
    # Extract content from AnyMessage
    message_types: Dict[str, str] = {
      'HumanMessage': 'HUMAN',
      'AIMessage': 'AI',
      'FunctionMessage': 'FUNCTION',
      'ToolMessage': 'TOOL',
    }
    for msg in collected_messages:
      if isinstance(msg, BaseMessage):
        msg_id: str = msg.id
        content: str = str(msg.content)
        datatype: Optional[str] = message_types.get(msg.__class__.__name__, None)

        if msg_id not in ids and content not in data:
          extracted: Dict[str, Any] = {'content': content, 'type': datatype, 'id': msg_id}
          yield extracted
          data.append(content)
          ids.append(msg_id)

  async def aget_thread_state(self) -> Dict[str, Any]:
    config: Dict[str, Any] = self._get_config()
    # state: the instance of langgraph.pregel.types.StateSnapshot class
    snapshot = await self.app.aget_state(config)
    # Create response
    for data in self._generate_response(snapshot.values):
      yield self._formatter('history', [data])

  async def ainvoke(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]) -> Dict[str, Any]:
    try:
      self._validate(message)
      config: Dict[str, Any] = self._get_config()
      outputs: Any = await self.app.ainvoke(message, config)
      # Create response
      for data in self._generate_response(outputs):
        yield self._formatter('data', [data])
    except InputMessageValidationError as ex:
      yield self._formatter('error', self._get_error(str(ex)))
    except Exception as ex:
      err_msg: str = str(ex)
      g_logger.error(f'ChatController[ainvoke]{err_msg}')
      yield self._formatter('error', self._get_error(err_msg))

  async def astream(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]):
    try:
      self._validate(message)

      async for chunk in self._astream_state(message):
        if isinstance(chunk, str):
          # For example, the matched chunk is `run_id`.
          data: Any = orjson.dumps({'run_id': chunk}).decode()
          yield self._formatter('metadata', data)
        else:
          data: Any = self.dumps([message_chunk_to_message(target) for target in chunk]).decode()
          yield self._formatter('data', data)
    except InputMessageValidationError as ex:
      yield self._formatter('error', self._get_error(str(ex)))
    except Exception as ex:
      err_msg: str = str(ex)
      g_logger.error(f'ChatController[astream]{err_msg}')
      yield self._formatter('error', self._get_error(err_msg))

    yield self._formatter('end', {})

  async def event_stream(self, contents):
    if 'chat_history' == contents['type']:
      async for response in self.aget_thread_state():
        yield self._converter(response)
    else:
      message: Union[Sequence[AnyMessage], Dict[str, Any]] = contents['message']
      callback: Any = self.astream if self._is_stream() else self.ainvoke

      async for response in callback(message):
        yield self._converter(response)