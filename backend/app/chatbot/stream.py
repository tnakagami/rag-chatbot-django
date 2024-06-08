import orjson
import functools
import logging
from typing import Any, Dict, Optional, Sequence, Union
from collections.abc import AsyncIterator
from dataclasses import dataclass
from langchain.pydantic_v1 import ValidationError
from langchain_core.messages import AnyMessage, BaseMessage, message_chunk_to_message

g_logger = logging.getLogger(__name__)

class InputMessageValidationError(Exception):
  pass

@dataclass
class ResponseMessage:
  id: str
  content: str
  type: str

@dataclass
class Error:
  error: str
  status_code: int = 500

@dataclass
class EventResponse:
  event: str
  data: Union[list[ResponseMessage], str, Error, Dict[str, Any]]

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

  def _converter(self, response: EventResponse) -> bytes:
    return self.dumps(response) + b'\n\n'

  async def _astream_state(self, inputs: Union[Sequence[AnyMessage], Dict[str, Any]]) -> AsyncIterator[Union[BaseMessage, list[BaseMessage], str]]:
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
        is_dict = isinstance(msg, dict)
        msg_id: Optional[str] = msg['id'] if is_dict else msg.id

        if msg_id not in messages:
          messages[msg_id] = msg
          result = [msg]
        else:
          if isinstance(messages[msg_id], dict):
            new_content = msg['content'] if is_dict else str(msg.content)
            content = messages[msg_id]['content'] + new_content
            messages.update({msg_id: {'content': content, 'id': msg_id}})
            result = [messages[msg_id]]
          else:
            new_content = msg['content'] if is_dict else msg
            template = messages[msg_id] + new_content
            messages[msg_id] = template.format_messages()
            # Update id
            for target in messages[msg_id]:
              target.id = msg_id
            result = messages[msg_id]

        yield result

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

  def _generate_response(self, collected_messages: list[Union[AnyMessage, Dict[str, Any]]]) -> AsyncIterator[ResponseMessage]:
    hashed: Dict[str, bool] = {}
    # Extract content from AnyMessage
    message_types: Dict[str, str] = {
      # For BaseMessage
      'HumanMessage': 'HUMAN',
      'AIMessage': 'AI',
      'FunctionMessage': 'FUNCTION',
      'ToolMessage': 'TOOL',
      # For dict
      'human': 'HUMAN',
      'ai': 'AI',
      'function': 'FUNCTION',
      'tool': 'TOOL',
    }
    for msg in collected_messages:
      if isinstance(msg, (BaseMessage, dict)):
        is_dict = isinstance(msg, dict)
        type_name = msg.get('type', '') if is_dict else msg.__class__.__name__
        msg_id: str = msg['id'] if is_dict else msg.id
        content: str = msg['content'] if is_dict else str(msg.content)
        data_type: Optional[str] = message_types.get(type_name, 'ANONYMOUS')

        if hashed.get(msg_id, True):
          extracted = ResponseMessage(id=msg_id, content=content, type=data_type)
          yield extracted
          hashed[msg_id] = False

  async def aget_thread_state(self) -> AsyncIterator[ResponseMessage]:
    config: Dict[str, Any] = self._get_config()
    # state: the instance of langgraph.pregel.types.StateSnapshot class
    snapshot = await self.app.aget_state(config)
    # Create response
    for data in self._generate_response(snapshot.values):
      yield EventResponse(event='history', data=[data])

  async def ainvoke(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]) -> AsyncIterator[ResponseMessage]:
    try:
      self._validate(message)
      config: Dict[str, Any] = self._get_config()
      outputs: Any = await self.app.ainvoke(message, config)
      # Create response
      for data in self._generate_response(outputs):
        yield EventResponse(event='data', data=[data])
    except InputMessageValidationError as ex:
      yield EventResponse(event='error', data=Error(error=str(ex)))
    except Exception as ex:
      err_msg: str = str(ex)
      g_logger.error(f'ChatController[ainvoke]{err_msg}')
      yield EventResponse(event='error', data=Error(error=err_msg))

  async def astream(self, message: Union[Sequence[AnyMessage], Dict[str, Any]]) -> AsyncIterator[ResponseMessage]:
    try:
      self._validate(message)

      async for chunk in self._astream_state(message):
        if isinstance(chunk, str):
          # For example, the matched chunk is `run_id`.
          yield EventResponse(event='metadata', data={'run_id': chunk})
        else:
          outputs: Union[list[BaseMessage], Dict[str, Any]] = orjson.loads(
            self.dumps([message_chunk_to_message(target) for target in chunk])
          )
          # Create response
          for data in self._generate_response(outputs):
            yield EventResponse(event='stream', data=data)
    except InputMessageValidationError as ex:
      yield EventResponse(event='error', data=Error(error=str(ex)))
    except Exception as ex:
      err_msg: str = str(ex)
      g_logger.error(f'ChatController[astream]{err_msg}')
      yield EventResponse(event='error', data=Error(error=err_msg))

    yield EventResponse(event='end', data={})

  async def event_stream(self, contents) -> AsyncIterator[bytes]:
    if 'chat_history' == contents['type']:
      async for response in self.aget_thread_state():
        yield self._converter(response)
    else:
      message: Union[Sequence[AnyMessage], Dict[str, Any]] = contents['message']
      callback: Any = self.astream if self._is_stream() else self.ainvoke

      async for response in callback(message):
        yield self._converter(response)
