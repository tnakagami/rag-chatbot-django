import pytest
import orjson
from dataclasses import asdict
from chatbot import stream
from langchain.pydantic_v1 import ValidationError
from langchain_core.messages import (
  HumanMessage,
  AIMessage,
  FunctionMessage,
  ToolMessage,
)

class FakeLogger:
  def __init__(self):
    self.message = ''
  def error(self, msg):
    self.message = msg

class FakeValidator:
  def __init__(self, is_valid):
    self.is_valid = is_valid

  def validate(self, *args, **kwargs):
    if not self.is_valid:
      raise ValidationError(self, 'err')

class FakeSnapshot:
  def __init__(self, data):
    self.values = data
    self.next = []

class FakeApp:
  def __init__(self, *args, is_stream=False, is_valid=True, **kwargs):
    self._is_streaming = is_stream
    self.data = kwargs.get('data', [])
    self._validator = FakeValidator(is_valid)

  async def astream_events(self, inputs, config, *args, **kwargs):
    for value in self.data:
      yield value

  def get_input_schema(self):
    return self._validator

  async def aget_state(self, config):
    data = await self.ainvoke([], {})

    return FakeSnapshot(data)

  async def ainvoke(self, inputs, config):
    return self.data

def compare_any_message(target, exact):
  return all([
    str(target.content) == str(exact.content),
    target.id == exact.id
  ])

def compare_dict_message(target, exact):
  return all([target[key] == exact[key]] for key in ['content', 'id'])

@pytest.fixture
def get_same_run_id_message():
  run_id = '012'

  data = [
    {'run_id': run_id, 'event': 'on_chat_start'},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': [HumanMessage(content='same-message', id='human-001'), HumanMessage(content='same-message', id='human-001')],
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'messages': [{'id': 'human-011', 'content': 'next-human-message'}, {'id': 'human-011', 'content': 'next-other-message'}]},
    }},
  ]
  expected = {
    'run_id': run_id,
    '1st_msg': HumanMessage(content='same-message', id='human-001'),
    '2nd_msg': [{'id': 'human-011', 'content': 'next-human-message'}, {'id': 'human-011', 'content': 'next-other-message'}]
  }

  return data, expected

@pytest.fixture
def get_same_run_id_and_no_message():
  run_id = '023'

  data = [
    {'run_id': run_id, 'event': 'on_chat_start'},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': [],
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'messages': []},
    }},
  ]
  expected = {
    'run_id': run_id,
  }

  return data, expected

@pytest.fixture
def get_different_run_id_message():
  run_id = 'a01'

  data = [
    {'run_id': '102', 'event': 'on_chat_start'},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': HumanMessage(content='1st', id='human-101'),
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': HumanMessage(content='-message', id='human-101'),
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': HumanMessage(content='2nd', id='human-102'),
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'id': 'human-102', 'content': '-message'},
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'id': 'human-103', 'content': '3rd'},
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': HumanMessage(content='-message', id='human-103'),
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'id': 'human-104', 'content': '4th'},
    }},
    {'run_id': run_id, 'event': 'on_chat_stream', 'data': {
      'chunk': {'id': 'human-104', 'content': '-massage'},
    }},
  ]
  expected = {
    'run_id': '102',
    '1st_msg': HumanMessage(content='1st', id='human-101'),
    '2nd_msg': [HumanMessage(content='1st', id='human-101'), HumanMessage(content='-message', id='human-101')],
    '3rd_msg': HumanMessage(content='2nd', id='human-102'),
    '4th_msg': [HumanMessage(content='2nd', id='human-102'), HumanMessage(content='-message', id='human-102')],
    '5th_msg': {'id': 'human-103', 'content': '3rd'},
    '6th_msg': {'id': 'human-103', 'content': '3rd-message'},
    '7th_msg': {'id': 'human-104', 'content': '4rd'},
    '8th_msg': {'id': 'human-104', 'content': '4rd-message'},
  }

  return data, expected

@pytest.mark.chatbot
@pytest.mark.private
def test_check_dump_function():
  class FakeDict:
    def __init__(self):
      pass
    def dict(self):
      return 'dummy'
  class FakeInvalidObj:
    def __init__(self):
      self.dict = 0

  controller = stream.ChatbotController(app=FakeApp(), pk=0)
  func = controller.dumps.keywords['default']

  try:
    func(FakeDict())
  except TypeError:
    pytest.fail('Raise Exception when the function `controller.dumps` is called.')

  with pytest.raises(TypeError):
    func(FakeInvalidObj())
  with pytest.raises(TypeError):
    func(object())

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.parametrize('is_stream', [True, False], ids=['stream-is-true', 'stream-is-false'])
def test_check_is_stream_method(is_stream):
  controller = stream.ChatbotController(app=FakeApp(is_stream=is_stream), pk=0)

  assert controller._is_stream() == is_stream

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.parametrize('response', [
  stream.EventResponse(event='data', data=[stream.ResponseMessage(id='abc123', content='is-list', type='HUMAN')]),
  stream.EventResponse(event='data', data=stream.ResponseMessage(id='456xyz', content='is-message-instance', type='HUMAN')),
  stream.EventResponse(event='stream', data={'message': 'is-stream', 'id': 'stu789'}),
  stream.EventResponse(event='stream', data=[{'message': 'is-stream-list1', 'id': 'stu789'}, {'message': 'is-stream-list2', 'id': 'stu789'}]),
  stream.EventResponse(event='data', data='run-id'),
  stream.EventResponse(event='err', data=stream.Error(error='error-instance')),
  stream.EventResponse(event='end', data={}),
], ids=[
  'is-list-data',
  'is-message-data',
  'is-stream-data',
  'is-stream-list',
  'is-string',
  'is-error-instance',
  'is-end-message',
])
def test_check_converter_method(response):
  expected = '{}\n\n'.format(asdict(response)).replace(' ', '').replace("'", '"').encode('utf-8')
  controller = stream.ChatbotController(app=FakeApp(), pk=0)
  out = controller._converter(response)

  assert out == expected

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_run_id_of_astream_state_method():
  data = [{'event': 'on_chat_start', 'run_id': '001'}]
  expected = '001'
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller._astream_state([])
  run_id = await anext(generator)

  assert run_id == expected

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_same_run_id_of_astream_state_method(get_same_run_id_message):
  data, expected = get_same_run_id_message
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller._astream_state([])
  run_id = await anext(generator)
  response_1st = await anext(generator)
  response_2nd = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert len(response_1st) == 1
  assert len(response_2nd) == 2
  assert run_id == expected['run_id']
  assert compare_any_message(response_1st[0], expected['1st_msg'])
  assert all([compare_dict_message(target, exact) for target, exact in zip(response_2nd, expected['2nd_msg'])])

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_no_message_of_astream_state_method(get_same_run_id_and_no_message):
  data, expected = get_same_run_id_and_no_message
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller._astream_state([])
  run_id = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert run_id == expected['run_id']

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_different_message_of_astream_state_method(get_different_run_id_message):
  data, expected = get_different_run_id_message
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller._astream_state([])
  run_id = await anext(generator)
  response_1st = await anext(generator)
  response_2nd = await anext(generator)
  response_3rd = await anext(generator)
  response_4th = await anext(generator)
  response_5th = await anext(generator)
  response_6th = await anext(generator)
  response_7th = await anext(generator)
  response_8th = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert run_id == expected['run_id']
  assert compare_any_message(response_1st[0], expected['1st_msg'])
  assert all([compare_any_message(target, exact) for target, exact in zip(response_2nd, expected['2nd_msg'])])
  assert compare_any_message(response_3rd[0], expected['3rd_msg'])
  assert all([compare_any_message(target, exact) for target, exact in zip(response_4th, expected['4th_msg'])])
  assert compare_dict_message(response_5th[0], expected['5th_msg'])
  assert compare_dict_message(response_6th[0], expected['6th_msg'])
  assert compare_dict_message(response_7th[0], expected['7th_msg'])
  assert compare_dict_message(response_8th[0], expected['8th_msg'])

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_not_match_event_of_astream_state_method():
  expected_id = 'ab1'
  data = [
    {'run_id': 'ab1', 'event': 'on_chat_start'},
    {'run_id': 'xyz', 'event': 'on_chat_end', 'data': []},
  ]
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller._astream_state([])
  run_id = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert run_id == expected_id

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.parametrize('is_valid', [False, True], ids=['is-valid', 'is-invalid'])
def test_check_validate_method(mocker, is_valid):
  controller = stream.ChatbotController(app=FakeApp(is_valid=is_valid), pk=0)
  _faker = FakeLogger()
  mocker.patch.object(stream.g_logger, 'error', side_effect=lambda msg: _faker.error(msg))
  mocker.patch('pydantic.v1.error_wrappers.ValidationError.__str__', return_value='dummy-err')

  if is_valid:
    try:
      _ = controller._validate([])
    except:
      pytest.fail('Raise Exception when the instance method `_validate` is called.')
  else:
    with pytest.raises(stream.InputMessageValidationError) as ex:
      _ = controller._validate([])

  assert _faker.message == '' if is_valid else 'ChatController[validate]' in _faker.message

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.parametrize('thread_id', [3, 5, 6], ids=lambda val: f'thread-{val}')
def test_check_get_config_method(thread_id):
  controller = stream.ChatbotController(app=FakeApp(), pk=thread_id)
  config = controller._get_config()
  keys = config.keys()

  assert 'configurable' in keys
  assert 'thread_id' in config['configurable'].keys()
  assert config['configurable']['thread_id'] == thread_id

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.parametrize('messages,expected', [
  ([
    HumanMessage(content='human-message', id='human-001'),
    AIMessage(example=False, content='ai-message', id='ai-001', tool_calls=[]),
    FunctionMessage(id='func-001', content='function-message', name='dummy-function'),
    ToolMessage(tool_call_id='tool-call-001', content='tool-message', id='tool-001'),
  ], [
    stream.ResponseMessage(id='human-001', content='human-message', type='HUMAN'),
    stream.ResponseMessage(id='ai-001', content='ai-message', type='AI'),
    stream.ResponseMessage(id='func-001', content='function-message', type='FUNCTION'),
    stream.ResponseMessage(id='tool-001', content='tool-message', type='TOOL'),
  ]),
  ([
    HumanMessage(content='duplicated-message', id='human-011'),
    AIMessage(example=False, content='ai-message', id='ai-011', tool_calls=[]),
    HumanMessage(content='duplicated-message', id='human-011'),
  ], [
    stream.ResponseMessage(id='human-011', content='duplicated-message', type='HUMAN'),
    stream.ResponseMessage(id='ai-011', content='ai-message', type='AI'),
  ])
], ids=['general-pattern', 'duplicated-pattern'])
def test_check_generate_response_method(messages, expected):
  controller = stream.ChatbotController(app=FakeApp(), pk=0)
  generator = controller._generate_response(messages)
  outputs = [ret for ret in generator]

  assert len(outputs) == len(expected)
  assert all([all(
    [target.type == exact.type, target.content == exact.content, target.id == exact.id]
  )] for target, exact in zip(outputs, expected))

@pytest.mark.chatbot
@pytest.mark.private
def test_check_dict_pattern_of_generate_response_method():
  messages = [{'id': 'human-001', 'content': 'human-message'}]
  controller = stream.ChatbotController(app=FakeApp(), pk=0)
  generator = controller._generate_response(messages)
  outputs = [ret for ret in generator]

  assert len(outputs) == 1
  assert outputs[0].id == 'human-001'
  assert outputs[0].content == 'human-message'
  assert outputs[0].type == 'ANONYMOUS'

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_no_message_of_aget_thread_state_method():
  data = []
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller.aget_thread_state()
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_single_message_of_aget_thread_state_method():
  data = [HumanMessage(id='human-001', content='human-message')]
  expected_data = stream.ResponseMessage(id='human-001', content='human-message', type='HUMAN')
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller.aget_thread_state()
  output = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert output.event == 'history'
  assert len(output.data) == 1
  assert output.data[0].id == expected_data.id
  assert output.data[0].content == expected_data.content
  assert output.data[0].type == expected_data.type

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_multi_messages_of_aget_thread_state_method():
  data = [
    HumanMessage(id='human-001', content='human-message1'),
    HumanMessage(id='human-002', content='human-message2'),
    HumanMessage(id='human-003', content='human-message3'),
  ]
  expected_data = [
    stream.ResponseMessage(id='human-001', content='human-message1', type='HUMAN'),
    stream.ResponseMessage(id='human-002', content='human-message2', type='HUMAN'),
    stream.ResponseMessage(id='human-003', content='human-message3', type='HUMAN'),
  ]
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller.aget_thread_state()
  outputs = [val async for val in generator]

  assert len(outputs) == len(expected_data)
  assert all([_out.event == 'history' for _out in outputs])
  assert all([len(_out.data) for _out in outputs])
  assert all([all([
    _out.data[0].id == exact.id,
    _out.data[0].content == exact.content,
    _out.data[0].type == exact.type,
  ])] for _out, exact in zip(outputs, expected_data))

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
async def test_check_ainvoke_method():
  data = [HumanMessage(id='human-001', content='human-message1')]
  expected = stream.ResponseMessage(id='human-001', content='human-message1', type='HUMAN')
  controller = stream.ChatbotController(app=FakeApp(data=data), pk=0)
  generator = controller.ainvoke([])
  output = await anext(generator)
  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)

  assert output.event == 'data'
  assert len(output.data) == 1
  assert output.data[0].id == expected.id
  assert output.data[0].content == expected.content
  assert output.data[0].type == expected.type

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
@pytest.mark.parametrize('is_valid,raise_exception,expected,error_log', [
  (False, False, stream.EventResponse(event='error', data=stream.Error(error='dummy-err')), ''),
  (True, True, stream.EventResponse(event='error', data=stream.Error(error='err-ainvoke')), 'ChatController[ainvoke]'),
], ids=['raise-validation-error', 'raise-exception'])
async def test_check_invalid_data_of_ainvoke_method(mocker, is_valid, raise_exception, expected, error_log):
  data = [HumanMessage(id='human-001', content='human-message')]
  controller = stream.ChatbotController(app=FakeApp(data=data, is_valid=is_valid), pk=0)
  _faker = FakeLogger()
  mocker.patch.object(stream.g_logger, 'error', side_effect=lambda msg: _faker.error(msg))
  mocker.patch('pydantic.v1.error_wrappers.ValidationError.__str__', return_value='dummy-err')

  if raise_exception:
    mocker.patch.object(controller.app, 'ainvoke', side_effect=Exception('err-ainvoke'))
  generator = controller.ainvoke([])
  output = await anext(generator)

  assert isinstance(output, stream.EventResponse)
  assert output.event == expected.event
  assert output.data.status_code == expected.data.status_code
  assert output.data.error == expected.data.error
  assert error_log in _faker.message

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
@pytest.mark.parametrize('messages,expected', [
  ([
    {'run_id': 'abc', 'event': 'on_chat_start'},
    {'run_id': 'abc', 'event': 'on_chat_stream', 'data': {
      'chunk': [HumanMessage(content='human-message', id='human-001'), HumanMessage(content='human-message', id='human-001')],
    }},
  ], [
    stream.ResponseMessage(id='human-001', content='human-message', type='HUMAN')
  ]),
  ([
    {'run_id': 'abc', 'event': 'on_chat_start'},
    {'run_id': 'abc', 'event': 'on_chat_stream', 'data': {
      'chunk': [HumanMessage(content='1st-message', id='human-001'), HumanMessage(content='2nd-message', id='human-002')],
    }},
  ], [
    stream.ResponseMessage(id='human-001', content='1st-message', type='HUMAN'),
    stream.ResponseMessage(id='human-002', content='2nd-message', type='HUMAN'),
  ]),
  ([
    {'run_id': 'abc', 'event': 'on_chat_start'},
    {'run_id': 'abc', 'event': 'on_chat_stream', 'data': {
      'chunk': {'messages': [{'id': 'human-011', 'content': '1st-message'}, {'id': 'human-012', 'content': '2nd-message'}]},
    }},
  ], [
    stream.ResponseMessage(id='human-012', content='1st-message', type='ANONYMOUS'),
    stream.ResponseMessage(id='human-012', content='2nd-message', type='ANONYMOUS'),
  ]),
], ids=['single-human-message', 'multi-human-message', 'dict-data'])
async def test_check_astream_method(messages, expected):
  exact_run_id = 'abc'
  controller = stream.ChatbotController(app=FakeApp(data=messages), pk=0)
  generator = controller.astream([])
  runs = await anext(generator)
  outputs = [val async for val in generator]
  ends = outputs.pop()

  assert runs.event == 'metadata'
  assert 'run_id' in runs.data.keys()
  assert runs.data['run_id'] == exact_run_id
  assert len(outputs) == len(expected)
  assert all([
    all([
      target.event == 'stream',
      target.data.id == exact.id,
      target.data.content == exact.content,
      target.data.type == exact.type,
    ])
  ] for target, exact in zip(outputs, expected))
  assert ends.event == 'end'
  assert len(ends.data) == 0

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
@pytest.mark.parametrize('is_valid,raise_exception,expected,error_log', [
  (False, False, stream.EventResponse(event='error', data=stream.Error(error='dummy-err')), ''),
  (True, True, stream.EventResponse(event='error', data=stream.Error(error='err-astream')), 'ChatController[astream]'),
], ids=['raise-validation-error', 'raise-exception'])
async def test_check_invalid_data_of_astream_method(mocker, is_valid, raise_exception, expected, error_log):
  data = [HumanMessage(id='human-001', content='human-message')]
  controller = stream.ChatbotController(app=FakeApp(data=data, is_valid=is_valid), pk=0)
  _faker = FakeLogger()
  mocker.patch.object(stream.g_logger, 'error', side_effect=lambda msg: _faker.error(msg))
  mocker.patch('pydantic.v1.error_wrappers.ValidationError.__str__', return_value='dummy-err')

  if raise_exception:
    mocker.patch.object(controller, '_astream_state', side_effect=Exception('err-astream'))
  generator = controller.astream([])
  output = await anext(generator)

  assert isinstance(output, stream.EventResponse)
  assert output.event == expected.event
  assert output.data.status_code == expected.data.status_code
  assert output.data.error == expected.data.error
  assert error_log in _faker.message

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
@pytest.mark.parametrize('content', [
  {'type': 'chat_history', 'message': []},
  {'type': 'chat_message', 'message': []},
], ids=['chat-history', 'chat-message'])
async def test_check_event_stream_method(mocker, content):
  base_data = stream.EventResponse(event='dummy', data='sample')
  expected = orjson.dumps(base_data) + b'\n\n'

  async def fake_callback():
    yield base_data

  controller = stream.ChatbotController(app=FakeApp(), pk=0)
  mocker.patch.object(controller, 'aget_thread_state', return_value=fake_callback())
  mocker.patch.object(controller, 'astream', return_value=fake_callback())
  mocker.patch.object(controller, 'ainvoke', return_value=fake_callback())
  generator = controller.event_stream(content)
  output = await anext(generator)

  assert output == expected

@pytest.mark.chatbot
@pytest.mark.private
@pytest.mark.asyncio
@pytest.mark.parametrize('content', [
  {'type': 'chat_history', 'message': []},
  {'type': 'chat_message', 'message': []},
], ids=['no-message-of-chat-history', 'no-message-of-chat-message'])
async def test_check_no_message_of_event_stream_method(content):
  controller = stream.ChatbotController(app=FakeApp(data=[]), pk=0)
  generator = controller.event_stream(content)

  with pytest.raises(StopAsyncIteration):
    _ = await anext(generator)
