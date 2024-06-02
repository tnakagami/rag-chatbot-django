import pytest
from chatbot.models.utils import executors
# Related libraries for test execution
from asgiref.sync import async_to_sync
from chatbot.models.utils import ArxivTool, WikipediaTool, FireworksLLM, OllamaLLM
from langgraph.checkpoint import BaseCheckpointSaver, CheckpointAt
from langgraph.prebuilt import ToolExecutor as LangGraphToolExecutor
from langchain.schema.runnable import Runnable
from langchain.schema.messages import (
  AIMessage,
  FunctionMessage,
  HumanMessage,
  ToolMessage,
)

class TemporaryCheckpointSaver(BaseCheckpointSaver):
  def __init__(self):
    import pickle
    super().__init__(serde=pickle, at=CheckpointAt.END_OF_STEP)

@pytest.fixture(params=['no_tools', 'several_tools'])
def get_base_executor_arg(request):
  if request.param == 'no_tools':
    tools = []
  elif request.param == 'several_tools':
    tools = [
      ArxivTool().get_tools(),
      WikipediaTool().get_tools(),
    ]

  kwargs = {
    'tools': tools,
    'is_interrupt': True,
    'checkpoint': TemporaryCheckpointSaver(),
  }

  return kwargs

@pytest.fixture(params=[FireworksLLM(api_key='api-key'), OllamaLLM()], ids=['fireworks', 'ollama'])
def get_tool_or_xml_executor_arg(get_base_executor_arg, request):
  specific = {
    'llm': request.param.get_llm(is_embedded=False),
  }
  kwargs = get_base_executor_arg
  kwargs.update(specific)
  system_message = 'test-message'

  return kwargs, system_message

@pytest.fixture
def base_executor_checker():
  def inner(kwargs, executor):
    return all([
      executor.is_interrupt == kwargs['is_interrupt'],
      isinstance(executor.tool_executor, LangGraphToolExecutor),
      isinstance(executor.checkpoint, type(kwargs['checkpoint']))
    ])

  return inner

@pytest.fixture
def tool_or_xml_executor_checker(base_executor_checker):
  base_checker = base_executor_checker

  def inner(kwargs, executor):
    _llm = kwargs['llm']
    _tools = kwargs['tools']
    compare_class = type(_llm)

    if _tools:
      for attr_name, params in [('bind_tools', {'tools': _tools}), ('bind', {'stop': []})]:
        _func = getattr(_llm, attr_name, None)

        if callable(_func):
          try:
            _func(**params)
            compare_class = Runnable
            break
          except:
            pass

    return all([
      base_checker(kwargs, executor),
      isinstance(executor.llm, compare_class),
    ])

  return inner

@pytest.fixture
def get_tool_executor_messages(mocker):
  targets = [
    executors._LiberalToolMessage(
      tool_call_id=1,
      name='Tool message1',
      content='Tool data',
    ),
    FunctionMessage(
      name='Function message1',
      content='Function data',
    ),
    AIMessage(
      example=False,
      name='AI message1',
      tool_calls=[ToolMessage(id=2, tool_call_id=2, name='AI tool call', args={}, content='tool (id=2)')],
      additional_kwargs={'search': 'weather in SF'},
      content='AI data',
    )
  ]

  expected = [
    ToolMessage(
      tool_call_id=1,
      name='Tool message1',
      content='Tool data',
    ),
    HumanMessage(
      content='Function data',
    ),
    AIMessage(
      example=False,
      name='AI message1',
      tool_calls=[ToolMessage(id=2, tool_call_id=2, name='AI tool call', args={}, content='tool (id=2)')],
      additional_kwargs={'search': 'weather in SF'},
      content='AI data',
    )
  ]

  def inner(target_messages, expected_messages):
    target_ai_message = target_messages[-1]
    expected_ai_message = expected_messages[-1]
    target_ai_tools = target_ai_message.tool_calls[0]
    expected_ai_tools = expected_ai_message.tool_calls[0]

    return all([
      all([
        isinstance(_target, type(_expected)),
        str(_target.content) == str(_expected.content),
        _target.additional_kwargs == _expected.additional_kwargs,
        _target.response_metadata == _expected.response_metadata,
      ])
      for _target, _expected in zip(target_messages, expected_messages)
    ]) and all([
      target_messages[0].tool_call_id == expected_messages[0].tool_call_id,
      str(target_messages[0].name) == str(expected_messages[0].name),
      target_ai_message.example == expected_ai_message.example,
      str(target_ai_message.name) == str(expected_ai_message.name),
      isinstance(target_ai_tools, type(expected_ai_tools)),
      target_ai_tools['id'] == expected_ai_tools['id'],
      target_ai_tools['tool_call_id'] == expected_ai_tools['tool_call_id'],
      str(target_ai_tools['name']) == str(expected_ai_tools['name']),
      str(target_ai_tools['content']) == str(expected_ai_tools['content']),
    ])

  return targets, expected, inner

@pytest.fixture(params=[FireworksLLM(api_key='api-key'), OllamaLLM()], ids=['fireworks', 'ollama'])
def get_xml_executor_args(request):
  kwargs = {
    'llm': request.param.get_llm(is_embedded=False),
    'tools': [
      ArxivTool().get_tools(),
      WikipediaTool().get_tools(),
    ],
    'is_interrupt': True,
    'checkpoint': TemporaryCheckpointSaver(),
  }

  return kwargs

@pytest.mark.chatbot
@pytest.mark.util
def test_check_base_executor(get_base_executor_arg, base_executor_checker):
  kwargs = get_base_executor_arg
  checker = base_executor_checker
  instance = executors._BaseExecutor(**kwargs)
  messages = ['sample', 'messages']
  system_message = 'dummy-system-messages'

  with pytest.raises(NotImplementedError):
    _ = instance.get_messages(messages, system_message)
  with pytest.raises(NotImplementedError):
    _ = instance.should_continue(messages)
  with pytest.raises(NotImplementedError):
    _ = async_to_sync(instance.call_tool)(messages)
  with pytest.raises(TypeError):
    _ = instance.get_app(system_message)

  assert instance.llm is None
  assert checker(kwargs, instance)

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.parametrize('executor_class,messages', [
  (executors.ToolExecutor, [AIMessage(content='Hi')]),
  (executors.XmlExecutor, [AIMessage(content='<tool>search</tool><tool_input>weather in SF</tool_input>')]),
], ids=['tool', 'xml'])
def test_check_executor_member_and_method(get_tool_or_xml_executor_arg, tool_or_xml_executor_checker, executor_class, messages):
  kwargs, system_message = get_tool_or_xml_executor_arg
  checker = tool_or_xml_executor_checker
  instance = executor_class(**kwargs)

  try:
    _ = instance.get_messages(messages, system_message)
  except Exception as ex:
    pytest.fail('Raise Exception when the instance method `get_messages` is called.')
  try:
    _ = instance.should_continue(messages)
  except Exception as ex:
    pytest.fail('Raise Exception when the instance method `should_continue` is called.')
  try:
    _ = async_to_sync(instance.call_tool)(messages)
  except Exception as ex:
    pytest.fail('Raise Exception when the instance method `call_tool` is called.')
  try:
    app = instance.get_app(system_message)
  except Exception as ex:
    pytest.fail('Raise Exception when the instance method `get_app` is called.')

  assert checker(kwargs, instance)
  assert isinstance(app, Runnable)
  assert hasattr(app, '_is_streaming')

@pytest.mark.chatbot
@pytest.mark.util
def test_check_get_messages_method_of_tool_executor(get_tool_or_xml_executor_arg, get_tool_executor_messages):
  kwargs, system_message = get_tool_or_xml_executor_arg
  target_messages, expected_messages, checker = get_tool_executor_messages
  instance = executors.ToolExecutor(**kwargs)
  predicted_messages = instance.get_messages(target_messages, system_message)

  assert str(predicted_messages[0].content) == system_message
  assert checker(predicted_messages[1:], expected_messages)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_should_continue_method_of_tool_executor(get_tool_or_xml_executor_arg):
  kwargs, _ = get_tool_or_xml_executor_arg
  instance = executors.ToolExecutor(**kwargs)
  without_tools = [
    AIMessage(
      example=False,
      name='AI message1 without tool',
      tool_calls=[],
      content='AI data1',
    )
  ]
  with_tools = [
    AIMessage(
      example=False,
      name='AI message2 with tool',
      tool_calls=[ToolMessage(id=1, tool_call_id=1, name='AI tool call', args={}, content='with tool')],
      additional_kwargs={'search': 'weather in SF'},
      content='AI data2',
    )
  ]

  assert instance.should_continue(without_tools) == 'end'
  assert instance.should_continue(with_tools) == 'continue'

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.asyncio
async def test_check_call_tool_method_of_tool_executor(get_tool_or_xml_executor_arg, mocker):
  kwargs, _ = get_tool_or_xml_executor_arg
  instance = executors.ToolExecutor(**kwargs)

  async def callback_for_tool_calls(actions):
    return [str(action.tool) for action in actions]
  mocker.patch('chatbot.models.utils.executors.LangGraphToolExecutor.abatch', side_effect=callback_for_tool_calls)

  tool_calls = [
    ToolMessage(id=1, tool_call_id=1, name='AI tool call1', content='tool (id=1)', args={}),
    ToolMessage(id=2, tool_call_id=2, name='AI tool call2', content='tool (id=2)', args={'param': 2}),
  ]
  messages = [
    AIMessage(
      example=False,
      name='AI message with tools',
      tool_calls=tool_calls,
      content='AI data',
    )
  ]
  tool_messages = await instance.call_tool(messages)

  assert len(tool_messages) == len(tool_calls)
  assert all([
    str(tool_call.name) == str(tool_message.name) and str(tool_call.name) == str(tool_message.content)
    for tool_call, tool_message in zip(tool_calls, tool_messages)
  ])

@pytest.mark.chatbot
@pytest.mark.util
def test_check_collapse_message_method_of_xml_executor(get_xml_executor_args):
  # Target messages
  ai_message_only = [
    AIMessage(
      example=False,
      name='AI message1',
      tool_calls=[ToolMessage(id=1, tool_call_id=1, name='AI tool call1', args={'search': 'weather'}, content='tool (id=1)')],
      content='<tool>search</tool><tool_input>weather</tool_input>',
    )
  ]
  invalid_messages = ai_message_only + [
    AIMessage(
      example=False,
      name='AI message2',
      content='It is sunny.',
    )
  ]
  valid_messages = invalid_messages + [
    HumanMessage(
      content='You are an engineer.',
    ),
    HumanMessage(
      content='<input>Please tell me about your role.</input>',
    )
  ]
  # Expected messages
  expected_ai_message_only = AIMessage(
    content='<tool>search</tool><tool_input>weather</tool_input>',
  )
  expected_ai_message_for_valid_messages = AIMessage(
    content=''.join([
      '<tool>search</tool><tool_input>weather</tool_input>',
      '<observation>It is sunny.</observation>',
      'You are an engineer.',
      '<observation><input>Please tell me about your role.</input></observation>',
    ])
  )

  # Check
  kwargs = get_xml_executor_args
  instance = executors.XmlExecutor(**kwargs)
  predicted_ai_message_only = instance._collapse_message(ai_message_only)
  predicted_valid_messages = instance._collapse_message(valid_messages)
  with pytest.raises(ValueError) as ex:
    _ = instance._collapse_message(invalid_messages)

  assert str(predicted_ai_message_only.content) == str(expected_ai_message_only.content)
  assert str(predicted_valid_messages.content) == str(expected_ai_message_for_valid_messages.content)
  assert 'Unexpected' in str(ex.value)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_construct_chat_history_method_of_xml_executor(get_xml_executor_args):
  # Target messages
  only_human_message = [HumanMessage(content='You are an engineer.')]
  several_messages = [
    executors._LiberalFunctionMessage(
      name='Calculator',
      content='Calculate 3**2',
    ),
    executors._LiberalFunctionMessage(
      name='Notificator',
      content='The answer is 9.',
    ),
    HumanMessage(
      content='How is the weather tomorrow?',
    ),
    AIMessage(
      example=False,
      name='AI with tool',
      tool_calls=[ToolMessage(id=1, tool_call_id=1, name='AI tool', args={'search': 'weather', 'location': 'Japan'}, content='tool (id=1)')],
      content='<tool>search</tool><tool_input>weather</tool_input>',
    ),
    executors._LiberalFunctionMessage(
      name='Notificator',
      content='The answer is sunny.',
    ),
    AIMessage(
      example=False,
      name='AI answer',
      content='It is sunny.',
    )
  ]
  # Expected messages
  expected_several_messages = [
    AIMessage(
      content=''.join([
        'Calculate 3**2',
        '<observation>The answer is 9.</observation>',
      ])
    ),
    HumanMessage(
      content='How is the weather tomorrow?'
    ),
    AIMessage(
      content=''.join([
        '<tool>search</tool><tool_input>weather</tool_input>',
        '<observation>The answer is sunny.</observation>',
        'It is sunny.',
      ])
    )
  ]
  # Check
  kwargs = get_xml_executor_args
  instance = executors.XmlExecutor(**kwargs)
  predicted_only_human_message = instance._construct_chat_history(only_human_message)
  predicted_several_messages = instance._construct_chat_history(several_messages)

  assert len(predicted_only_human_message) == len(only_human_message)
  assert isinstance(predicted_only_human_message[0], type(only_human_message[0]))
  assert str(predicted_only_human_message[0].content) == str(only_human_message[0].content)
  assert len(expected_several_messages) == len(predicted_several_messages)
  assert isinstance(expected_several_messages[0], type(predicted_several_messages[0]))
  assert isinstance(expected_several_messages[1], type(predicted_several_messages[1]))
  assert isinstance(expected_several_messages[2], type(predicted_several_messages[2]))
  assert str(expected_several_messages[0].content) == str(predicted_several_messages[0].content)
  assert str(expected_several_messages[1].content) == str(predicted_several_messages[1].content)
  assert str(expected_several_messages[2].content) == str(predicted_several_messages[2].content)

@pytest.mark.chatbot
@pytest.mark.util
def test_check_should_continue_method_of_xml_executor(get_xml_executor_args):
  kwargs = get_xml_executor_args
  instance = executors.XmlExecutor(**kwargs)
  without_tools = [
    HumanMessage(
      content='Hi',
    )
  ]
  with_tools = [
    ToolMessage(
      tool_call_id=1,
      name='Tool message1',
      content='<tool>search</tool><tool_input>weather</tool_input>',
    )
  ]

  assert instance.should_continue(without_tools) == 'end'
  assert instance.should_continue(with_tools) == 'continue'

@pytest.mark.chatbot
@pytest.mark.util
@pytest.mark.asyncio
async def test_check_call_tool_method_of_xml_executor(get_xml_executor_args, mocker):
  kwargs = get_xml_executor_args
  instance = executors.XmlExecutor(**kwargs)

  async def callback_for_tool_calls(action):
    return f'{action.tool}-{action.tool_input}'
  mocker.patch('chatbot.models.utils.executors.LangGraphToolExecutor.ainvoke', side_effect=callback_for_tool_calls)

  tool_messages = [
    [ToolMessage(id=1, tool_call_id=1, name='Tool1', content='<tool>Notification</tool>')],
    [ToolMessage(id=2, tool_call_id=2, name='Tool2', content='<tool>Calculator</tool><tool_input>2**3</tool_input>')],
    [ToolMessage(id=3, tool_call_id=3, name='Tool3', content='<tool>Calculator</tool><tool_input>2**3')],
  ]
  expected_names = [
    'Notification',
    'Calculator',
    'Calculator',
  ]
  function_messages = [
    await instance.call_tool(tool_messages[0]),
    await instance.call_tool(tool_messages[1]),
    await instance.call_tool(tool_messages[2]),
  ]

  assert all([
    name == str(func.name) and str(func.content)
    for name, func in zip(expected_names, function_messages)
  ])