from typing import Any, List, cast, Sequence
from langchain.tools import BaseTool
from langchain.tools.render import render_text_description
from langchain_core.language_models.base import LanguageModelLike
from langchain_core.messages import (
  AnyMessage,
  AIMessage,
  FunctionMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  MessageLikeRepresentation,
)
from langgraph.checkpoint import CheckpointAt
from langgraph.graph import END
from langgraph.graph.message import MessageGraph, Messages, add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation

class _LiberalFunctionMessage(FunctionMessage):
  content: Any

class _LiberalToolMessage(ToolMessage):
  content: Any

class _BaseExecutor:
  def __init__(self, tools: List[BaseTool], is_interrupt: bool, checkpoint: Any):
    self.llm = None
    self.tool_executor = ToolExecutor(tools)
    self.is_interrupt = is_interrupt
    self.template = '{system_message}'
    self.checkpoint = checkpoint

  def get_messages(self, messages, system_message):
    raise NotImplemented

  def should_continue(self, messages):
    raise NotImplemented

  async def call_tool(self, messages):
    raise NotImplemented

  def get_app(self, system_message: str):
    sys_msg = self.template.format(system_message)
    _get_messages = lambda messages: self.get_messages(messages, system_message=sys_msg)
    # Create agent
    agent = _get_messages | self.llm
    # Create workflow
    workflow = MessageGraph()
    # Define the two nodes we will cycle between
    workflow.add_node('agent', agent)
    workflow.add_node('action', self.call_tool)
    # Set the entrypoint as `agent`
    workflow.set_entry_point('agent')
    # Add a conditional edge
    workflow.add_conditional_edges(
      'agent',
      self.should_continue,
      {
        'continue': 'action',
        'end': END,
      }
    )
    # Add a normal edge from `tools` to `agent`
    workflow.add_edge('action', 'agent')
    app = workflow.compile(
      checkpoint=self.checkpoint,
      interrupt_before=['action'] if self.is_interrupt else None,
    ).with_types(
      input_type=Messages,
      output_type=Sequence[AnyMessage],
    )

    return app

class ToolExecutor(_BaseExecutor):
  def __init__(self, llm: LanguageModelLike, tools: List[BaseTool], is_interrupt: bool, checkpoint: Any):
    super().__init__(tools, is_interrupt, checkpoint)
    self.llm = llm.bind_tools(tools) if tools else llm
    self.template = '{system_message}'

  def get_messages(self, messages, system_message):
    msgs = [SystemMessage(content=system_message)]

    for _msg in messages:
      if isinstance(_msg, _LiberalToolMessage):
        _dict_msg = _msg.dict()
        _dict_msg['content'] = str(_dict_msg['content'])
        converted = ToolMessage(**_dict_msg)
      elif isinstance(_msg, _LiberalFunctionMessage):
        converted = HumanMessage(content=str(_msg.content))
      else:
        converted = _msg
      msgs.append(converted)

    return msgs

  def should_continue(self, messages):
    last_message = messages[-1]
    exec_type = 'continue' if last_message.tool_calls else 'end'

    return exec_type

  async def call_tool(self, messages):
    last_message = cast(AIMessage, messages[-1])
    actions = [
      ToolInvocation(tool=tool_call['name'], tool_input=tool_call['args']) 
      for tool_call in last_message.tool_calls
    ]
    responses = await self.tool_executor.abatch(actions)
    tool_messages = [
      _LiberalToolMessage(tool_call_id=tool_call['id'], name=tool_call['name'], content=response)
      for tool_call, response in zip(last_message.tool_calls, responses)
    ]

    return tool_messages

class XmlExecutor(_BaseExecutor):
  def __init__(self, llm: LanguageModelLike, tools: List[BaseTool], is_interrupt: bool, checkpoint: Any):
    super().__init__(tools, is_interrupt, checkpoint)
    self.llm = llm.bind(stop=['</tool_input>', '<observation>'])
    self.template = '\n'.join([
      '{system_message}', ''
      'You have access to the following tools:', '',
      render_text_description(tools), '',
      'In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags.', '',
      'You will then get back a response in the form <observation></observation>', '',
      'Begin!',
    ])

  def _collapse_message(self, messages):
    if (messages[-1], AIMessage):
      scratchpad = messages[:-1]
      final = messages[-1]
    else:
      scratchpad = messages
      final = None

    if len(scratchpad) % 2 != 0:
      raise ValueError('Unexpected')
    log = ''.join([
      f'{action.content}<observation>{observation.content}</observation>'
      for action, observation in zip(*([iter(scratchpad)]*2))
    ])

    if final is not None:
      log += final.content

    return AIMessage(content=log)

  def _construct_chat_history(self, messages):
    collapsed = []
    tmp_msg = []

    for _msg in messages:
      if isinstance(_msg, HumanMessage):
        if tmp_msg:
          collapsed.append(self._collapse_message(tmp_msg))
          tmp_msg = []
        collapsed.append(_msg)
      elif isinstance(_msg, _LiberalFunctionMessage):
        _dict_msg = _msg.dict()
        _dict_msg['content'] = str(_dict_msg['content'])
        tmp_msg.append(FunctionMessage(**_dict_msg))
      else:
        tmp_msg.append(_msg)

    if tmp_msg:
      collapsed.append(self._collapse_message(tmp_msg))

    return collapsed

  def get_messages(self, messages, system_message):
    msgs = [SystemMessage(content=system_message)] + self._construct_chat_history(messages)

    return msgs

  def should_continue(self, messages):
    last_message = messages[-1]
    exec_type = 'continue' if '</tool>' in last_message.content else 'end'

    return exec_type

  async def call_tool(self, messages):
    last_message = messages[-1]
    tool, tool_input = last_message.content.split('</tool>')
    _tool = tool.split('</tool>')[1]

    if '<tool_input>' not in tool_input:
      _tool_input = ''
    else:
      _tool_input = tool_input.split('<tool_input>')[1]

      if '</tool_input>' in _tool_input:
        _tool_input = _tool_input.split('</tool_input>')[0]

    action = ToolInvocation(tool=_tool, tool_input=_tool_input)
    response = await tool_executor.ainvoke(action)
    function_message = _LiberalFunctionMessage(content=response, name=action.tool)

    return function_message