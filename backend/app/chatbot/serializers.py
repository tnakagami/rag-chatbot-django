import json
import os
from rest_framework import serializers
from rest_framework.relations import MANY_RELATION_KWARGS, ManyRelatedField
from drf_spectacular.utils import extend_schema_field, extend_schema_serializer, OpenApiExample
from django.utils.translation import gettext_lazy
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from . import models, tasks
from .models.agents import AgentType, ToolType

class PrimaryKeyRelatedFieldEx(serializers.PrimaryKeyRelatedField):
  def __init__(self, **kwargs):
    self.queryset_response = kwargs.pop('queryset_response', False)
    self.related_name = kwargs.pop('related_name', None)
    super().__init__(**kwargs)

  class _ManyRelatedFieldEx(ManyRelatedField):
    def to_internal_value(self, data):
      if isinstance(data, str) or not hasattr(data, '__iter__'):
        self.fail('not_a_list', input_type=type(data).__name__)

      if not self.allow_empty and len(data) == 0:
        self.fail('empty')

      return self.child_relation.to_internal_value(data)

  @classmethod
  def many_init(cls, *args, **kwargs):
    list_kwargs = {'child_relation': cls(*args, **kwargs)}

    for key in kwargs:
      if key in MANY_RELATION_KWARGS:
        list_kwargs[key] = kwargs[key]

    return cls._ManyRelatedFieldEx(**list_kwargs)

  def get_queryset(self):
    try:
      view = self.context.get('view', None)
      user = view.request.user
      related_instance = getattr(user, self.related_name)
      queryset = related_instance.all()
    except Exception:
      queryset = super().get_queryset()

    return queryset

  def to_internal_value(self, data):
    if isinstance(data, list):
      if self.pk_field is not None:
        data = self.pk_field.to_internal_value(data)
      results = self.get_queryset().filter(pk__in=data)
      # Check if all data exists
      pk_list = results.values_list('pk', flat=True)
      diff = list(set(data) - set(pk_list))

      if len(diff) > 0:
        pk_value = ', '.join(map(str, diff))
        self.fail('does_not_exist', pk_value=pk_value)

      if self.queryset_response:
        out = results
      else:
        out = list(results)
    else:
      out = super().to_internal_value(data)

    return out

@extend_schema_field({
  'type': 'integer',
  'example': {
    'id': 'integer',
    'name': 'string',
  }
})
class _CustomTypeIdField(serializers.IntegerField):
  def __init__(self, type_id_cls, **kwargs):
    self.type_id_cls = type_id_cls
    super().__init__(**kwargs)

  def to_representation(self, value):
    type_id = int(value)
    name = self.type_id_cls(type_id).label

    return {'id': type_id, 'name': name}

class _CommonConfigSerializer(serializers.ModelSerializer):
  def to_internal_value(self, data):
    if isinstance(data, str):
      data = json.loads(data)
    out = super().to_internal_value(data)

    return out

  def validate(self, attrs):
    config_field_name = 'config'
    type_field_name, callback = self.get_typename_and_callback()
    type_id, base_config = self.Meta.model.get_config_form_args(instance=self.instance)
    target_id = attrs.get(type_field_name, type_id)
    target_config = attrs.get(config_field_name, base_config)
    fields = callback(target_id)

    # ====================
    # Validate config data
    # ====================
    # In the case of not setting config data
    if len(target_config) == 0 and len(fields) > 0:
      raise serializers.ValidationError(gettext_lazy('Config data is empty.'), code='invalid')
    # In the case of setting config data
    else:
      valid_keys = [target.get_key() for target in fields]
      # In the case of invalid keys of config data
      if any([key not in valid_keys for key in target_config.keys()]):
        raise serializers.ValidationError(gettext_lazy('Invalid keys exist in config data.'))
      use_config = target_config

    # Copy the attributes
    output = {key: val for key, val in attrs.items()}
    fields = callback(target_id, use_config)
    # In the case of the fields exist
    if len(fields) > 0:
      config = base_config
    else:
      config = {}
    # Update config data
    for target in fields:
      config.update(target.asdict())
    output[config_field_name] = config

    return output

  def get_typename_and_callback(self):
    raise NotImplementedError

class _CommonTypeIdsSerializer(serializers.BaseSerializer):
  def to_internal_value(self, data):
    return data

  def to_representation(self, validated_data):
    return {validated_data['name']: [{'id': value, 'name': label} for value, label in validated_data['choices']]}

class _BaseConfigFormatSerializer(serializers.Serializer):
  type_id = serializers.ChoiceField(choices=[], initial=0)

  def to_representation(self, validated_data):
    type_id = validated_data['type_id']
    name, fields = self.get_config_format(type_id)
    config = {}

    for target in fields:
      config.update(target.asdict())

    return {'name': name, 'format': config}

  def get_config_format(self, type_id):
    raise NotImplementedError

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Azure agent',
      value={
        'name': 'azure-chatbot',
        'agent_type': AgentType.AZURE.value,
        'config': {
          'model': 'azure-chatbot-model',
          'max_retries': 3,
          'api_key': 'azure-api-key',
          'endpoint': 'http://azure-endpoint.com',
          'version': 'test-version',
          'deployment': 'http://test-deployment.com',
        },
      },
      request_only=True,
      response_only=False,
    ),
    OpenApiExample(
      name='OpenAI agent',
      value={
        'name': 'openai-chatbot',
        'config': {
          'model': 'openai-chatbot-model',
          'max_retries': 3,
        },
      },
      request_only=True,
      response_only=False,
    )
  ]
)
class AgentSerializer(_CommonConfigSerializer):
  agent_type = _CustomTypeIdField(type_id_cls=AgentType, required=False)

  class Meta:
    model = models.Agent
    fields = ('pk', 'name', 'config', 'agent_type')
    read_only_fields = ['pk']

  def get_typename_and_callback(self):
    type_name = 'agent_type'
    callback = lambda type_id, config=None: AgentType.get_llm_fields(type_id, config or {}, is_embedded=False)

    return type_name, callback

class AgentTypeIdsSerializer(_CommonTypeIdsSerializer):
  def __init__(self, *args, **kwargs):
    super().__init__(data={'name': 'types', 'choices': AgentType.choices})

class AgentConfigFormatSerializer(_BaseConfigFormatSerializer):
  type_id = serializers.ChoiceField(choices=AgentType.choices, initial=AgentType.OPENAI)

  def get_config_format(self, type_id):
    name = AgentType(type_id).label
    targets = AgentType.get_llm_fields(type_id, {}, is_embedded=False)

    return name, targets

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Azure embedding',
      value={
        'name': 'azure-embedding',
        'emb_type': AgentType.AZURE.value,
        'distance_strategy': models.Embedding.DistanceType.EUCLIDEAN,
        'config': {
          'model': 'azure-embedding-model',
          'max_retries': 3,
          'api_key': 'azure-embedding-api-key',
          'endpoint': 'http://azure-embedding-endpoint.com',
          'version': 'test-version',
          'deployment': 'http://test-embedding-deployment.com',
        },
      },
      request_only=True,
      response_only=False,
    ),
    OpenApiExample(
      name='OpenAI embedding',
      value={
        'name': 'openai-embedding',
        'config': {
          'model': 'openai-embedding-model',
          'max_retries': 3,
        },
      },
      request_only=True,
      response_only=False,
    )
  ]
)
class EmbeddingSerializer(_CommonConfigSerializer):
  distance_strategy = _CustomTypeIdField(type_id_cls=models.Embedding.DistanceType, required=False)
  emb_type = _CustomTypeIdField(type_id_cls=AgentType, required=False)

  class Meta:
    model = models.Embedding
    fields = ('pk', 'name', 'config', 'distance_strategy', 'emb_type')
    read_only_fields = ['pk']

  def get_typename_and_callback(self):
    type_name = 'emb_type'
    callback = lambda type_id, config=None: AgentType.get_llm_fields(type_id, config or {}, is_embedded=True)

    return type_name, callback

class EmbeddingTypeIdsSerializer(_CommonTypeIdsSerializer):
  def __init__(self, *args, **kwargs):
    super().__init__(data={'name': 'types', 'choices': AgentType.embedding_choices})

class DistanceTypeIdsSerializer(_CommonTypeIdsSerializer):
  def __init__(self, *args, **kwargs):
    super().__init__(data={'name': 'distances', 'choices': models.Embedding.DistanceType.choices})

class EmbeddingConfigFormatSerializer(_BaseConfigFormatSerializer):
  type_id = serializers.ChoiceField(choices=AgentType.embedding_choices, initial=AgentType.OPENAI)

  def get_config_format(self, type_id):
    name = AgentType(type_id).label
    targets = AgentType.get_llm_fields(type_id, {}, is_embedded=True)

    return name, targets

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Action server',
      value={
        'name': 'action-server-tool',
        'tool_type': ToolType.ACTION_SERVER.value,
        'config': {
          'api_key': 'action-server-api-key',
          'url': 'http://dummy-action-server.com',
        },
      },
      request_only=True,
      response_only=False,
    ),
    OpenApiExample(
      name='Retriever',
      value={
        'name': 'Retriever-tool',
        'config': {
          'k': 5,
        },
      },
      request_only=True,
      response_only=False,
    ),
    OpenApiExample(
      name='Arxiv',
      value={
        'name': 'arxiv-tool',
        'tool_type': ToolType.ARXIV.value,
      },
      request_only=True,
      response_only=False,
    )
  ]
)
class ToolSerializer(_CommonConfigSerializer):
  config = serializers.JSONField(required=False)
  tool_type = _CustomTypeIdField(type_id_cls=ToolType, required=False)

  class Meta:
    model = models.Tool
    fields = ('pk', 'name', 'config', 'tool_type')
    read_only_fields = ['pk']

  def get_typename_and_callback(self):
    type_name = 'tool_type'
    callback = lambda type_id, config=None: ToolType.get_config_field(type_id, config or {})

    return type_name, callback

class ToolTypeIdsSerializer(_CommonTypeIdsSerializer):
  def __init__(self, *args, **kwargs):
    super().__init__(data={'name': 'types', 'choices': ToolType.choices})

class ToolConfigFormatSerializer(_BaseConfigFormatSerializer):
  type_id = serializers.ChoiceField(choices=ToolType.choices, initial=ToolType.RETRIEVER)

  def get_config_format(self, type_id):
    name = ToolType(type_id).label
    targets = ToolType.get_config_field(type_id, {})

    return name, targets

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Pseudo assistant',
      value={
        'name': 'pseudo-assistant',
        'system_message': 'You are a system engineer.',
        'agent_pk': 1,
        'embedding_pk': 2,
        'tool_pks': [
          4,
          7,
        ],
        'is_interrupt': True,
      },
      request_only=True,
      response_only=False,
    ),
    OpenApiExample(
      name='Sample assistant using default value',
      value={
        'name': 'pseudo-assistant',
        'agent_pk': 2,
        'embedding_pk': 3,
        'tool_pks': [
          9,
        ],
      },
      request_only=True,
      response_only=False,
    ),
  ]
)
class AssistantSerializer(serializers.ModelSerializer):
  agent = AgentSerializer(read_only=True)
  embedding = EmbeddingSerializer(read_only=True)
  tools = ToolSerializer(many=True, read_only=True)
  agent_pk = PrimaryKeyRelatedFieldEx(queryset=models.Agent.objects.none(), related_name='agent_configs', write_only=True)
  embedding_pk = PrimaryKeyRelatedFieldEx(queryset=models.Embedding.objects.none(), related_name='embedding_configs', write_only=True)
  tool_pks = PrimaryKeyRelatedFieldEx(queryset=models.Tool.objects.none(), related_name='tool_configs', many=True, write_only=True)

  class Meta:
    model = models.Assistant
    fields = ('pk', 'name', 'system_message', 'agent', 'agent_pk', 'embedding', 'embedding_pk', 'tools', 'tool_pks', 'is_interrupt')
    read_only_fields = ['pk']

  def to_internal_value(self, data):
    if isinstance(data, str):
      data = json.loads(data)
    data = super().to_internal_value(data)
    keys = data.keys()
    patterns = [
      ('agent', 'agent_pk'),
      ('embedding', 'embedding_pk'),
      ('tools', 'tool_pks'),
    ]

    for main_field, related_field in patterns:
      if related_field in keys:
        data[main_field] = data.pop(related_field)

    return data

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Collect own tasks',
      request_only=True,
      response_only=False,
    )
  ]
)
class AssistantTaskSerializer(serializers.BaseSerializer):
  def __init__(self, user, *args, **kwargs):
    super().__init__(data={'tasks': models.Assistant.objects.collect_own_tasks(user=user)})

  def to_internal_value(self, data):
    return data

  def to_representation(self, validated_data):
    queryset = validated_data.get('tasks')
    
    tasks = [
      {
        'task_name': record.task_name,
        'status': record.status,
        'date_created': models.convert_timezone(record.date_created, is_string=True),
      }
      for record in queryset
    ]

    return {'tasks': tasks}

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Pseudo document file',
      value={
        'assistant_pk': 1,
        'upload_files': [
          'sample1.txt',
          'sample2.pdf',
          'sample3.html',
          'sample4.docx',
        ],
      },
      request_only=True,
      response_only=False,
    )
  ]
)
class DocumentFileSerializer(serializers.ModelSerializer):
  assistant = AssistantSerializer(read_only=True)
  assistant_pk = PrimaryKeyRelatedFieldEx(queryset=models.Assistant.objects.none(), related_name='assistants', write_only=True)
  upload_files = serializers.ListField(
    child=serializers.FileField(
      max_length=models.DocumentFile.MAX_FILENAME_LENGTH,
      allow_empty_file=False,
      use_url=False
    ),
    required=False,
  )

  class Meta:
    model = models.DocumentFile
    fields = ('pk', 'assistant', 'assistant_pk', 'upload_files')
    read_only_fields = ['pk']
    MAX_TOTAL_FILESIZE = 10 * 1024 * 1024

  def _convert_human_readable_filesize(self, size, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi'):
      if abs(size) < 1024.0:
        out = f'{size:3.1f}{unit}{suffix}'
        break
      size /= 1024.0
    else:
      out = f'{size:.1f}Ti{suffix}'

    return out

  def validate_upload_files(self, upload_files):
    errors = {}
    total_size = 0

    for target_field in upload_files:
      total_size += target_field.size
      name, extension = os.path.splitext(target_field.name)
      ext = extension.lower()
      allowed_extensions = models.DocumentFile.get_valid_extensions()

      if ext not in allowed_extensions:
        allowed_exts = ', '.join(allowed_extensions)
        errors[name] = gettext_lazy(f'Invalid extension ({ext}). Allowed: {allowed_exts}')

    if total_size > self.Meta.MAX_TOTAL_FILESIZE:
      max_size = self._convert_human_readable_filesize(self.Meta.MAX_TOTAL_FILESIZE)
      current_size = self._convert_human_readable_filesize(total_size)
      errors['file_size'] = gettext_lazy(f'Max file size is {max_size}. Current size: {current_size}')

    if len(errors) > 0:
      raise serializers.ValidationError(errors)

    return upload_files

  def to_internal_value(self, data):
    if isinstance(data, str):
      data = json.loads(data)
    data = super().to_internal_value(data)
    main_field, related_field = 'assistant', 'assistant_pk'

    if related_field in data.keys():
      data[main_field] = data.pop(related_field)

    return data

  def save(self, **kwargs):
    validated_data = {**self.validated_data, **kwargs}
    assistant = validated_data['assistant']
    filefields = validated_data.get('upload_files', [])
    user = validated_data.get('user')
    # Preparation
    storage = FileSystemStorage()
    files = []

    for target in filefields:
      saved_name = storage.save(target.name, File(target))
      files += [{'path': storage.path(saved_name), 'name': target.name, 'saved_name': saved_name}]

    # Execute embedding process
    tasks.embedding_process.delay(assistant.pk, files, user.pk)

@extend_schema_serializer(
  examples=[
    OpenApiExample(
      name='Pseudo thread',
      value={
        'name': 'pseudo-thread',
        'assistant_pk': 1,
        'docfile_pks': [
          3,
          5,
          6,
        ],
      },
      request_only=True,
      response_only=False,
    )
  ]
)
class ThreadSerializer(serializers.ModelSerializer):
  assistant = AssistantSerializer(read_only=True)
  docfiles = DocumentFileSerializer(many=True, read_only=True)
  assistant_pk = PrimaryKeyRelatedFieldEx(queryset=models.Assistant.objects.none(), related_name='assistants', write_only=True)
  docfile_pks = serializers.PrimaryKeyRelatedField(queryset=models.DocumentFile.objects.all(), many=True, write_only=True)

  class Meta:
    model = models.Thread
    fields = ('pk', 'assistant', 'assistant_pk', 'name', 'docfiles', 'docfile_pks')
    read_only_fields = ['pk']
    _patterns = [
      ('assistant', 'assistant_pk', '_validate_assistant'),
      ('docfiles', 'docfile_pks', '_validate_docfiles'),
    ]

  def to_internal_value(self, data):
    if isinstance(data, str):
      data = json.loads(data)
    data = super().to_internal_value(data)
    keys = data.keys()

    for main_field, related_field, _ in self.Meta._patterns:
      if related_field in keys:
        data[main_field] = data.pop(related_field)

    return data

  def _validate_assistant(self, assistant):
    err = {}

    # In the case of updating the thread
    if self.instance is not None and self.instance.assistant.pk != assistant.pk:
      err['assistant_pk'] = gettext_lazy("Invalid primary key of assistant because it's prohibited to set different primary key of assistant.")

    return err

  def _validate_docfiles(self, docfiles):
    err = {}

    try:
      view = self.context.get('view', None)
      user = view.request.user
      is_valid = all([instance.is_owner(user) for instance in docfiles])

      if not is_valid:
        raise Exception()
    except Exception:
      err['docfile_pks'] = gettext_lazy('Invalid docfiles exist. Please check if the owner of these docfiles are yours.')

    return err

  def validate(self, attrs):
    keys = attrs.keys()
    errors = {}
    # Check each field
    for main_field, _, method_name in self.Meta._patterns:
      if main_field in keys:
        callback = getattr(self, method_name)
        errors.update(callback(attrs[main_field]))
    # Check error
    if len(errors) > 0:
      raise serializers.ValidationError(errors)

    return attrs