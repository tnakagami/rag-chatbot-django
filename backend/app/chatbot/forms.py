import os
from django.core.exceptions import ValidationError
from django import forms
from django.utils.translation import gettext_lazy
from . import models
from .models.agents import AgentType, ToolType

class _BaseModelForm(forms.ModelForm):
  template_name = 'renderer/custom_model_form.html'

  def __init__(self, user, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.user = user

    for field in self.fields.values():
      _classes = field.widget.attrs.get('class', '')
      field.widget.attrs['class'] = f'{_classes} form-control'
      field.widget.attrs['placeholder'] = field.help_text

  def save(self, *args, **kwargs):
    instance = super().save(commit=False)
    instance.user = self.user
    instance = self.customize(instance, *args, **kwargs)
    instance.save()

    return instance

  def customize(self, instance, *args, **kwargs):
    # Default process: do nothing
    return instance

class _BaseConfigForm(forms.Form):
  template_name = 'renderer/config_form.html'

  def __init__(self, *args, type_id=None, config=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.base_fields = {}
    self.fields = {}
    base_attrs = {'class': 'form-control configs'}

    if type_id is None:
      fields = []
    else:
      fields = self.get_fields(type_id, config=config or {})
    # Store fields status
    self._valid_fields = len(fields) > 0

    for target in fields:
      name = target.name
      label = target.label
      attrs = {'data-configname': name, 'placeholder': name}
      attrs.update(base_attrs)
      val = target.data

      if target.is_str:
        _field_class = forms.CharField
        widget = forms.TextInput(attrs=attrs)
      elif target.is_bool:
        choices = ((True, gettext_lazy('Active')), (False, gettext_lazy('Not active')))
        _field_class = lambda **kwargs: forms.TypedChoiceField(choices=choices, **kwargs)
        widget = forms.Select(attrs=attrs)
      elif target.is_int:
        _field_class = forms.IntegerField
        widget = forms.NumberInput(attrs=attrs)
      else:
        _field_class = forms.JSONField
        widget = forms.Textarea(attrs=attrs)
        val = {} if val is None else val

      self.base_fields[name] = _field_class(label=label, initial=val, widget=widget)
      self.fields[name] = _field_class(label=label, initial=val, widget=widget)

  @property
  def config_exists(self):
    return self._valid_fields

  def get_fields(self, type_id, config):
    # Default process: return empty list
    return []

# =========
# = Agent =
# =========
class AgentForm(_BaseModelForm):
  class Meta:
    model = models.Agent
    fields = ('name', 'agent_type', 'config')
    widgets = {
      'agent_type': forms.Select(attrs={'id': 'type-id', 'data-url': ''}),
      'config': forms.Textarea(attrs={'id': 'config-id', 'class': 'h-auto'}),
    }

  def __init__(self, config_form_url, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['agent_type'].widget.attrs['data-url'] = config_form_url

  def customize(self, instance, *args, **kwargs):
    agent_type, config = instance.get_config()
    fields = AgentType.get_llm_fields(agent_type, config=config, is_embedded=False)
    config = {}

    for target in fields:
      config.update(target.asdict())
    instance.config = config

    return instance

class AgentConfigForm(_BaseConfigForm):
  def get_fields(self, type_id, config):
    return AgentType.get_llm_fields(type_id, config, is_embedded=False)

# =============
# = Embedding =
# =============
class EmbeddingForm(_BaseModelForm):
  class Meta:
    model = models.Embedding
    fields = ('name', 'distance_strategy', 'emb_type', 'config')
    widgets = {
      'emb_type': forms.Select(attrs={'id': 'type-id', 'data-url': ''}),
      'config': forms.Textarea(attrs={'id': 'config-id', 'class': 'h-auto'}),
    }

  def __init__(self, config_form_url, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['emb_type'].widget.attrs['data-url'] = config_form_url

  def customize(self, instance, *args, **kwargs):
    emb_type, config = instance.get_config()
    fields = AgentType.get_llm_fields(emb_type, config=config, is_embedded=True)
    config = {}

    for target in fields:
      config.update(target.asdict())
    instance.config = config

    return instance

class EmbeddingConfigForm(_BaseConfigForm):
  def get_fields(self, type_id, config):
    return AgentType.get_llm_fields(type_id, config, is_embedded=True)

# ========
# = Tool =
# ========
class ToolForm(_BaseModelForm):
  class Meta:
    model = models.Tool
    fields = ('name', 'tool_type', 'config')
    widgets = {
      'tool_type': forms.Select(attrs={'id': 'type-id', 'data-url': ''}),
      'config': forms.Textarea(attrs={'id': 'config-id', 'class': 'h-auto', 'disabled': 'disabled'}),
    }

  def __init__(self, config_form_url, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['tool_type'].widget.attrs['data-url'] = config_form_url

  def customize(self, instance, *args, **kwargs):
    tool_type, config = instance.get_config()
    fields = ToolType.get_config_field(tool_type, config=config)
    config = {}

    for target in fields:
      config.update(target.asdict())
    instance.config = config

    return instance

class ToolConfigForm(_BaseConfigForm):
  def get_fields(self, type_id, config):
    return ToolType.get_config_field(type_id, config)

# =============
# = Assistant =
# =============
class AssistantForm(_BaseModelForm):
  class Meta:
    model = models.Assistant
    fields = ('name', 'system_message', 'agent', 'embedding', 'tools', 'is_interrupt')
    widgets = {
      'system_message': forms.Textarea(),
      'tools': forms.SelectMultiple(attrs={'class': 'h-auto overflow-auto'})
    }

  is_interrupt = forms.TypedChoiceField(
    choices=((True, gettext_lazy('Interrupt before the action node.')), (False, gettext_lazy('Interrupt a function execution immediately.'))),
    widget=forms.Select(),
    required=False,
  )

  def clean(self):
    cleaned_data = super().clean()
    # Get target model's instances
    agent = cleaned_data.get('agent', None)
    embedding = cleaned_data.get('embedding', None)
    tools = cleaned_data.get('tools', [])
    judged = [
      agent.is_owner(self.user) if agent is not None else True,
      embedding.is_owner(self.user) if embedding is not None else True,
      all([tool.is_owner(self.user) for tool in tools]),
    ]
    is_valid = all(judged)
    # Check owner of each instance
    if not is_valid:
      raise ValidationError(gettext_lazy('Invalid agent, embedding, or tools exist. Please check your selected items.'))

    return cleaned_data

  def save(self):
    instance = super().save()
    self.save_m2m()

    return instance

# ==========
# = Thread =
# ==========
class ThreadForm(forms.ModelForm):
  template_name = 'renderer/custom_model_form.html'

  class Meta:
    model = models.Thread
    fields = ('name', 'docfiles')
    widgets = {
      'docfiles': forms.SelectMultiple(attrs={'class': 'h-auto overflow-auto'})
    }

  def __init__(self, assistant, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.assistant = assistant

    for field in self.fields.values():
      _classes = field.widget.attrs.get('class', '')
      field.widget.attrs['class'] = f'{_classes} form-control'
      field.widget.attrs['placeholder'] = field.help_text

  def clean(self):
    if self.assistant is None:
      raise ValidationError(gettext_lazy('Invalid assistant exist.'))

    cleaned_data = super().clean()
    docfiles = cleaned_data.get('docfiles', [])
    valid_files = self.assistant.docfiles.all().values_list('pk', flat=True)
    is_valid = all([instance.pk in valid_files for instance in docfiles])

    if not is_valid:
      raise ValidationError(gettext_lazy('Invalid docfiles exist. Please check your selected items.'))

    return cleaned_data

  def save(self, *args, **kwargs):
    instance = super().save(commit=False)
    instance.assistant = self.assistant
    instance.save()
    self.save_m2m()

    return instance