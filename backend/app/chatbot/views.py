from rest_framework_simplejwt.tokens import RefreshToken
from django.utils.translation import gettext_lazy
from django.urls import reverse_lazy, reverse
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.utils.functional import cached_property
from django.views.generic import (
  View,
  TemplateView,
  ListView,
  CreateView,
  UpdateView,
  DeleteView,
  DetailView,
  FormView,
)
from view_breadcrumbs import (
  BaseBreadcrumbMixin,
  ListBreadcrumbMixin,
  CreateBreadcrumbMixin,
  UpdateBreadcrumbMixin,
  DetailBreadcrumbMixin,
)
from . import forms, models

class JsonResponseMixin:
  raise_exception = True
  http_method_names = ['get']

  def get(self, request, *args, **kwargs):
    context = self.get_context_data(*args, **kwargs)
    data = self.converter(**context)

    return JsonResponse(data)

  def converter(self, **kwargs):
    return {key: value for key, value in kwargs.items() if isinstance(value, (str, int, list, dict))}

class IsOwner(UserPassesTestMixin):
  def test_func(self):
    user = self.request.user
    instance = self.get_object()

    return instance.is_owner(user)

class _BaseCreateUpdateView(LoginRequiredMixin):
  raise_exception = True

  def get_form_kwargs(self, *args, **kwargs):
    kwargs = super().get_form_kwargs(*args, **kwargs)
    kwargs['user'] = self.request.user

    return kwargs

class _CommonCreateUpdateWithConfigView(_BaseCreateUpdateView):
  model = None
  form_class = None
  config_form_class = None
  template_name = None
  config_form_url = None
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Settings'), reverse_lazy('chatbot:settings')),
  ]
  success_url = reverse_lazy('chatbot:settings')

  def get_form_kwargs(self, *args, **kwargs):
    kwargs = super().get_form_kwargs(*args, **kwargs)
    kwargs['config_form_url'] = self.config_form_url

    return kwargs

class _CommonConfigCollection(JsonResponseMixin, View):
  config_form_class = None
  http_method_names = ['get']

  def get_context_data(self, *args, **kwargs):
    params = {}
    type_id = self.request.GET.get('type_id', None)

    if type_id is not None:
      params.update({'type_id': int(type_id)})
    config_form = self.config_form_class(**params)
    context = {
      'config_html_form': config_form.render(),
      'config_exists': config_form.config_exists,
    }

    return context

class _CommonDeleteView(LoginRequiredMixin, IsOwner, DeleteView):
  raise_exception = True
  model = None
  success_url = None
  http_method_names = ['post']

class JsonWebTokenView(LoginRequiredMixin, JsonResponseMixin, View):
  http_method_names = ['get']

  def get_context_data(self, *args, **kwargs):
    refresh = RefreshToken.for_user(self.request.user)
    access_token = str(refresh.access_token)
    context = {
      'token': access_token
    }

    return context

class Index(LoginRequiredMixin, ListBreadcrumbMixin, ListView):
  model = models.Assistant
  template_name = 'chatbot/index.html'
  paginate_by = 10
  context_object_name = 'assistants'
  crumbs = [(gettext_lazy('Chatbot'), '')]

  def get_queryset(self):
    user = self.request.user

    return self.model.objects.collection_with_docfiles(user=user)

class TaskListView(LoginRequiredMixin, BaseBreadcrumbMixin, TemplateView):
  template_name = 'chatbot/tasks.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Tasks'), ''),
  ]

  def get_context_data(self, *args, **kwargs):
    user = self.request.user
    context = super().get_context_data(*args, **kwargs)
    tasks = models.Assistant.objects.collect_own_tasks(user=user)
    context['tasks'] = [
      {
        'task_id': record.task_id,
        'name': record.task_name,
        'status': record.status,
        'created': models.convert_timezone(record.date_created),
      }
      for record in tasks
    ]

    return context

class SettingListView(LoginRequiredMixin, BaseBreadcrumbMixin, TemplateView):
  template_name = 'chatbot/settings.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Settings'), ''),
  ]

  def get_context_data(self, *args, **kwargs):
    user = self.request.user
    context = super().get_context_data(*args, **kwargs)
    context['agents'] = user.agent_configs.all()
    context['embeddings'] = user.embedding_configs.all()
    context['tools'] = user.tool_configs.all()

    return context

class AgentConfigFormCollection(_CommonConfigCollection):
  config_form_class = forms.AgentConfigForm

class AgentCreateView(_CommonCreateUpdateWithConfigView, CreateBreadcrumbMixin, CreateView):
  model = models.Agent
  form_class = forms.AgentForm
  config_form_class = forms.AgentConfigForm
  template_name = 'chatbot/agent_form.html'
  config_form_url = reverse_lazy('chatbot:collect_agent_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Create agent'), '')]

class AgentUpdateView(_CommonCreateUpdateWithConfigView, IsOwner, UpdateBreadcrumbMixin, UpdateView):
  model = models.Agent
  form_class = forms.AgentForm
  config_form_class = forms.AgentConfigForm
  template_name = 'chatbot/agent_form.html'
  config_form_url = reverse_lazy('chatbot:collect_agent_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Update agent'), '')]

class AgentDeleteView(_CommonDeleteView):
  model = models.Agent
  success_url = reverse_lazy('chatbot:settings')

class EmbeddingConfigFormCollection(_CommonConfigCollection):
  config_form_class = forms.EmbeddingConfigForm

class EmbeddingCreateView(_CommonCreateUpdateWithConfigView, CreateBreadcrumbMixin, CreateView):
  model = models.Embedding
  form_class = forms.EmbeddingForm
  config_form_class = forms.EmbeddingConfigForm
  template_name = 'chatbot/embedding_form.html'
  config_form_url = reverse_lazy('chatbot:collect_embedding_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Create embedding'), '')]

class EmbeddingUpdateView(_CommonCreateUpdateWithConfigView, IsOwner, UpdateBreadcrumbMixin, UpdateView):
  model = models.Embedding
  form_class = forms.EmbeddingForm
  config_form_class = forms.EmbeddingConfigForm
  template_name = 'chatbot/embedding_form.html'
  config_form_url = reverse_lazy('chatbot:collect_embedding_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Update embedding'), '')]

class EmbeddingDeleteView(_CommonDeleteView):
  model = models.Embedding
  success_url = reverse_lazy('chatbot:settings')

class ToolConfigFormCollection(_CommonConfigCollection):
  config_form_class = forms.ToolConfigForm

class ToolCreateView(_CommonCreateUpdateWithConfigView, CreateBreadcrumbMixin, CreateView):
  model = models.Tool
  form_class = forms.ToolForm
  config_form_class = forms.ToolConfigForm
  template_name = 'chatbot/tool_form.html'
  config_form_url = reverse_lazy('chatbot:collect_tool_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Create tool'), '')]

class ToolUpdateView(_CommonCreateUpdateWithConfigView, IsOwner, UpdateBreadcrumbMixin, UpdateView):
  model = models.Tool
  form_class = forms.ToolForm
  config_form_class = forms.ToolConfigForm
  template_name = 'chatbot/tool_form.html'
  config_form_url = reverse_lazy('chatbot:collect_tool_config')
  crumbs = _CommonCreateUpdateWithConfigView.crumbs + [(gettext_lazy('Update tool'), '')]

class ToolDeleteView(_CommonDeleteView):
  model = models.Tool
  success_url = reverse_lazy('chatbot:settings')

# =============
# = Assistant =
# =============
class AssistantCreateView(_BaseCreateUpdateView, CreateBreadcrumbMixin, CreateView):
  model = models.Assistant
  form_class = forms.AssistantForm
  template_name = 'chatbot/assistant_form.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Create assistant'), ''),
  ]
  success_url = reverse_lazy('chatbot:index')

class AssistantUpdateView(_BaseCreateUpdateView, IsOwner, UpdateBreadcrumbMixin, UpdateView):
  model = models.Assistant
  form_class = forms.AssistantForm
  template_name = 'chatbot/assistant_form.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Update assistant'), ''),
  ]
  success_url = reverse_lazy('chatbot:index')

class AssistantDetailView(LoginRequiredMixin, IsOwner, DetailBreadcrumbMixin, DetailView):
  raise_exception = True
  model = models.Assistant
  template_name = 'chatbot/target_assistant.html'
  context_object_name = 'assistant'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Assistant'), ''),
  ]

class AssistantDeleteView(_CommonDeleteView):
  model = models.Assistant
  success_url = reverse_lazy('chatbot:index')

# ================
# = DocumentFile =
# ================
class DocumentFileView(LoginRequiredMixin, IsOwner, BaseBreadcrumbMixin, TemplateView):
  raise_exception = True
  template_name = 'chatbot/document_file_form.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Create document file'), ''),
  ]
  http_method_names = ['get']

  def get_object(self):
    pk = self.kwargs.get('assistant_pk')
    instance = models.Assistant.objects.get_or_none(pk=pk)

    return instance

  def get_context_data(self, *args, **kwargs):
    context = super().get_context_data(*args, **kwargs)
    context['docfile_url'] = reverse('api:chatbot:docfile_list')
    context['token_url'] = reverse('chatbot:token')
    context['assistant_pk'] = self.kwargs.get('assistant_pk')

    return context

class DocumentFileDeleteView(_CommonDeleteView):
  model = models.DocumentFile
  success_url = reverse_lazy('chatbot:index')

# ==========
# = Thread =
# ==========
class _ThreadCreateUpdateView(LoginRequiredMixin, IsOwner):
  raise_exception = True
  model = models.Thread
  form_class = forms.ThreadForm
  template_name = 'chatbot/thread_form.html'

  @cached_property
  def crumbs(self):
    return [
    (gettext_lazy('Chatbot'), reverse('chatbot:index')),
    (gettext_lazy('Assistant'), self.get_success_url()),
    (gettext_lazy('Create thread'), ''),
  ]

  def get_form_kwargs(self, *args, **kwargs):
    kwargs = super().get_form_kwargs(*args, **kwargs)
    kwargs['assistant'] = self.get_assistant()

    return kwargs

  def get_context_data(self, *args, **kwargs):
    context = super().get_context_data(*args, **kwargs)
    context['assistant'] = self.get_assistant()

    return context

  def get_success_url(self):
    return reverse('chatbot:detail_assistant', kwargs={'pk': self.get_pk()})

class ThreadCreateView(_ThreadCreateUpdateView, CreateBreadcrumbMixin, CreateView):
  def get_pk(self):
    return self.kwargs.get('assistant_pk')

  def get_object(self):
    return self.get_assistant()

  def get_assistant(self):
    pk = self.get_pk()
    assistant = models.Assistant.objects.get_or_none(pk=pk)

    return assistant

class ThreadUpdateView(_ThreadCreateUpdateView, UpdateBreadcrumbMixin, UpdateView):
  def get_pk(self):
    thread = self.get_object()
    pk = thread.assistant.pk

    return pk

  def get_assistant(self):
    thread = self.get_object()
    assistant = thread.assistant

    return assistant

class ThreadDetailView(LoginRequiredMixin, IsOwner, DetailBreadcrumbMixin, DetailView):
  raise_exception = True
  model = models.Thread
  template_name = 'chatbot/target_thread.html'
  context_object_name = 'thread'

  def _get_parent_link(self):
    thread = self.get_object()
    pk = thread.assistant.pk

    return reverse('chatbot:detail_assistant', kwargs={'pk': pk})

  @cached_property
  def crumbs(self):
    return [
    (gettext_lazy('Chatbot'), reverse('chatbot:index')),
    (gettext_lazy('Assistant'), self._get_parent_link()),
    (gettext_lazy('Thread'), ''),
  ]

  def get_context_data(self, *args, **kwargs):
    context = super().get_context_data(*args, **kwargs)
    context['token_url'] = reverse('chatbot:token')
    context['stream_url'] = reverse('api:chatbot:event_stream')

    return context

class ThreadDeleteView(_CommonDeleteView):
  model = models.Thread
  context_object_name = 'thread'

  def get_success_url(self):
    pk = self.object.assistant.pk

    return reverse('chatbot:detail_assistant', kwargs={'pk': pk})