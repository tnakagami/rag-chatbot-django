from django.utils.translation import gettext_lazy
from django.urls import reverse_lazy, reverse
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
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
from . import forms
from . import models

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
    instance = self.get_object()
    is_valid = instance.user.pk == self.request.user.pk

    return is_valid

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

  def get(self, request, *args, **kwargs):
    # Ignore direct access
    return self.handle_no_permission()

class Index(LoginRequiredMixin, ListBreadcrumbMixin, ListView):
  model = models.Assistant
  template_name = 'chatbot/index.html'
  paginate_by = 10
  context_object_name = 'assistants'
  crumbs = [(gettext_lazy('Chatbot'), '')]

  def get_queryset(self):
    user = self.request.user

    return self.model.objects.collection_with_docfiles(user=user)

class SettingListView(LoginRequiredMixin, BaseBreadcrumbMixin, TemplateView):
  template_name = 'chatbot/settings.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Settings'), ''),
  ]

  def get_context_data(self, *args, **kwargs):
    user = self.request.user
    context = super().get_context_data(*args, **kwargs)
    context['agents'] = models.Agent.objects.get_own_items(user)
    context['embeddings'] = models.Embedding.objects.get_own_items(user)
    context['tools'] = models.Tool.objects.get_own_items(user)

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

class AssistantDeleteView(_CommonDeleteView):
  model = models.Assistant
  success_url = reverse_lazy('chatbot:index')

# ================
# = DocumentFile =
# ================
class DocumentFileCreateView(IsOwner, BaseBreadcrumbMixin, FormView):
  raise_exception = True
  form_class = forms.DocumentFileForm
  template_name = 'chatbot/document_file_form.html'
  crumbs = [
    (gettext_lazy('Chatbot'), reverse_lazy('chatbot:index')),
    (gettext_lazy('Create document file'), ''),
  ]
  success_url = reverse_lazy('chatbot:index')

  def get_object(self):
    pk = self.kwargs.get('assistant_pk')
    assistant = models.Assistant.objects.get_or_none(pk=pk)

    return assistant

  def get_form_kwargs(self, *args, **kwargs):
    kwargs = super().get_form_kwargs(*args, **kwargs)
    kwargs['assistant'] = self.get_object()

    return kwargs

  def form_valid(self, form):
    form.create_document_files()

    return super().form_valid(form)

class DocumentFileDeleteView(_CommonDeleteView):
  model = models.DocumentFile
  success_url = reverse_lazy('chatbot:index')