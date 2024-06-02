from django.contrib import admin
from django.utils.translation import gettext_lazy
from .models import Agent, Embedding, Tool, Assistant, DocumentFile, Thread
from .models.rag import EmbeddingStore, LangGraphCheckpoint

@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
  model = Agent
  fields = ['name', 'user', 'agent_type', 'config']
  list_display = ('name', 'user', 'agent_type')
  list_select_related = ('user',)
  list_filter = ('user', 'agent_type')
  search_fields = ('name', 'user', 'agent_type')
  ordering = ('pk',)

@admin.register(Embedding)
class EmbeddingAdmin(admin.ModelAdmin):
  model = Embedding
  fields = ['name', 'user', 'emb_type', 'config', 'distance_strategy']
  list_display = ('name', 'user', 'emb_type', 'distance_strategy')
  list_select_related = ('user',)
  list_filter = ('user', 'emb_type', 'distance_strategy')
  search_fields = ('name', 'user', 'emb_type')
  ordering = ('pk',)

@admin.register(Tool)
class ToolAdmin(admin.ModelAdmin):
  model = Tool
  fields = ['name', 'user', 'tool_type', 'config']
  list_display = ('name', 'user', 'tool_type')
  list_select_related = ('user',)
  list_filter = ('user', 'tool_type')
  search_fields = ('name', 'user', 'tool_type')
  ordering = ('pk',)

@admin.register(Assistant)
class AssistantAdmin(admin.ModelAdmin):
  model = Assistant
  fields = ['name', 'user', 'system_message', 'agent', 'embedding', 'tools', 'is_interrupt']
  list_display = ('name', 'user')
  list_select_related = ('user', 'agent', 'embedding')
  list_filter = ('user',)
  search_fields = ('name', 'user')
  ordering = ('pk',)

@admin.register(DocumentFile)
class DocumentFileAdmin(admin.ModelAdmin):
  model = DocumentFile
  fields = ['name', 'assistant', 'is_active']
  list_display = ('name', 'assistant', 'is_active')
  list_select_related = ('assistant',)
  list_filter = ('name', 'assistant', 'is_active')
  search_fields = ('name', 'assistant', 'is_active')
  ordering = ('pk',)

@admin.register(Thread)
class ThreadAdmin(admin.ModelAdmin):
  model = Thread
  fields = ['name', 'assistant']
  list_display = ('name', 'assistant')
  list_select_related = ('assistant',)
  list_filter = ('name', 'assistant')
  search_fields = ('name', 'assistant')
  ordering = ('pk',)

@admin.register(EmbeddingStore)
class EmbeddingStoreAdmin(admin.ModelAdmin):
  model = EmbeddingStore
  fields = ['assistant', 'embedding', 'document']
  list_display = ('assistant', 'short_document')
  list_select_related = ('assistant',)
  list_filter = ('assistant', 'document')
  search_fields = ('assistant', 'document')
  ordering = ('pk',)

  @admin.display(description='text')
  def short_document(self, instance):
    max_len = 32
    document = instance.document
    text = document[:max_len] if len(document) > max_len else document

    return text

@admin.register(LangGraphCheckpoint)
class LangGraphCheckpointAdmin(admin.ModelAdmin):
  model = LangGraphCheckpoint
  fields = ['thread', 'current_time', 'previous_time']
  list_display = ('thread', 'current_time', 'previous_time')
  list_select_related = ('thread',)
  list_filter = ('thread', 'current_time', 'previous_time')
  search_fields = ('thread', 'current_time', 'previous_time')
  ordering = ('pk',)