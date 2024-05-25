from django.urls import include, path
from . import views

urlpatterns = [
  path('', views.Index.as_view(), name='index'),
  path('settings/', views.SettingListView.as_view(), name='settings'),
  # Agent
  path('settings/config/agent', views.AgentConfigFormCollection.as_view(), name='collect_agent_config'),
  path('settings/create/agent', views.AgentCreateView.as_view(), name='create_agent'),
  path('settings/update/agent/<int:pk>', views.AgentUpdateView.as_view(), name='update_agent'),
  path('settings/delete/agent/<int:pk>', views.AgentDeleteView.as_view(), name='delete_agent'),
  # Embedding
  path('settings/config/embedding', views.EmbeddingConfigFormCollection.as_view(), name='collect_embedding_config'),
  path('settings/create/embedding', views.EmbeddingCreateView.as_view(), name='create_embedding'),
  path('settings/update/embedding/<int:pk>', views.EmbeddingUpdateView.as_view(), name='update_embedding'),
  path('settings/delete/embedding/<int:pk>', views.EmbeddingDeleteView.as_view(), name='delete_embedding'),
  # Tool
  path('settings/config/tool', views.ToolConfigFormCollection.as_view(), name='collect_tool_config'),
  path('settings/create/tool', views.ToolCreateView.as_view(), name='create_tool'),
  path('settings/update/tool/<int:pk>', views.ToolUpdateView.as_view(), name='update_tool'),
  path('settings/delete/tool/<int:pk>', views.ToolDeleteView.as_view(), name='delete_tool'),
  # Assistant
  path('create/assistant', views.AssistantCreateView.as_view(), name='create_assistant'),
  path('update/assistant/<int:pk>', views.AssistantUpdateView.as_view(), name='update_assistant'),
  path('detail/assistant/<int:pk>', views.AssistantDetailView.as_view(), name='detail_assistant'),
  path('delete/assistant/<int:pk>', views.AssistantDeleteView.as_view(), name='delete_assistant'),
  # DocumentFile
  path('create/docfile/<int:assistant_pk>', views.DocumentFileView.as_view(), name='create_docfile'),
  path('delete/docfile/<int:pk>', views.DocumentFileDeleteView.as_view(), name='delete_docfile'),
  # Thread
  path('create/thread/<int:assistant_pk>', views.ThreadCreateView.as_view(), name='create_thread'),
  path('update/thread/<int:pk>', views.ThreadUpdateView.as_view(), name='update_thread'),
  path('detail/thread/<int:pk>', views.ThreadDetailView.as_view(), name='detail_thread'),
  path('delete/thread/<int:pk>', views.ThreadDeleteView.as_view(), name='delete_thread'),
]