from django.urls import path
from . import drf_views, routers

config_router = routers.CustomConfigRouter()
config_router.register('agents', drf_views.AgentViewSet, basename='agent')
config_router.register('embeddings', drf_views.EmbeddingViewSet, basename='embedding')
config_router.register('tools', drf_views.ToolViewSet, basename='tool')
simple_router = routers.CustomSimpleRouter()
simple_router.register('assistants', drf_views.AssistantViewSet, basename='assistant')
simple_router.register('threads', drf_views.ThreadViewSet, basename='thread')
docfile_router = routers.CustomDocfileRouter()
docfile_router.register('docfiles', drf_views.DocumentFileViewSet, basename='docfile')
event_stream_urls = [
  path('event-stream', drf_views.EventStreamView.as_view(), name='event_stream'),
]

urlpatterns = config_router.urls + simple_router.urls + docfile_router.urls + event_stream_urls