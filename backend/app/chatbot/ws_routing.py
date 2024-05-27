from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/thread/<int:thread_pk>', consumers.ThreadConsumer.as_asgi()),
]