"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django_asgi_lifespan.asgi import get_asgi_application
from .define_module import setup_default_setting

setup_default_setting()
async_django_application = get_asgi_application()

# Load websocket's routing setting after Django default setting is loaded
from chatbot import ws_routing

application = ProtocolTypeRouter({
  'http': async_django_application,
  'lifespan': async_django_application,
  'websocket': AllowedHostsOriginValidator(
    AuthMiddlewareStack(
      URLRouter(
        ws_routing.websocket_urlpatterns
      )
    )
  )
})