"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

from django_asgi_lifespan.asgi import get_asgi_application
from .define_module import setup_default_setting

setup_default_setting()
django_application = get_asgi_application()

async def application(scope, receive, send):
  if scope['type'] in {'http', 'lifespan'}:
    await django_application(scope, receive, send)
  else:
    raise NotImplementedError(
      f"Unknown scope type {scope['type']}"
    )