"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

from django_asgi_lifespan.asgi import get_asgi_application
from .define_module import setup_default_setting

setup_default_setting()
async_django_application = get_asgi_application()

async def application(scope, receive, send):
  if scope['type'] in {'http', 'lifespan'}:
    await async_django_application(scope, receive, send)
  else:
    raise NotImplementedError(f"Unknown scope type {scope['type']}")