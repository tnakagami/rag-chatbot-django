from django.apps import AppConfig
from django_asgi_lifespan.register import register_lifespan_manager
from config.context import httpx_lifespan_manager

class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'

    def ready(self):
        import chatbot.schema
        register_lifespan_manager(context_manager=httpx_lifespan_manager)