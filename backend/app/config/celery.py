from celery import Celery
from .define_module import setup_default_setting
from django.conf import settings

# Set the default Django settings
setup_default_setting()
# Create Celery application
app = Celery('config')
# Setup configure of Celery application
app.config_from_object('django.conf:settings', namespace='CELERY')
# Load task modules from all registered Django apps.
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)