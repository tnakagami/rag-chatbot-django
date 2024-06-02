from celery import Celery
from celery.schedules import crontab
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

@app.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
  from chatbot import tasks

  sender.add_periodic_task(
    crontab(minute=3, hour=0),
    tasks.delete_successful_tasks.signature(),
  )