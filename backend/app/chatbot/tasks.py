import logging
from celery import shared_task, Task
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from django_celery_results.models import TaskResult
from pathlib import Path
from . import models

g_logger = logging.getLogger(__name__)

def on_success_of_embedding_process(retval, task_id, args, kwargs):
  try:
    task = TaskResult.objects.get(task_id=task_id)
    task.delete()
  except Exception as ex:
    g_logger.warn(f'Task({task_id}) cannot be deleted.')

@shared_task
def embedding_process(assistant_pk, files):
  storage = FileSystemStorage()
  filefields = []

  for target in files:
    name = target['name']
    path_info = Path(target['path'])
    filefields += [File(path_info.open(mode='rb'), name=name)]

  assistant = models.Assistant.objects.get_or_none(pk=assistant_pk)
  ids = models.DocumentFile.from_files(assistant, filefields)

  for target in filefields:
    if not target.closed:
      target.close()
    storage.delete(target.name)

  return ids
