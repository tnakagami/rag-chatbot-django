from celery import shared_task, states
from celery.utils.log import get_task_logger
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from django_celery_results.models import TaskResult
from pathlib import Path
from . import models

g_logger = get_task_logger(__name__)

@shared_task
def delete_successful_tasks():
  queryset = TaskResult.objects.filter(status=states.SUCCESS)

  if queryset.count() > 0:
    try:
      count, _ = queryset.delete()
      g_logger.info(f'The {count} tasks are deleted.')
    except Exception as ex:
      g_logger.error(f'Failed to delete the records that the status of celery task is {states.SUCCESS}({ex}).')

@shared_task(bind=True)
def embedding_process(self, assistant_pk, files, user_pk):
  storage = FileSystemStorage()
  filefields = []

  for target in files:
    name = target['name']
    path_info = Path(target['path'])
    filefields += [File(path_info.open(mode='rb'), name=name)]

  assistant = models.Assistant.objects.get_or_none(pk=assistant_pk)
  # Update database record
  assistant.set_task_result(self.request.id, user_pk)
  # Execute embedding process
  ids = models.DocumentFile.from_files(assistant, filefields)
  # Delete stored files
  for field, target in zip(filefields, files):
    saved_name = target['saved_name']

    if not field.closed:
      field.close()
    storage.delete(saved_name)

  return ids
