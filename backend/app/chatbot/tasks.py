from celery import shared_task
from django.core.files import File
from django.core.files.storage import FileSystemStorage
from pathlib import Path
from . import models

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
