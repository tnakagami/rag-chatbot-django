import pytest
from chatbot.tasks import delete_successful_tasks, embedding_process
# For tests
import chatbot.tasks
from celery import states
from django_celery_results.models import TaskResult
from . import factories

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('num_tasks,is_raise,expected,log_message', [
  (0, True, 1, ''),
  (1, False, 1, 'The 1 tasks are deleted.'),
  (2, False, 1, 'The 2 tasks are deleted.'),
  (1, True, 2, 'Failed to delete the records'),
], ids=['no-tasks-exist', 'one-task-record-can-delete', 'two-task-records-can-delete', 'failed-to-delete-task'])
def test_check_delete_successful_tasks_method(mocker, num_tasks, is_raise, expected, log_message):
  class FakeLogger:
    def __init__(self):
      self.message = ''
    def store(self, message):
      self.message = message

  _ = factories.TaskResultFactory.create_batch(num_tasks, status=states.SUCCESS)
  _ = factories.TaskResultFactory(status=states.PENDING)
  fake_logger = FakeLogger()
  mocker.patch.object(chatbot.tasks.g_logger, 'info', side_effect=lambda msg: fake_logger.store(msg))
  mocker.patch.object(chatbot.tasks.g_logger, 'error', side_effect=lambda msg: fake_logger.store(msg))

  if is_raise:
    mocker.patch('django.db.models.query.QuerySet.delete', side_effect=Exception())
  delete_successful_tasks()
  total = TaskResult.objects.all().count()

  assert log_message in fake_logger.message
  assert total == expected

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('is_close,expected_count', [
  (True, 2),
  (False, 3),
  (False, 1),
], ids=['is-close', 'is-not-close-count-3', 'is-not-close-count-1'])
def test_check_embedding_process_method(mocker, is_close, expected_count):
  class FakeFile:
    def __init__(self, *args, **kwargs):
      self._closed = False
    
    @property
    def closed(self):
      return self._closed
    @closed.setter
    def closed(self, is_closed):
      self._closed = is_closed
    def close(self):
      pass

  class FakePath:
    def __init__(self, *args, **kwargs):
      pass
    def open(self, *args, **kwargs):
      return object()

  mocker.patch('chatbot.tasks.File', new=FakeFile)
  mocker.patch('chatbot.tasks.FileSystemStorage.delete', return_value=None)
  mocker.patch('chatbot.tasks.Path', new=FakePath)
  mocker.patch('chatbot.models.Assistant.set_task_result', return_value=None)
  mocker.patch('chatbot.models.DocumentFile.from_files', side_effect=lambda instance, filefields: [idx for idx, _ in enumerate(filefields)])
  # Create test data
  user = factories.UserFactory()
  assistant = factories.AssistantFactory(
    user=user,
    agent=factories.AgentFactory(user=user),
    embedding=factories.EmbeddingFactory(user=user),
  )
  files = [
    {'name': f'test{idx}', 'path': f'sample{idx}', 'saved_name': f'test{idx}-test.txt'}
    for idx in range(expected_count)
  ]

  try:
    ids = embedding_process(assistant.pk, files, user.pk)
  except:
    pytest.fail(f'Raise exception when is_close is {is_close}')

  assert len(ids) == len(files)