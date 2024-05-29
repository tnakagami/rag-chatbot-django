import pytest
from chatbot.tasks import embedding_process
# For tests
from . import factories

@pytest.mark.chatbot
@pytest.mark.drf
@pytest.mark.django_db
@pytest.mark.parametrize('is_close,expected_count', [
  (True, 2),
  (False, 3),
  (False, 1),
], ids=['is-close', 'is-not-close-count-3', 'is-not-close-count-1'])
def test_check_embedding_process(mocker, is_close, expected_count):
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