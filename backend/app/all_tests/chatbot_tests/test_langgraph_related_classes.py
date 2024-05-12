import pytest
from chatbot.models.utils.checkpoint import _ThreadConfig, DjangoPostgresCheckpoint
from chatbot.models.utils.vectorstore import DistanceStrategy, CustomVectorStore
# For test
import numpy as np
from datetime import datetime, timedelta
from django.utils.timezone import make_aware
from asgiref.sync import sync_to_async
from chatbot.models.rag import LangGraphCheckpoint, EmbeddingStore
from langchain.schema import Document
from langchain_core.runnables import ConfigurableFieldSpec, RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from . import factories

def _convert_datetime2isoformat(timestamp):
  return timestamp.isoformat() if timestamp else None

@pytest.fixture
def get_checkpoint_instance():
  return DjangoPostgresCheckpoint(manager=LangGraphCheckpoint.objects)

@pytest.fixture
@pytest.mark.django_db
def create_checkpoints_of_same_thread(django_db_blocker):
  def inner():
    current_time = make_aware(datetime(2000, 1, 2, 9, 0, 0))
    previous_time = make_aware(datetime(2000, 1, 1, 9, 0, 0))

    with django_db_blocker.unblock():
      thread = factories.ThreadFactory()
      _ = factories.CheckpointFactory(thread=thread, current_time=current_time,  previous_time=previous_time)
      _ = factories.CheckpointFactory(thread=thread, current_time=previous_time, no_previous_exists=True)
      thread_id = thread.pk

    current_ts = _convert_datetime2isoformat(current_time)
    previous_ts = _convert_datetime2isoformat(previous_time)
    config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=current_ts).get_config_dict())

    return thread_id, current_ts, previous_ts, config

  return inner

@pytest.fixture
@pytest.mark.django_db
def create_checkpoints_of_different_thread(django_db_blocker):
  def inner():
    time = make_aware(datetime(2000, 1, 2, 9, 0, 0))

    with django_db_blocker.unblock():
      thread1st = factories.ThreadFactory()
      thread2nd = factories.ThreadFactory()
      _ = factories.CheckpointFactory(thread=thread1st, current_time=time, no_previous_exists=True)
      _ = factories.CheckpointFactory(thread=thread2nd, current_time=time, no_previous_exists=True)
      thread_id = thread1st.pk

    current_ts = _convert_datetime2isoformat(time)
    config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=current_ts).get_config_dict())

    return thread_id, current_ts, config

  return inner

# ====================================
# = Test of DjangoPostgresCheckpoint =
# ====================================
@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.parametrize([
  'thread_id', 'thread_ts', 'exptected_ts', 'expected_config',
], [
  (3, '2010-01-22T18:54:32', datetime(2010,1,22,18,54,32), {'configrable': {'thread_id': 3, 'thread_ts': '2010-01-22T18:54:32'}}),
  (3, datetime(2010,1,22,0,0,0), datetime(2010,1,22,0,0,0), {'configrable': {'thread_id': 3, 'thread_ts': '2010-01-22T00:00:00'}}),
], ids=['is-isoformat', 'is-datetime-format'])
def test_check_checkpoint_of_valid_config(thread_id, thread_ts, exptected_ts, expected_config):
  instance = _ThreadConfig(thread_id=thread_id, thread_ts=thread_ts)
  config = instance.get_config_dict()

  assert instance.thread_id == thread_id
  assert instance.thread_ts == exptected_ts
  assert config == expected_config

@pytest.mark.chatbot
@pytest.mark.langgraph
def test_check_checkpoint_when_ts_is_none():
  instance = _ThreadConfig(thread_id=1, thread_ts=None)
  config = instance.get_config_dict()

  assert instance.thread_id == 1
  assert instance.thread_ts is None
  assert config is None

@pytest.mark.chatbot
@pytest.mark.langgraph
def test_check_config_specs_member(get_checkpoint_instance):
  instance = get_checkpoint_instance
  thread_id_spec, thread_ts_spec = instance.config_specs

  assert isinstance(thread_id_spec, ConfigurableFieldSpec)
  assert isinstance(thread_ts_spec, ConfigurableFieldSpec)
  assert thread_id_spec.id == 'thread_id'
  assert thread_id_spec.name == 'Thread ID'

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_list_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  thread_id, current_ts, previous_ts, config = getter()
  outputs = instance.list(config)
  conf1st, _, conf2nd = next(outputs)
  conf3rd, _, no_prev = next(outputs)
  _key = 'configrable'

  assert _key in conf1st.keys()
  assert _key in conf2nd.keys()
  assert _key in conf3rd.keys()
  assert no_prev is None
  assert conf1st[_key]['thread_id'] == thread_id
  assert conf2nd[_key]['thread_id'] == thread_id
  assert conf3rd[_key]['thread_id'] == thread_id
  assert conf1st[_key]['thread_ts'] == current_ts
  assert conf2nd[_key]['thread_ts'] == previous_ts
  assert conf3rd[_key]['thread_ts'] == previous_ts
  with pytest.raises(StopIteration):
    _ = next(outputs)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_list_method_of_invalid_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  _ = getter()
  record = instance.manager.order_by('-thread__pk').first()
  thread_id = record.thread.pk + 1
  config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=record.current_time).get_config_dict())
  outputs = instance.list(config)

  with pytest.raises(StopIteration):
    _ = next(outputs)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_get_tuple_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  thread_id, current_ts, previous_ts, config = getter()
  conf1st, _, conf2nd = instance.get_tuple(config)
  _key = 'configrable'

  assert _key in conf1st.keys()
  assert _key in conf2nd.keys()
  assert conf1st[_key]['thread_id'] == thread_id
  assert conf1st[_key]['thread_ts'] == current_ts
  assert conf2nd[_key]['thread_id'] == thread_id
  assert conf2nd[_key]['thread_ts'] == previous_ts

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_get_tuple_method_of_different_checkpoint(get_checkpoint_instance, create_checkpoints_of_different_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_different_thread
  thread_id, timestamp, config = getter()
  conf1st, _, conf2nd = instance.get_tuple(config)
  _key = 'configrable'

  assert _key in conf1st.keys()
  assert conf1st[_key]['thread_id'] == thread_id
  assert conf1st[_key]['thread_ts'] == timestamp
  assert conf2nd is None

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_get_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  thread_id, _, _, config = getter()
  record = instance.manager.filter(thread__pk=thread_id).order_by('-thread__pk').first()
  output_checkpoint = instance.get(config)

  assert record.checkpoint == output_checkpoint

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_get_method_no_checkpoint(get_checkpoint_instance):
  instance = get_checkpoint_instance
  record = instance.manager.order_by('-thread__pk').first()
  thread_id = record.thread.pk + 1 if record else 1
  timestamp = make_aware(datetime(2015,10,14,17,34,13))
  config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=timestamp).get_config_dict())
  output = instance.get(config)

  assert output is None

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_put_method_when_no_previous_exists(get_checkpoint_instance):
  instance = get_checkpoint_instance
  thread = factories.ThreadFactory()
  thread_id = thread.pk
  current_time = make_aware(datetime(2001,2,3,10,12,9))
  timestamp = _convert_datetime2isoformat(current_time)
  config = RunnableConfig({'configrable': {'thread_id': thread_id, 'thread_ts': None}})
  checkpoint = empty_checkpoint()
  checkpoint['ts'] = timestamp
  # Execute target method
  out = instance.put(config, checkpoint)
  _key = 'configrable'

  assert _key in out.keys()
  assert out[_key]['thread_id'] == thread_id
  assert out[_key]['thread_ts'] == timestamp
  assert instance.manager.filter(thread__pk=thread_id, current_time=current_time).exists()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_put_method_when_previous_exists(get_checkpoint_instance, create_checkpoints_of_different_thread):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_different_thread
  thread_id, previous_ts, config = getter()
  previous_time = datetime.fromisoformat(previous_ts)
  new_time = previous_time + timedelta(days=1)
  timestamp = _convert_datetime2isoformat(new_time)
  checkpoint = empty_checkpoint()
  checkpoint['ts'] = timestamp
  # Execute target method
  out = instance.put(config, checkpoint)
  _key = 'configrable'

  assert out[_key]['thread_id'] == thread_id
  assert out[_key]['thread_ts'] == timestamp
  assert instance.manager.filter(
    thread__pk=thread_id,
    current_time=new_time,
    previous_time=previous_time,
  ).exists()

@pytest.fixture
def finalize_db_connection(django_db_blocker):
  def inner():
    with django_db_blocker.unblock():
      from django.db import connections

      for connection in connections.all():
        connection.close()

  return inner

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_alist_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread, finalize_db_connection):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  thread_id, current_ts, previous_ts, config = await sync_to_async(getter)()
  outputs = instance.alist(config)
  conf1st, _, conf2nd = await anext(outputs)
  conf3rd, _, no_prev = await anext(outputs)
  _key = 'configrable'

  assert _key in conf1st.keys()
  assert _key in conf2nd.keys()
  assert _key in conf3rd.keys()
  assert no_prev is None
  assert conf1st[_key]['thread_id'] == thread_id
  assert conf2nd[_key]['thread_id'] == thread_id
  assert conf3rd[_key]['thread_id'] == thread_id
  assert conf1st[_key]['thread_ts'] == current_ts
  assert conf2nd[_key]['thread_ts'] == previous_ts
  assert conf3rd[_key]['thread_ts'] == previous_ts

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_alist_method_of_invalid_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread, finalize_db_connection):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  _ = await sync_to_async(getter)()
  targets = await sync_to_async(instance.manager.select_related('thread').order_by)('-thread__pk')
  record = await targets.afirst()
  thread_id = record.thread.pk + 1
  config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=record.current_time).get_config_dict())
  outputs = instance.alist(config)

  with pytest.raises(StopAsyncIteration):
    _ = await anext(outputs)

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_aget_tuple_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread, finalize_db_connection):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  thread_id, current_ts, previous_ts, config = await sync_to_async(getter)()
  conf1st, _, conf2nd = await instance.aget_tuple(config)
  _key = 'configrable'

  assert _key in conf1st.keys()
  assert _key in conf2nd.keys()
  assert conf1st[_key]['thread_id'] == thread_id
  assert conf1st[_key]['thread_ts'] == current_ts
  assert conf2nd[_key]['thread_id'] == thread_id
  assert conf2nd[_key]['thread_ts'] == previous_ts

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_aget_method_of_checkpoint(get_checkpoint_instance, create_checkpoints_of_same_thread, finalize_db_connection):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_same_thread
  _, _, _, config = await sync_to_async(getter)()
  targets = await sync_to_async(instance.manager.select_related('thread').order_by)('thread__pk')
  record = await targets.afirst()
  output_checkpoint = await instance.aget(config)

  assert record.checkpoint == output_checkpoint

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_aget_method_no_checkpoint(get_checkpoint_instance, finalize_db_connection):
  instance = get_checkpoint_instance
  targets = await sync_to_async(instance.manager.select_related('thread').order_by)('-thread__pk')
  record = await targets.afirst()
  thread_id = record.thread.pk + 1 if record else 1
  timestamp = make_aware(datetime(2015,10,14,17,34,13))
  config = RunnableConfig(_ThreadConfig(thread_id=thread_id, thread_ts=timestamp).get_config_dict())
  output = await instance.aget(config)

  assert output is None

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_aput_method(get_checkpoint_instance, create_checkpoints_of_different_thread, finalize_db_connection):
  instance = get_checkpoint_instance
  getter = create_checkpoints_of_different_thread
  thread_id, previous_ts, config = await sync_to_async(getter)()
  previous_time = datetime.fromisoformat(previous_ts)
  new_time = previous_time + timedelta(days=1)
  timestamp = _convert_datetime2isoformat(new_time)
  checkpoint = empty_checkpoint()
  checkpoint['ts'] = timestamp
  # Execute target method
  out = await instance.aput(config, checkpoint)
  _key = 'configrable'
  filtered_query = await sync_to_async(instance.manager.filter)(
    thread__pk=thread_id,
    current_time=new_time,
    previous_time=previous_time,
  )

  assert out[_key]['thread_id'] == thread_id
  assert out[_key]['thread_ts'] == timestamp
  assert await filtered_query.aexists()

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()

# =============================
# = Test of CustomVectorStore =
# =============================
@pytest.fixture
def get_vectorstore_instance():
  class DummyEmbedding:
    def __init__(self, ndim):
      self.nearest_neighbor = -1
      self.emb_query = [float(idx) for idx in range(ndim)]

    def embed_query(self, text):
      return self.emb_query

    def embed_documents(self, texts):
      return [self.embed_query(text=query) for query in texts]

  return CustomVectorStore(
    manager=EmbeddingStore.objects,
    strategy=DistanceStrategy.COSINE,
    embedding_function=DummyEmbedding(ndim=10),
  )

@pytest.fixture
def get_instance_with_data(get_vectorstore_instance):
  instance = get_vectorstore_instance
  stores = factories.EmbeddingStoreFactory.build_batch(5)
  texts = [record.document for record in stores]
  embds = [record.embedding for record in stores]

  return instance, texts, embds

@pytest.fixture
def get_vectors(get_normalizer):
  normalizer = get_normalizer
  base_vector = normalizer(np.linspace(0.1, 2.1, 10))
  scales = np.array([-2.01, 1.88, 0.99, 2.3, -1.7, 0.001])
  nearest_neighbor = np.abs(scales - 1).argsort()[0]

  return base_vector, scales, nearest_neighbor

@pytest.fixture
@pytest.mark.django_db
def create_embeddingstores(django_db_blocker, get_vectors, get_normalizer, get_vectorstore_instance):
  normalizer = get_normalizer
  nearest_embedding, scales, nearest_neighbor = get_vectors
  instance = get_vectorstore_instance
  instance.embeddings.emb_query = nearest_embedding
  instance.embeddings.nearest_neighbor = nearest_neighbor

  ndim = nearest_embedding.shape[0]
  embeddings = [normalizer(np.power(nearest_embedding, scale)) for scale in scales]
  records = []

  with django_db_blocker.unblock():
    assistant = factories.AssistantFactory()

    for embedding in embeddings:
      record = factories.EmbeddingStoreFactory.build(assistant=assistant, ndim=ndim)
      record.embedding = embedding
      record.save()
      records.append(record)

  return instance, records

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_add_embeddings_method_of_vectorstore(get_instance_with_data):
  instance, texts, embeddings = get_instance_with_data
  assistant = factories.AssistantFactory()
  outputs = instance.add_embeddings(
    texts=texts,
    embeddings=embeddings,
    assistant_id=assistant.pk,
  )
  records = list(instance.manager.all().order_by('pk'))

  assert all([original == predicted for original, predicted in zip(texts, outputs)])
  assert len(records) == len(texts)
  assert all([
    all([record.document == document, (np.fabs(record.embedding - embedding) < 1e-7).all()])
    for record, document, embedding in zip(records, texts, embeddings)
  ])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_add_texts_method_of_vectorstore(get_instance_with_data):
  instance, texts, _ = get_instance_with_data
  assistant = factories.AssistantFactory()
  outputs = instance.add_texts(
    texts=texts,
    assistant_id=assistant.pk,
  )
  records = list(instance.manager.order_by('pk'))

  assert all([original == predicted for original, predicted in zip(texts, outputs)])
  assert len(records) == len(texts)
  assert all([record.document == document for record, document in zip(records, texts)])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore(create_embeddingstores):
  instance, records = create_embeddingstores
  ids = [str(record.pk) for record in records]
  result = instance.delete(ids)

  assert result
  assert instance.manager.all().count() == 0

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore_when_specific_records_are_deleted(create_embeddingstores):
  instance, records = create_embeddingstores
  ids = [str(record.pk) for record in records]
  threshold = len(ids) // 2
  targets = ids[:threshold]
  rests = list(map(int, ids[threshold:]))
  result = instance.delete(targets)

  assert result
  assert instance.manager.filter(pk__in=rests).count() == len(rests)
  assert instance.manager.filter(pk__in=targets).count() == 0

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore_when_invalid_ids_are_included(create_embeddingstores):
  instance, records = create_embeddingstores
  ids = [str(record.pk) for record in records]
  targets = list(iter(ids))
  targets[0] = '-'
  targets[-1] = 'abc'
  rests = [int(ids[0]), int(ids[-1])]
  result = instance.delete(targets)

  assert result
  assert instance.manager.all().count() == len(rests)
  assert instance.manager.filter(pk__in=rests).count() == len(rests)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore_when_ids_is_empty(create_embeddingstores):
  instance, records = create_embeddingstores
  result = instance.delete([])

  assert not result
  assert instance.manager.all().count() == len(records)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore_when_ids_is_none(create_embeddingstores):
  instance, records = create_embeddingstores
  result = instance.delete(None)

  assert not result
  assert instance.manager.all().count() == len(records)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_delete_method_of_vectorstore_in_the_case_of_raise_exception(create_embeddingstores, mocker):
  from django.db import IntegrityError
  instance, records = create_embeddingstores
  ids = [str(records[0])]
  mocker.patch('django.db.models.query.QuerySet.delete', side_effect=IntegrityError())
  result = instance.delete(ids)

  assert not result

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.parametrize([
  'distance_type',
  'score_fn',
], [
  (None, lambda xs, val: True),
  (DistanceStrategy.EUCLIDEAN, None),
  (DistanceStrategy.COSINE, None),
  (DistanceStrategy.MAX_INNER_PRODUCT, None),
], ids=['set-score-function', 'is-euclidean', 'is-cosine', 'is-max-inner-product'])
def test_check_score_fn_of_vectorstore(get_vectorstore_instance, distance_type, score_fn):
  base_instance = get_vectorstore_instance
  funcs = {
    DistanceStrategy.EUCLIDEAN: base_instance._euclidean_relevance_score_fn,
    DistanceStrategy.COSINE: base_instance._cosine_relevance_score_fn,
    DistanceStrategy.MAX_INNER_PRODUCT: base_instance._max_inner_product_relevance_score_fn,
  }
  instance = CustomVectorStore(
    manager=base_instance.manager,
    strategy=distance_type,
    embedding_function=base_instance.embeddings,
    relevance_score_fn=score_fn,
  )
  expected = funcs.get(distance_type, score_fn)
  ret_func = instance._select_relevance_score_fn()

  assert callable(ret_func)
  assert isinstance(ret_func, type(expected))

@pytest.mark.chatbot
@pytest.mark.langgraph
def test_check_invalid_score_fn_of_vectorstore(get_vectorstore_instance):
  base_instance = get_vectorstore_instance
  instance = CustomVectorStore(
    manager=base_instance.manager,
    strategy=None,
    embedding_function=base_instance.embeddings,
  )

  with pytest.raises(ValueError) as ex:
    _ = instance._select_relevance_score_fn()
  assert 'No supported' in str(ex.value)

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
@pytest.mark.parametrize('distance_type', [
  DistanceStrategy.EUCLIDEAN,
  DistanceStrategy.COSINE,
  DistanceStrategy.MAX_INNER_PRODUCT,
], ids=['Euclidean-distance', 'Cosine-similarity', 'max-inner-product'])
def test_check_private_method_of_vectorstore(create_embeddingstores, distance_type):
  base_instance, records = create_embeddingstores
  instance = CustomVectorStore(
    manager=base_instance.manager,
    strategy=distance_type,
    embedding_function=base_instance.embeddings,
  )
  assistant = records[0].assistant
  ids = [_record.pk for _record in records]
  nn_pk = ids[instance.embeddings.nearest_neighbor]
  # Collect private method
  target_func = instance._CustomVectorStore__collect_records
  # Call the function
  queryset = target_func(instance.embeddings.emb_query, assistant_id=assistant.pk)
  record = queryset.first()

  assert record.pk == nn_pk

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.parametrize('exist_embeddings',[True, False])
def test_check_results_to_docs_and_scores_method_of_vectorstore(get_vectorstore_instance, exist_embeddings):
  class DummyAssistant:
    def __init__(self):
      self.pk = 1

  class DummyQueryset:
    def __init__(self, idx):
      self.assistant = DummyAssistant()
      self.pk = idx
      self.document = 'abc' * idx
      self.distance = idx * 1.5

  base_instance = get_vectorstore_instance
  patterns = {
    True: base_instance.embeddings,
    False: None
  }
  instance = CustomVectorStore(
    manager=base_instance.manager,
    strategy=None,
    embedding_function=patterns.get(exist_embeddings),
  )
  results = [DummyQueryset(idx + 1) for idx in range(3)]
  outputs = instance._results_to_docs_and_scores(results)

  assert all([(dist is not None) == exist_embeddings for _, dist in outputs])
  assert all([isinstance(text, Document) for text, _ in outputs])

@pytest.mark.chatbot
@pytest.mark.langgraph
def test_check_results_to_docs_method_of_vectorstore(get_vectorstore_instance):
  instance = get_vectorstore_instance
  docs_and_scores = [(Document(page_content='aaa'), 1.2) for _ in range(3)]
  docs = instance._results_to_docs(docs_and_scores)

  assert len(docs_and_scores) == len(docs)
  assert all([isinstance(converted, type(original[0])) for converted, original in zip(docs, docs_and_scores)])
  assert all([converted.page_content == original[0].page_content for converted, original in zip(docs, docs_and_scores)])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_similarity_search_of_vectorstore(create_embeddingstores):
  instance, records = create_embeddingstores
  query = 'abc'
  assistant = records[0].assistant
  size = 3
  outputs = instance.similarity_search(query, k=size, assistant_id=assistant.pk)

  assert len(outputs) == size
  assert all([isinstance(record, Document) for record in outputs])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_similarity_search_with_score_of_vectorstore(create_embeddingstores):
  instance, records = create_embeddingstores
  query = 'abc'
  assistant = records[0].assistant
  size = 3
  outputs = instance.similarity_search_with_score(query, k=size, assistant_id=assistant.pk)

  assert len(outputs) == size
  assert all([isinstance(record, Document) and score is not None for record, score in outputs])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_mmr_search_of_vectorstore(create_embeddingstores):
  instance, records = create_embeddingstores
  query = 'abc'
  assistant = records[0].assistant
  size = 3
  fetch_k = 8
  outputs = instance.max_marginal_relevance_search(query, k=size, fetch_k=fetch_k, assistant_id=assistant.pk)

  assert all([isinstance(record, Document) for record in outputs])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_mmr_search_with_score_of_vectorstore(create_embeddingstores):
  instance, records = create_embeddingstores
  query = 'abc'
  assistant = records[0].assistant
  size = 3
  fetch_k = 8
  outputs = instance.max_marginal_relevance_search_with_score(query, k=size, fetch_k=fetch_k, assistant_id=assistant.pk)

  assert all([isinstance(record, Document) and score is not None for record, score in outputs])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.django_db
def test_check_from_text_method_of_vectorstore(get_instance_with_data):
  base_instance, texts, _ = get_instance_with_data
  assistant = factories.AssistantFactory()
  instance = CustomVectorStore.from_texts(
    manager=base_instance.manager,
    strategy=base_instance._distance_strategy,
    assistant_id=assistant.pk,
    texts=texts,
    embedding=base_instance.embeddings,
  )
  records = base_instance.manager.all()

  assert isinstance(instance.manager, type(base_instance.manager))
  assert isinstance(instance._distance_strategy, type(base_instance._distance_strategy))
  assert isinstance(instance.embeddings, type(base_instance.embeddings))
  assert records.count() == len(texts)
  assert all([original == _record.document for original, _record in zip(texts, records)])

@pytest.mark.chatbot
@pytest.mark.langgraph
@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_check_afrom_text_method_of_vectorstore(get_vectorstore_instance, finalize_db_connection):
  base_instance = get_vectorstore_instance
  texts = ['abc' for _ in range(10)]
  assistant = await sync_to_async(factories.AssistantFactory)()
  instance = await CustomVectorStore.afrom_texts(
    manager=base_instance.manager,
    strategy=base_instance._distance_strategy,
    texts=texts,
    embedding=base_instance.embeddings,
    assistant_id=assistant.pk,
  )
  documents = [record.document async for record in base_instance.manager.all()]

  assert isinstance(instance.manager, type(base_instance.manager))
  assert isinstance(instance._distance_strategy, type(base_instance._distance_strategy))
  assert isinstance(instance.embeddings, type(base_instance.embeddings))
  assert len(documents) == len(texts)
  assert all([original == document for original, document in zip(texts, documents)])

  finalizer = finalize_db_connection
  await sync_to_async(finalizer)()