import factory
from factory.fuzzy import FuzzyChoice, FuzzyText, BaseFuzzyAttribute, random
from faker import Factory as FakerFactory
from django.utils import timezone
from all_tests.account_tests.factories import UserFactory, _clip
from chatbot.models.rag import (
  Agent,
  Embedding,
  Tool,
  Assistant,
  Thread,
  EmbeddingStore,
  LangGraphCheckpoint,
)
from chatbot.models.agents import (
  AgentArgs, 
  ToolArgs, 
  AgentType, 
  ToolType,
)

faker = FakerFactory.create()

def _get_values(choices):
  return [val for val, _ in choices]

agent_types = _get_values(AgentType.choices)
distance_strategies = _get_values(Embedding.DistanceType.choices)
emb_types = _get_values(AgentType.get_embedding_choices())
tool_types = _get_values(ToolType.choices)

class _FuzzyVectore(BaseFuzzyAttribute):
  def __init__(self, ndim):
    super().__init__()
    self.ndim = ndim

  def fuzz(self):
    return [random.randgen.uniform(0, 1) for _ in range(self.ndim)]

class _FuzzyCheckpoint(BaseFuzzyAttribute):
  def fuzz(self):
    count = random.randgen.randrange(1, 10)
    return {idx: random.randgen.uniform(0, 1) for idx in range(count)}

class AgentFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = Agent

  user = factory.SubFactory(UserFactory)
  name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))
  config = {}
  agent_type = FuzzyChoice(agent_types)

class EmbeddingFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = Embedding

  user = factory.SubFactory(UserFactory)
  name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))
  config = {}
  distance_strategy = FuzzyChoice(distance_strategies)
  emb_type = FuzzyChoice(emb_types)

class ToolFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = Tool

  user = factory.SubFactory(UserFactory)
  name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))
  config = {}
  tool_type = FuzzyChoice(tool_types)

class AssistantFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = Assistant

  user = factory.SubFactory(UserFactory)
  name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))
  system_message = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))
  agent = factory.SubFactory(AgentFactory)
  embedding = factory.SubFactory(EmbeddingFactory)

  @factory.post_generation
  def tools(self, create, extracted, **kwargs):
    if not create or not extracted:
      # Simple build, or nothing to add, do nothing.
      return None

    # Add the iterable of groups using bulk addition
    self.tools.add(*extracted)

class ThreadFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = Thread

  assistant = factory.SubFactory(AssistantFactory)
  name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 255))

class EmbeddingStoreFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = EmbeddingStore

  class Params:
    ndim = 10

  assistant = factory.SubFactory(AssistantFactory)
  embedding = _FuzzyVectore(ndim=Params.ndim)
  document = FuzzyText(length=64)

  @classmethod
  def _create(cls, model_class, *args, **kwargs):
    manager = cls._get_manager(model_class)
    assistant = kwargs.pop('assistant')

    return manager.create(assistant_id=assistant.pk, *args, **kwargs)

class CheckpointFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = LangGraphCheckpoint
    exclude = ('no_previous_exists',)

  no_previous_exists = False
  thread = factory.SubFactory(ThreadFactory)
  current_time = factory.LazyFunction(timezone.now)
  previous_time = factory.Maybe(
    'no_previous_exists',
    yes_declaration=None,
    no_declaration=factory.LazyFunction(timezone.now),
  )
  checkpoint = _FuzzyCheckpoint()