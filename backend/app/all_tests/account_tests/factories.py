import factory
from faker import Factory as FakerFactory
from django.utils import timezone
from account import models

faker = FakerFactory.create()

def _clip(value: str, max_len: int) -> str:
  if len(value) > max_len:
    clipped = value[:max_len]
  else:
    clipped = value

  return clipped

class UserFactory(factory.django.DjangoModelFactory):
  class Meta:
    model = models.User

  username = factory.LazyAttribute(lambda instance: _clip(faker.name(), 128))
  email = factory.LazyAttribute(lambda instance: _clip(f'{instance.username}@example.com', 128).lower())
  screen_name = factory.LazyAttribute(lambda instance: _clip(faker.name(), 128))
  date_joined = factory.LazyFunction(timezone.now)