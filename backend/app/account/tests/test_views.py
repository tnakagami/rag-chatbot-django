import pytest
from dataclasses import dataclass
from django.urls import reverse
from .. import models

@dataclass
class _UserInfo:
  username: str
  email: str
  password: str
  user: models.User

@pytest.fixture(scope='module')
def init_records(django_db_blocker):
  # First user
  hoge = {
    'username': 'hoge',
    'email': 'hogehoge@example.com',
    'password': 'password1',
  }
  # Second user
  foo = {
    'username': 'foo',
    'email': 'foobar@example.com',
    'password': 'password2',
  }

  with django_db_blocker.unblock():
    users = [
      _UserInfo(**hoge, user=models.User.objects.create_user(**hoge)),
      _UserInfo(**foo, user=models.User.objects.create_user(**foo))
    ]

  return users

def test_index_view_get_access(client):
  url = reverse('account:index')
  response = client.get(url)

  assert response.status_code == 200

def test_login_view_get_access(client):
  url = reverse('account:login')
  response = client.get(url)

  assert response.status_code == 200

@pytest.mark.django_db
def test_login_view_post_access(init_records, client):
  user = init_records[0]
  params = {
    'username': user.username,
    'password': user.password,
  }
  url = reverse('account:login')
  response = client.post(url, params)

  assert response.status_code == 302
  assert response['Location'] == reverse('account:index')

@pytest.mark.django_db
def test_with_authenticated_client(init_records, client):
  info = init_records[0]
  client.force_login(info.user)
  url = reverse('account:user_profile', kwargs={'pk': info.user.pk})
  response = client.get(url)

  assert response.status_code == 200

@pytest.mark.django_db
def test_without_authentication(init_records, client):
  info = init_records[0]
  url = reverse('account:user_profile', kwargs={'pk': info.user.pk})
  response = client.get(url)

  assert response.status_code == 403

@pytest.mark.django_db
def test_invalid_user_page(init_records, client):
  own = init_records[0]
  other = init_records[1]
  client.force_login(own.user)
  url = reverse('account:user_profile', kwargs={'pk': other.user.pk})
  response = client.get(url)

  assert response.status_code == 403