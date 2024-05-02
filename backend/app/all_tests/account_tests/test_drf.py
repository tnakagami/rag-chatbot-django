import pytest
import json
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView
from rest_framework.test import APIRequestFactory
from rest_framework import status
from django.urls import reverse
from django.contrib.auth import get_user_model
from . import factories
from account import drf_views

User = get_user_model()

@pytest.fixture
def init_records(django_db_blocker):
  with django_db_blocker.unblock():
    screen_name = 'sample'
    info = {
      'user': factories.UserFactory(username='hoge', screen_name=screen_name),
      'screen_name': screen_name,
    }

  return info

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_can_authorize_by_using_jwt(init_records):
  # Get user
  password = 'test-pass'
  user = init_records['user']
  user.set_password(password)
  user.save()
  # Get token
  factory = APIRequestFactory()
  auth_url = reverse('api:token_obtain_pair')
  view = TokenObtainPairView.as_view()
  request = factory.post(auth_url, data={'username': user.username, 'password': password})
  response = view(request)
  assert response.status_code == status.HTTP_200_OK
  assert bool(response.data)
  assert 'access' in response.data.keys()

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_can_refresh_jwt_token(init_records):
  user = init_records['user']
  refresh = RefreshToken.for_user(user)
  factory = APIRequestFactory()
  refresh_url = reverse('api:token_refresh')
  view = TokenRefreshView.as_view()
  request = factory.post(
    refresh_url,
    data=json.dumps({'refresh': str(refresh)}),
    content_type='application/json'
  )
  response = view(request)
  assert response.status_code == status.HTTP_200_OK
  assert bool(response.data)
  assert 'access' in response.data.keys()

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_verify_jwt_token(init_records):
  user = init_records['user']
  refresh = RefreshToken.for_user(user)
  access_token = refresh.access_token
  factory = APIRequestFactory()
  refresh_url = reverse('api:token_verify')
  view = TokenVerifyView.as_view()
  request = factory.post(
    refresh_url,
    data=json.dumps({'token': str(access_token)}),
    content_type='application/json'
  )
  response = view(request)
  assert response.status_code == status.HTTP_200_OK

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_invalid_jwt_token(init_records):
  from datetime import datetime, timedelta
  user = init_records['user']
  refresh = RefreshToken.for_user(user)
  access_token = refresh.access_token
  factory = APIRequestFactory()
  refresh_url = reverse('api:token_verify')
  view = TokenVerifyView.as_view()
  access_token.set_exp(from_time=datetime.min, lifetime=timedelta(seconds=0))
  request = factory.post(
    refresh_url,
    data=json.dumps({'token': str(access_token)}),
    content_type='application/json'
  )
  response = view(request)
  assert response.status_code == status.HTTP_401_UNAUTHORIZED
  assert 'detail' in response.data.keys()
  assert 'expired' in str(response.data.get('detail', ''))

@pytest.fixture
def drf_user_profile_settings():
  factory = APIRequestFactory()
  view = drf_views.DrfUserProfile.as_view()
  url = reverse('api:account:user_profile')
  callback = lambda method, **kwargs: view(getattr(factory, method)(url, **kwargs))

  return callback

@pytest.mark.account
@pytest.mark.drf
def test_drf_user_profile_is_not_authenticated(drf_user_profile_settings):
  callback = drf_user_profile_settings
  response = callback('get')

  assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_user_profile_getmethod(init_records, drf_user_profile_settings):
  target = 'screen_name'
  user = init_records['user']
  screen_name = init_records[target]
  refresh = RefreshToken.for_user(user)
  callback = drf_user_profile_settings
  # Access user profile page
  response = callback('get', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}')

  assert response.status_code == status.HTTP_200_OK
  assert target in response.data.keys()
  assert response.data[target] == screen_name

@pytest.mark.account
@pytest.mark.drf
@pytest.mark.django_db
def test_drf_user_profile_patchmethod(init_records, drf_user_profile_settings):
  target = 'screen_name'
  new_screen_name = 'new-test-name'
  user = init_records['user']
  refresh = RefreshToken.for_user(user)
  callback = drf_user_profile_settings
  # Access user profile page
  response = callback('patch', HTTP_AUTHORIZATION=f'JWT {refresh.access_token}', data={target: new_screen_name})
  modified_user = User.objects.get(pk=user.pk)
  assert response.status_code == status.HTTP_200_OK
  assert target in response.data.keys()
  assert response.data[target] == new_screen_name
  assert modified_user.screen_name == new_screen_name
