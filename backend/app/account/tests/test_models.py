import pytest
from django.core.exceptions import ValidationError
from django.db.utils import IntegrityError, DataError
from pytest_factoryboy import register
from account import models
from . import factories

register(factories.UserFactory)

@pytest.mark.django_db
class TestUserModel:
  def test_user_factory(self, user_factory):
    assert user_factory is factories.UserFactory
  def test_user(self, user):
    assert isinstance(user, models.User)

  # In the case of creating user
  @pytest.mark.parametrize('user__username,user__email,user__screen_name,expected_name,expected_email,expected_screen', [
    ('hoge', 'hoge@example.com', 'hogehoge', 'hoge', 'hoge@example.com', 'hogehoge'), 
    ('1'*128, '{}@ok.com'.format('1'*121), '1'*128, '1'*128, '{}@ok.com'.format('1'*121), '1'*128),
    ('foo', 'foo@ok.com', '', 'foo', 'foo@ok.com', ''), 
  ])
  def test_user_creation(self, user, expected_name, expected_email, expected_screen):
    assert user.username == expected_name
    assert user.email == expected_email
    assert user.screen_name == expected_screen

  # In the case of creating superuser
  def test_superuser_creation(self):
    _user = factories.UserFactory.build()
    superuser = models.User.objects.create_superuser(
      username=_user.username,
      email=_user.email,
      is_staff=True,
      is_superuser=True,
    )
    assert superuser.is_staff
    assert superuser.is_superuser

  # In the case of creating superuser without `is_staff=True`
  def test_superuser_is_not_staffuser(self):
    _user = factories.UserFactory.build()
    err = 'Superuser must have is_staff=True.'
    with pytest.raises(ValueError) as ex:
      superuser = models.User.objects.create_superuser(
        username=_user.username,
        email=_user.email,
        is_staff=False,
        is_superuser=True,
      )
    assert err in ex.value.args

  # In the case of creating superuser without `is_superuser=True`
  def test_superuser_is_not_superuser(self):
    _user = factories.UserFactory.build()
    err = 'Superuser must have is_superuser=True.'
    with pytest.raises(ValueError) as ex:
      superuser = models.User.objects.create_superuser(
        username=_user.username,
        email=_user.email,
        is_staff=True,
        is_superuser=False,
      )
    assert err in ex.value.args

  # In the case of empty username
  def test_empty_username(self):
    _user = factories.UserFactory.build()
    err = 'The given username must be set'

    with pytest.raises(ValueError) as ex:
      models.User.objects.create_user(username='', email=_user.email)
    assert err in ex.value.args

  # In the case of empty email
  def test_empty_email(self):
    _user = factories.UserFactory.build()
    err = 'The given email must be set'

    with pytest.raises(ValueError) as ex:
      models.User.objects.create_user(username=_user.username, email='')
    assert err in ex.value.args

  # In the case of invalid username
  def test_invalid_username(self):
    with pytest.raises(DataError):
      user = factories.UserFactory.build(username='1'*129)
      user.save()

  # In the case of invalid screen name
  def test_invalid_screen_name(self):
    with pytest.raises(DataError):
      user = factories.UserFactory.build(screen_name='1'*129)
      user.save()

  # In the case of registering the same username
  def test_invalid_same_username(self):
    username = 'hoge'
    valid_user = factories.UserFactory.build(username=username)
    valid_user.save()

    with pytest.raises(IntegrityError):
      invalid_user = factories.UserFactory.build(username=username)
      invalid_user.save()

  # In the case of registering the same email
  def test_invalid_same_email(self):
    email = 'hoge@example.com'
    valid_user = factories.UserFactory.build(email=email)
    valid_user.save()

    with pytest.raises(IntegrityError):
      invalid_user = factories.UserFactory.build(email=email)
      invalid_user.save()

  # In the case of user's shot name
  @pytest.mark.parametrize('user__username,user__screen_name,expected', [
    ('1'*32, '2'*32, '2'*32,), 
    ('1'*32, '', '1'*32,),
    ('1'*32, '2'*33, '{}...'.format('2'*32)), 
    ('1'*33, '', '{}...'.format('1'*32)), 
  ])
  def test_user_shortname(self, user, expected):
    assert user.get_short_name() == expected

  # In the case of user's full name
  @pytest.mark.parametrize('user__username,user__screen_name,expected', [
    ('1'*32, '2'*32, '2'*32,), 
    ('1'*32, '', '1'*32,),
    ('1'*33, '2'*33, '2'*33), 
    ('1'*33, '', '1'*33),
    ('1'*128, '2'*128, '2'*128),
    ('1'*128, '', '1'*128),
  ])
  def test_user_fullname(self, user, expected):
    assert user.get_full_name() == expected