import pytest
from account import forms

@pytest.mark.account
@pytest.mark.form
@pytest.mark.parametrize('screen_name,is_valid', [
  ('name', True),
  ('1'*128, True),
  ('', True),
  (None, True),
  ('1'*129, False),
])
def test_user_profile_form(screen_name, is_valid):
  params = {
    'screen_name': screen_name,
  }
  form = forms.UserProfileForm(data=params)
  assert form.is_valid() is is_valid