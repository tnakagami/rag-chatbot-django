from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.utils.translation import gettext_lazy

class LoginForm(AuthenticationForm):
  screen_name = forms.CharField(
    label=gettext_lazy('screen name'), 
    max_length=128,
    widget=forms.TextInput(attrs={
      'class': 'form-control',
    })
  )

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    for field in self.fields.values():
      field.widget.attrs['class'] = 'form-control'
