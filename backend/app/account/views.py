from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import TemplateView, UpdateView, DetailView
from django.utils.translation import gettext_lazy
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse
from view_breadcrumbs import BaseBreadcrumbMixin, DetailBreadcrumbMixin, UpdateBreadcrumbMixin
from . import models, forms

class Index(BaseBreadcrumbMixin, TemplateView):
  template_name = 'account/index.html'
  crumbs = []

class LoginPage(BaseBreadcrumbMixin, LoginView):
  template_name = 'account/login.html'
  form_class = forms.LoginForm
  crumbs = []

class LogoutPage(LogoutView):
  template_name = 'account/index.html'

class IsOwner(UserPassesTestMixin):
  def test_func(self):
    user = self.get_object()
    is_valid = user.pk == self.request.user.pk

    return is_valid

class UserProfilePage(LoginRequiredMixin, IsOwner, DetailBreadcrumbMixin, DetailView):
  raise_exception = True
  model = models.User
  template_name = 'account/user_profile.html'
  context_object_name = 'owner'
  crumbs = [(gettext_lazy('User Profile'), 'user_profile')]

class UpdateUserProfile(LoginRequiredMixin, IsOwner, UpdateBreadcrumbMixin, UpdateView):
  raise_exception = True
  model = models.User
  form_class = forms.UserProfileForm
  template_name = 'account/profile_form.html'
  crumbs = [(gettext_lazy('Update User Profile'), 'update_profile')]

  def get_success_url(self):
    return reverse('account:user_profile', kwargs={'pk': self.kwargs['pk']})
