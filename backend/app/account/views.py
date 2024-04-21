from django.contrib.auth.views import LoginView, LogoutView
from django.views.generic import (
  TemplateView,
  CreateView,
  UpdateView,
  DetailView,
  DeleteView
)
from django.utils.translation import gettext_lazy
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.urls import reverse_lazy, reverse
from view_breadcrumbs import (
  BaseBreadcrumbMixin,
  CreateBreadcrumbMixin,
  DetailBreadcrumbMixin,
  UpdateBreadcrumbMixin
)
from . import forms

class Index(BaseBreadcrumbMixin, TemplateView):
  template_name = 'account/index.html'
  crumbs = []

class LoginPage(BaseBreadcrumbMixin, LoginView):
  template_name = 'account/login.html'
  form_class = forms.LoginForm
  crumbs = []

class LogoutPage(LogoutView):
  template_name = 'account/index.html'
