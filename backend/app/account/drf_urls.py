from django.urls import path
from . import drf_views

urlpatterns = [
  path('profile/', drf_views.DrfUserProfile.as_view(), name='user_profile'),
]
