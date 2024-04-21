from django.urls import include, path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'account'

urlpatterns = [
  path('', views.Index.as_view(), name='index'),
  path('login/', views.LoginPage.as_view(), name='login'),
  path('logout/', views.LogoutPage.as_view(), name='logout'),
]