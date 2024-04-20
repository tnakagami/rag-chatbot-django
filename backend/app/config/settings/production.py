import os
from .base import *

ALLOWED_HOSTS = os.getenv('DJANGO_WWW_VHOST').split(',')

DATABASES = {
  'default': {
      'ENGINE': 'django.db.backends.postgresql',
      'NAME': os.getenv('POSTGRES_DB'),
      'USER': os.getenv('POSTGRES_USER'),
      'PASSWORD': os.getenv('POSTGRES_PASSWORD'),
      'HOST': os.getenv('DB_HOST'),
      'PORT': os.getenv('DB_PORT'),
  },
}