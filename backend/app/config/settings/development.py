import os
from .base import *

DEBUG = True
ALLOWED_HOSTS = ['*']

DATABASES = {
  'default': {
    'ENGINE': 'django.db.backends.postgresql',
    'NAME': os.getenv('POSTGRES_DB'),
    'USER': os.getenv('POSTGRES_USER'),
    'PASSWORD': os.getenv('POSTGRES_PASSWORD'),
    'HOST': os.getenv('DB_HOST'),
    'PORT': 5432,
    'TEST': {
      'NAME': 'test_db',
    },
  },
}