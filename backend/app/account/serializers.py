from rest_framework.serializers import ModelSerializer
from . import models

class UserProfileSerializer(ModelSerializer):
  class Meta:
    model = models.User
    fields = ('screen_name', )