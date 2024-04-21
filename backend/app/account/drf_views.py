from rest_framework.generics import RetrieveAPIView, UpdateAPIView
from rest_framework.permissions import BasePermission 
from rest_framework_simplejwt.authentication import JWTAuthentication
from . import models, serializers

class DrfIsOwner(BasePermission):
  def has_permission(self, request, view):
    return request.user.is_authenticated

  def has_object_permission(self, request, view, instance):
    is_valid = instance.pk == request.user.pk

    return is_valid

class DrfUserProfile(RetrieveAPIView, UpdateAPIView):
  queryset = models.User.objects.all()
  serializer_class = serializers.UserProfileSerializer
  permission_classes = [DrfIsOwner]
  authentication_classes = [JWTAuthentication]
  http_method_names = ['get', 'patch']
