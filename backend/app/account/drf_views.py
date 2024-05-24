from rest_framework.generics import RetrieveAPIView, UpdateAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from drf_spectacular.utils import extend_schema
from django.utils.translation import gettext_lazy
from . import models, serializers

@extend_schema(tags=[gettext_lazy('Account')])
class DrfUserProfile(RetrieveAPIView, UpdateAPIView):
  queryset = models.User.objects.all()
  serializer_class = serializers.UserProfileSerializer
  permission_classes = [IsAuthenticated]
  authentication_classes = [JWTAuthentication]
  http_method_names = ['get', 'patch']
  lookup_field = 'pk'

  def get_object(self):
    # Force `user.pk` to be specified
    self.kwargs[self.lookup_field] = self.request.user.pk

    return super().get_object()