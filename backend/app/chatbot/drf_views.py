from rest_framework import status
from rest_framework.mixins import (
  ListModelMixin,
  CreateModelMixin,
  UpdateModelMixin,
  RetrieveModelMixin,
  DestroyModelMixin,
)
from rest_framework.viewsets import GenericViewSet, ModelViewSet
from rest_framework.permissions import IsAuthenticated, BasePermission
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from drf_spectacular.utils import (
  extend_schema_view,
  extend_schema,
  OpenApiParameter,
  OpenApiExample,
)
from django.utils.translation import gettext_lazy
from . import serializers, models

class IsOwner(BasePermission):
  def has_object_permission(self, request, view, instance):
    return instance.is_owner(request.user)

@extend_schema_view(
  update=extend_schema(parameters=[OpenApiParameter(name='id', type=int, location=OpenApiParameter.PATH)]),
  partial_update=extend_schema(parameters=[OpenApiParameter(name='id', type=int, location=OpenApiParameter.PATH)]),
  destroy=extend_schema(parameters=[OpenApiParameter(name='id', type=int, location=OpenApiParameter.PATH)]),
)
class _CommonListCreateUpdateDestroyViewSet(
  ListModelMixin,
  CreateModelMixin,
  UpdateModelMixin,
  DestroyModelMixin,
  GenericViewSet,
):
  permission_classes = [IsAuthenticated, IsOwner]
  authentication_classes = [JWTAuthentication]

  def perform_create(self, serializer):
    serializer.save(user=self.request.user)

@extend_schema(tags=[gettext_lazy('Agent')])
class AgentViewSet(_CommonListCreateUpdateDestroyViewSet):
  serializer_class = serializers.AgentSerializer

  @extend_schema(
    description=gettext_lazy('Get type ids of agent model'),
    request=serializers.AgentTypeIdsSerializer,
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='types', url_name='types')
  def get_type_ids(self, request, pk=None):
    serializer = serializers.AgentTypeIdsSerializer()
    serializer.is_valid()

    return Response(serializer.data)

  @extend_schema(
    description=gettext_lazy('Get configuration format of agent model'),
    request=serializers.AgentConfigFormatSerializer,
    parameters=[OpenApiParameter(name='type_id', type=int, location=OpenApiParameter.QUERY)],
    examples=[
      OpenApiExample(
        name=label,
        description=gettext_lazy('Get configuration format of target agent'),
        value=value,
        parameter_only=('type_id', 'query'),
        request_only=True,
        response_only=False,
      )
      for value, label in serializers.AgentType.choices
    ],
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='config-format', url_name='config_format')
  def get_config_format(self, request, pk=None):
    serializer = serializers.AgentConfigFormatSerializer(None, data=request.query_params)
    serializer.is_valid(raise_exception=True)

    return Response(serializer.data)

  def get_queryset(self):
    return self.request.user.agent_configs.all()

@extend_schema(tags=[gettext_lazy('Embedding')])
class EmbeddingViewSet(_CommonListCreateUpdateDestroyViewSet):
  serializer_class = serializers.EmbeddingSerializer

  @extend_schema(
    description=gettext_lazy('Get type ids of embedding model'),
    request=serializers.EmbeddingTypeIdsSerializer,
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='types', url_name='types')
  def get_type_ids(self, request, pk=None):
    serializer = serializers.EmbeddingTypeIdsSerializer()
    serializer.is_valid()

    return Response(serializer.data)

  @extend_schema(
    description=gettext_lazy('Get distance types of embedding model'),
    request=serializers.DistanceTypeIdsSerializer,
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='distances', url_name='distances')
  def get_distances_types(self, request, pk=None):
    serializer = serializers.DistanceTypeIdsSerializer()
    serializer.is_valid()

    return Response(serializer.data)

  @extend_schema(
    description=gettext_lazy('Get configuration format of embedding model'),
    request=serializers.EmbeddingConfigFormatSerializer,
    parameters=[OpenApiParameter(name='type_id', type=int, location=OpenApiParameter.QUERY)],
    examples=[
      OpenApiExample(
        name=label,
        description=gettext_lazy('Get configuration format of target embedding'),
        value=value,
        parameter_only=('type_id', 'query'),
        request_only=True,
        response_only=False,
      )
      for value, label in serializers.AgentType.embedding_choices
    ],
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='config-format', url_name='config_format')
  def get_config_format(self, request, pk=None):
    serializer = serializers.EmbeddingConfigFormatSerializer(None, data=request.query_params)
    serializer.is_valid(raise_exception=True)

    return Response(serializer.data)

  def get_queryset(self):
    return self.request.user.embedding_configs.all()

@extend_schema(tags=[gettext_lazy('Tool')])
class ToolViewSet(_CommonListCreateUpdateDestroyViewSet):
  serializer_class = serializers.ToolSerializer

  @extend_schema(
    description=gettext_lazy('Get type ids of tool model'),
    request=serializers.ToolTypeIdsSerializer,
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='types', url_name='types')
  def get_type_ids(self, request, pk=None):
    serializer = serializers.ToolTypeIdsSerializer()
    serializer.is_valid()

    return Response(serializer.data)

  @extend_schema(
    description=gettext_lazy('Get configuration format of tool model'),
    request=serializers.EmbeddingConfigFormatSerializer,
    parameters=[OpenApiParameter(name='type_id', type=int, location=OpenApiParameter.QUERY)],
    examples=[
      OpenApiExample(
        name=label,
        description=gettext_lazy('Get configuration format of target tool'),
        value=value,
        parameter_only=('type_id', 'query'),
        request_only=True,
        response_only=False,
      )
      for value, label in serializers.ToolType.choices
    ],
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='config-format', url_name='config_format')
  def get_config_format(self, request, pk=None):
    serializer = serializers.ToolConfigFormatSerializer(None, data=request.query_params)
    serializer.is_valid(raise_exception=True)

    return Response(serializer.data)

  def get_queryset(self):
    return self.request.user.tool_configs.all()

@extend_schema(tags=[gettext_lazy('Assistant')])
class AssistantViewSet(ModelViewSet):
  permission_classes = [IsAuthenticated, IsOwner]
  authentication_classes = [JWTAuthentication]
  serializer_class = serializers.AssistantSerializer
  queryset = models.Assistant.objects.none()

  def perform_create(self, serializer):
    serializer.save(user=self.request.user)

  @extend_schema(
    description=gettext_lazy('Get tasks'),
    request=serializers.AssistantTaskSerializer,
  )
  @action(methods=['get'], detail=False, permission_classes=[IsAuthenticated], url_path='tasks', url_name='tasks')
  def get_tasks(self, request, pk=None):
    serializer = serializers.AssistantTaskSerializer(user=self.request.user)
    serializer.is_valid()

    return Response(serializer.data)

  def get_queryset(self):
    return self.request.user.assistants.all()

@extend_schema(tags=[gettext_lazy('DocumentFile')])
class DocumentFileViewSet(
  ListModelMixin,
  DestroyModelMixin,
  GenericViewSet,
):
  permission_classes = [IsAuthenticated, IsOwner]
  authentication_classes = [JWTAuthentication]
  parser_claasses = [MultiPartParser, FormParser]
  serializer_class = serializers.DocumentFileSerializer
  queryset = models.DocumentFile.objects.none()

  # Customize create method
  def create(self, request, *args, **kwargs):
    serializer = self.get_serializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    self.perform_create(serializer)

    return Response([], status=status.HTTP_202_ACCEPTED)

  def perform_create(self, serializer):
    serializer.save(user=self.request.user)

  def get_queryset(self):
    return models.DocumentFile.objects.collect_own_files(self.request.user)

@extend_schema(tags=[gettext_lazy('Thread')])
class ThreadViewSet(ModelViewSet):
  permission_classes = [IsAuthenticated, IsOwner]
  authentication_classes = [JWTAuthentication]
  serializer_class = serializers.ThreadSerializer
  queryset = models.Thread.objects.none()

  def get_queryset(self):
    return models.Thread.objects.collect_own_threads(self.request.user)