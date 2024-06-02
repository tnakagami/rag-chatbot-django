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
from rest_framework.authentication import BaseAuthentication
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication
from adrf.views import APIView as AsyncAPIView
from drf_spectacular.utils import (
  extend_schema_view,
  extend_schema,
  OpenApiParameter,
  OpenApiExample,
)
from django.utils.translation import gettext_lazy
from django.shortcuts import get_object_or_404
from django.http import StreamingHttpResponse
from asgiref.sync import sync_to_async
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

# =====================
# = Server Side Event =
# =====================
class AsyncJWTAuthentication(BaseAuthentication):
  def __init__(self, *args, **kwargs):
    self.wrapper = JWTAuthentication(*args, **kwargs)

  async def authenticate(self, request):
    return await sync_to_async(self.wrapper.authenticate)(request)

class AsyncIsOwner(BasePermission):
  async def has_object_permission(self, request, view, instance):
    return await sync_to_async(instance.is_owner)(request.user)

@extend_schema(
  tags=[gettext_lazy('EventStream')],
  description=gettext_lazy('Human in the loop'),
  request=serializers.LangChainChatbotSerializer,
)
class EventStreamView(AsyncAPIView):
  permission_classes = [IsAuthenticated, AsyncIsOwner]
  authentication_classes = [AsyncJWTAuthentication]
  serializer_class = serializers.LangChainChatbotSerializer
  http_method_names = ['post']

  async def aget_serializer(self, *args, **kwargs):
    kwargs.setdefault('context', {
      'request': self.request,
      'format': self.format_kwarg,
      'view': self,
    })

    return self.serializer_class(*args, **kwargs)

  async def post(self, request, *args, **kwargs):
    serializer = await self.aget_serializer(user=self.request.user, data=request.data)
    await serializer.ais_valid(raise_exception=True)
    controller = await serializer.aget_controller()
    contents = await serializer.aget_contents()

    return StreamingHttpResponse(controller.event_stream(contents), content_type='text/event-stream')