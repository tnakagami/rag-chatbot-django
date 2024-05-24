from django.urls import include, path
from django.utils.translation import gettext_lazy
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView
from drf_spectacular.utils import extend_schema

wrapper = extend_schema(tags=[gettext_lazy('Token')])
wrapper(TokenObtainPairView)
wrapper(TokenRefreshView)
wrapper(TokenVerifyView)

urlpatterns = [
  path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
  path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
  path('token/verify/', TokenVerifyView.as_view(), name='token_verify'),
  path('account/', include(('account.drf_urls', 'account'))),
  path('chatbot/', include(('chatbot.drf_urls', 'chatbot'))),
]