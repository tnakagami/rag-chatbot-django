"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls.i18n import i18n_patterns
from django.contrib import admin
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import (
  SpectacularAPIView,
  SpectacularRedocView,
  SpectacularSwaggerView,
)

restful_docs =[
  path('schema/', SpectacularAPIView.as_view(), name='schema'),
  path('docs/', SpectacularSwaggerView.as_view(url_name='restful:schema'), name='swagger-ui'),
  path('redoc/', SpectacularRedocView.as_view(url_name='restful:schema'), name='redoc'),
]

urlpatterns = [
    path('admin/', admin.site.urls),
    path('markdownx/', include('markdownx.urls')),
    path('i18n/', include('django.conf.urls.i18n')),
    path('api/v1/', include(('config.drf_urls', 'api'))),
] + i18n_patterns(
    path('', include(('account.urls', 'account'))),
    path('chatbot/', include(('chatbot.urls', 'chatbot'))),
    path('restful/', include((restful_docs, 'restful'))),
)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)