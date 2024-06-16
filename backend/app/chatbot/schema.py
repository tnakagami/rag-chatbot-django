from drf_spectacular.authentication import TokenScheme

class AsyncJWTAuthenticationScheme(TokenScheme):
  target_class = 'chatbot.drf_views.AsyncJWTAuthentication'
  name = 'AsyncJWTAuthentication'
