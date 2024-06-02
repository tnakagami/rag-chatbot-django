from rest_framework.routers import Route, DynamicRoute, SimpleRouter

class CustomConfigRouter(SimpleRouter):
  routes = [
    Route(
      url=r'^{prefix}{trailing_slash}$',
      mapping={'get': 'list', 'post': 'create'},
      name='{basename}_list',
      detail=False,
      initkwargs={'suffix': 'List'},
    ),
    DynamicRoute(
      url=r'^{prefix}/{url_path}{trailing_slash}$',
      name='{basename}_{url_name}',
      detail=False,
      initkwargs={},
    ),
    Route(
      url=r'^{prefix}/{lookup}{trailing_slash}$',
      mapping={'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'},
      name='{basename}_detail',
      detail=True,
      initkwargs={'suffix': 'Detail'},
    )
  ]

class CustomSimpleRouter(SimpleRouter):
  routes = [
    Route(
      url=r'^{prefix}{trailing_slash}$',
      mapping={'get': 'list', 'post': 'create'},
      name='{basename}_list',
      detail=False,
      initkwargs={'suffix': 'List'},
    ),
    DynamicRoute(
      url=r'^{prefix}/{url_path}{trailing_slash}$',
      name='{basename}_{url_name}',
      detail=False,
      initkwargs={},
    ),
    Route(
      url=r'^{prefix}/{lookup}{trailing_slash}$',
      mapping={'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'},
      name='{basename}_detail',
      detail=True,
      initkwargs={'suffix': 'Detail'},
    ),
  ]

class CustomDocfileRouter(SimpleRouter):
  routes = [
    Route(
      url=r'^{prefix}{trailing_slash}$',
      mapping={'get': 'list', 'post': 'create'},
      name='{basename}_list',
      detail=False,
      initkwargs={'suffix': 'List'},
    ),
    Route(
      url=r'^{prefix}/{lookup}{trailing_slash}$',
      mapping={'delete': 'destroy'},
      name='{basename}_detail',
      detail=True,
      initkwargs={'suffix': 'Detail'},
    ),
  ]