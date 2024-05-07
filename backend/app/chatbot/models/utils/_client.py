from urllib.parse import urlparse
import httpx

def get_client(proxy, is_async=True):
  client = None
  proxy_url = proxy or None

  if proxy_url:
    parsed_url = urlparse(proxy_url)

    if parsed_url.scheme and parsed_url.netloc and parsed_url.hostname:
      if is_async:
        client = httpx.AsyncClient(proxy=proxy_url)
      else:
        client = httpx.Client(proxy=proxy_url)

  return client