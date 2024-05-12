from dataclasses import dataclass
from typing import Any, Union
from urllib.parse import urlparse
import httpx

def get_client(proxy, is_async: bool = True) -> Union[httpx.Client, httpx.AsyncClient]:
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

@dataclass
class LocalField:
  name: str = ''
  value: Any = None
  default: Any = None
  data_type: type = object
  label: str = ''

  def _judge(self, target_type: type) -> bool:
    instance = self.data_type()

    return type(instance) == target_type

  @property
  def is_int(self) -> bool:
    return self._judge(int)
  @property
  def is_bool(self) -> bool:
    return self._judge(bool)
  @property
  def is_str(self) -> bool:
    return self._judge(str)
  @property
  def is_list(self) -> bool:
    return self._judge(list)
  @property
  def is_dict(self) -> bool:
    return self._judge(dict)

  @property
  def data(self) -> Any:
    if self.value is None:
      val = None
    else:
      try:
        if self.is_bool:
          val = self.data_type(eval(self.value))
        else:
          val = self.data_type(self.value)
      except Exception:
        try:
          tmp = str(self.value)
          val = self.data_type(eval(tmp))
        except Exception:
          val = self.default

    return val

  def asdict(self) -> dict:
    return dict([self.astuple()])

  def astuple(self) -> tuple:
    return tuple((self.name, self.data))