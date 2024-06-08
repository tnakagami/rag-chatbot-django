import mimetypes
from typing import BinaryIO, List, Optional, Any
from langchain_community.document_loaders import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.parsers import BS4HTMLParser, PDFMinerParser
from langchain_community.document_loaders.parsers.generic import MimeTypeBasedParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.runnables import RunnableSerializable, RunnableConfig
from .vectorstore import CustomVectorStore

HANDLERS = {
  'application/pdf': PDFMinerParser(),
  'text/plain': TextParser(),
  'text/html': BS4HTMLParser(),
  'application/msword': MsWordParser(),
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': MsWordParser(),
}

class IngestBlobRunnable(RunnableSerializable[BinaryIO, List[str]]):
  store: CustomVectorStore
  record_id: int
  batch_size: int = 128
  splitter: Optional[TextSplitter] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  parser: Optional[BaseBlobParser] = MimeTypeBasedParser(handlers=HANDLERS, fallback_parser=None)

  class Config:
    arbitrary_types_allowed = True

  def invoke(self, blob: Blob, config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[str]:
    docs2index = []
    ids = []
    kwargs.update({'assistant_id': self.record_id})

    for document in self.parser.lazy_parse(blob):
      docs = self.splitter.split_documents([document])

      for doc in docs:
        doc.page_content = doc.page_content.replace('\x00', 'x')
      docs2index.extend(docs)

      if len(docs2index) >= self.batch_size:
        ids.extend(self.store.add_documents(docs2index, **kwargs))
        docs2index = []

    if docs2index:
      ids.extend(self.store.add_documents(docs2index, **kwargs))

    return ids

  def convert_input2blob(self, field: Any, max_len: int = 1024) -> Blob:
    data = field.read()
    filename = field.name
    mimetype = self._guess_mimetype(filename, data, max_len=max_len)
    blob = Blob.from_data(
      data=data,
      path=filename,
      mime_type=mimetype,
    )

    return blob

  def _guess_mimetype(self, name: str, data: bytes, max_len: int = 1024) -> str:
    mime_type, _ = mimetypes.guess_type(name)

    if mime_type:
      out = mime_type
    else:
      if data.startswith(b'%PDF'):
        out = 'application/pdf'
      elif data.startswith((b'\x50\x4B\x03\x04', b'\x50\x4B\x05\x06', b'\x50\x4B\x07\x08')):
        out = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      elif data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        out = 'application/msword'
      elif data.startswith(b'\x09\x00\xff\x00\x06\x00'):
        out = 'application/vnd.ms-excel'
      else:
        try:
          length = len(data)

          if length > max_len:
            length = max_len
          decoded = data[:length].decode('utf-8', errors='ignore')

          if all(char in decoded for char in (',', '\n')) or all(char in decoded for char in ('\t', '\n')):
            out = 'text/csv'
          elif decoded.isprintable() or decoded == '':
            out = 'text/plain'
          else:
            out = 'application/octet-stream'
        except UnicodeDecodeError:
          out = 'application/octet-stream'

    return out
