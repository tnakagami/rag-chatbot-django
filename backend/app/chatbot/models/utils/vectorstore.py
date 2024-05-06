import numpy as np
import enum
import re
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from langchain_core.runnables.config import run_in_executor
from langchain.vectorstores.base import VectorStore
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from django.utils.translation import gettext_lazy
from django.db.models import Manager as DjangoManager
from django.db import IntegrityError

class DistanceStrategy(str, enum.Enum):
  EUCLIDEAN         = 'l2'
  COSINE            = 'cosine'
  MAX_INNER_PRODUCT = 'inner'

class CustomVectorStore(VectorStore):
  def __init__(
    self,
    manager: DjangoManager,
    strategy: DistanceStrategy,
    embedding_function: Embeddings,
    relevance_score_fn: Optional[Callable[[float], float]] = None,
  ):
    self.manager = manager
    self._distance_strategy = strategy
    self.embedding_function = embedding_function
    self._override_score_fn = relevance_score_fn
    self.__pattern = re.compile(r'[-+]?\d+')

  @property
  def embeddings(self) -> Embeddings:
    return self.embedding_function

  def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
    if ids:
      valid_ids = []

      for idx in ids:
        matched = self.__pattern.fullmatch(idx)

        if matched:
          valid_ids.append(int(matched.group(0)))

      try:
        self.manager.filter(pk__in=valid_ids).delete()
        is_deleted = True
      except IntegrityError:
        is_deleted = False
    else:
      is_deleted = False

    return is_deleted

  def add_texts(
    self,
    texts: Iterable[str],
    assistant_id: int,
    metadatas: Optional[List[dict]] = None,
    **kwargs: Any,
  ) -> List[str]:
    embeddings = self.embedding_function.embed_documents(list(texts))

    return self.add_embeddings(
      texts=texts,
      embeddings=embeddings,
      assistant_id=assistant_id,
      metadatas=metadatas,
      **kwargs,
    )

  def add_embeddings(
    self,
    texts: Iterable[str],
    embeddings: List[List[float]],
    assistant_id: int,
    metadatas: Optional[List[dict]] = None,
    **kwargs: Any,
  ) -> List[str]:
    outputs = []

    for text, embedding in zip(texts, embeddings):
      embedding_store = self.manager.create(
        assistant_id=assistant_id,
        embedding=embedding,
        document=text,
      )
      outputs.append(embedding_store.document)

    return outputs

  def _select_relevance_score_fn(self):
    # This method is called by `_similarity_search_with_relevance_scores` method of `VectorStore` class
    if self._override_score_fn is not None:
      relevance_score_fn = self._override_score_fn
    else:
      if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
        relevance_score_fn = self._euclidean_relevance_score_fn
      elif self._distance_strategy == DistanceStrategy.COSINE:
        relevance_score_fn = self._cosine_relevance_score_fn
      elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
        relevance_score_fn = self._max_inner_product_relevance_score_fn
      else:
        err = gettext_lazy('No supported normalization function for distance_strategy')
        raise ValueError(err)

    return relevance_score_fn

  def similarity_search(
    self, query: str, assistant_id: int, k: int = 4, **kwargs: Any
  ) -> List[Document]:
    embedding = self.embedding_function.embed_query(text=query)

    return self.similarity_search_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      **kwargs,
    )

  def similarity_search_with_score(
    self, query: str, assistant_id: int, k: int = 4, *args: Any, **kwargs: Any
  ) -> List[Tuple[Document, float]]:
    embedding = self.embedding_function.embed_query(text=query)

    return self.similarity_search_with_score_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      **kwargs,
    )

  def similarity_search_by_vector(
    self, embedding: List[float], assistant_id: int, k: int = 4, **kwargs: Any
  ) -> List[Document]:
    docs_and_scores = self.similarity_search_with_score_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      **kwargs,
    )

    return self._results_to_docs(docs_and_scores)

  def similarity_search_with_score_by_vector(
    self, embedding: List[float], assistant_id: int, k: int = 4, **kwargs: Any
  ) -> List[Tuple[Document, float]]:
    results = self.__collect_records(assistant_id, embedding)

    return self._results_to_docs_and_scores(results[:k])

  def __collect_records(self, assistant_id: int, embedding: List[float]):
    return self.manager.similarity_search_with_distance_by_vector(
      embedded_query=embedding,
      assistant_id=assistant_id,
      distance_strategy=self._distance_strategy,
    )

  def _results_to_docs(self, docs_and_scores: List[Tuple[Document, float]]):
    return [doc for doc, _ in docs_and_scores]

  def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
    docs = [
      (
        Document(page_content=record.document, metadata={'pk': record.pk, 'assistant': record.assistant.pk}),
        record.distance if self.embedding_function is not None else None,
      )
      for record in results
    ]

    return docs

  def max_marginal_relevance_search(
    self,
    query: str,
    assistant_id: int,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
  ) -> List[Document]:
    embedding = self.embedding_function.embed_query(query)

    return self.max_marginal_relevance_search_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      fetch_k=fetch_k,
      lambda_mult=lambda_mult,
      **kwargs,
    )

  def max_marginal_relevance_search_with_score(
    self,
    query: str,
    assistant_id: int,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
  ) -> List[Tuple[Document, float]]:
    embedding = self.embedding_function.embed_query(query)
    docs = self.max_marginal_relevance_search_with_score_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      fetch_k=fetch_k,
      lambda_mult=lambda_mult,
      **kwargs,
    )

    return docs

  def max_marginal_relevance_search_by_vector(
    self,
    embedding: List[float],
    assistant_id: int,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
  ) -> List[Document]:
    docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
      embedding=embedding,
      assistant_id=assistant_id,
      k=k,
      fetch_k=fetch_k,
      lambda_mult=lambda_mult,
      **kwargs,
    )

    return self._results_to_docs(docs_and_scores)

  def max_marginal_relevance_search_with_score_by_vector(
    self,
    embedding: List[float],
    assistant_id: int,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    **kwargs: Any,
  ) -> List[Tuple[Document, float]]:
    results = self.__collect_records(assistant_id, embedding)
    targets = results[:fetch_k]
    embedding_list = [record.embedding for record in targets]
    # Select maximal marginal relevance
    mmr_selected = maximal_marginal_relevance(
      np.array(embedding, dtype=np.float32),
      embedding_list,
      k=k,
      lambda_mult=lambda_mult,
    )
    # Collect candidates
    candidates = self._results_to_docs_and_scores(targets)

    return [record for idx, record in enumerate(candidates) if idx in mmr_selected]

  @classmethod
  def from_texts(
    cls,
    manager: DjangoManager,
    strategy: DistanceStrategy,
    assistant_id: int,
    texts: List[str],
    embedding: Embeddings,
    metadatas: Optional[List[dict]] = None,
    **kwargs: Any,
  ):
    embedding_vectors = embedding.embed_documents(list(texts))
    # Create instance
    store = cls(
      manager=manager,
      strategy=strategy,
      embedding_function=embedding,
    )
    store.add_embeddings(
      texts=texts,
      embeddings=embedding_vectors,
      assistant_id=assistant_id,
      metadatas=metadatas,
      **kwargs,
    )

    return store

  @classmethod
  async def afrom_texts(
    cls,
    manager: DjangoManager,
    strategy: DistanceStrategy,
    assistant_id: int,
    texts: List[str],
    embedding: Embeddings,
    metadatas: Optional[List[dict]] = None,
    **kwargs: Any,
  ):
    """Return VectorStore initialized from texts and embeddings."""
    return await run_in_executor(
      None, cls.from_texts, manager, strategy, assistant_id, texts, embedding, metadatas, **kwargs
    )
