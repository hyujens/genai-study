from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import hnswlib


class VectorRecord(TypedDict):
    id: str
    metadata: dict[str, Any]


class SearchResult(VectorRecord):
    similarity: float


@dataclass
class _Namespace:
    index: hnswlib.Index
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    id_to_label: dict[str, int] = field(default_factory=dict)
    label_to_id: dict[int, str] = field(default_factory=dict)
    next_label: int = 0
    deleted_count: int = 0


class VectorDB:
    """Small namespace-aware in-memory vector store backed by hnswlib."""

    def __init__(
        self,
        dim: int,
        max_elements: int,
        *,
        ef_construction: int = 200,
        m: int = 16,
        ef: int = 50,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be greater than zero")
        if max_elements <= 0:
            raise ValueError("max_elements must be greater than zero")

        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = m
        self.ef = ef
        self._namespaces: dict[str, _Namespace] = {}

    def insert(
        self,
        namespace: str,
        id: str,
        vector: Sequence[float],
        metadata: dict[str, Any],
    ) -> None:
        self._validate_vector(vector)
        ns = self._get_or_create_namespace(namespace)

        if id in ns.records:
            raise ValueError(f"vector ID already exists: {id}")
        if len(ns.records) >= self.max_elements:
            raise ValueError(f"namespace has reached max_elements: {namespace}")

        label = ns.next_label
        ns.next_label += 1
        replace_deleted = ns.deleted_count > 0
        ns.index.add_items([vector], [label], replace_deleted=replace_deleted)

        if replace_deleted:
            ns.deleted_count -= 1
        ns.records[id] = dict(metadata)
        ns.id_to_label[id] = label
        ns.label_to_id[label] = id

    def search(
        self, namespace: str, vector: Sequence[float], k: int = 3
    ) -> list[SearchResult]:
        self._validate_vector(vector)
        ns = self._namespaces.get(namespace)
        if ns is None or not ns.records or k <= 0:
            return []

        result_count = min(k, len(ns.records))
        if result_count > self.ef:
            ns.index.set_ef(result_count)

        labels, distances = ns.index.knn_query([vector], k=result_count)
        results: list[SearchResult] = []
        for label, distance in zip(labels[0], distances[0], strict=True):
            id = ns.label_to_id[int(label)]
            results.append(
                {
                    "id": id,
                    "similarity": 1.0 - float(distance),
                    "metadata": dict(ns.records[id]),
                }
            )
        return results

    def get(self, namespace: str, id: str) -> VectorRecord | None:
        ns = self._namespaces.get(namespace)
        if ns is None or id not in ns.records:
            return None
        return {"id": id, "metadata": dict(ns.records[id])}

    def update(
        self,
        namespace: str,
        id: str,
        vector: Sequence[float],
        metadata: dict[str, Any],
    ) -> None:
        self._validate_vector(vector)
        ns = self._require_record(namespace, id)
        ns.index.add_items([vector], [ns.id_to_label[id]])
        ns.records[id] = dict(metadata)

    def delete(self, namespace: str, id: str) -> None:
        ns = self._require_record(namespace, id)
        label = ns.id_to_label.pop(id)
        ns.index.mark_deleted(label)
        del ns.records[id]
        del ns.label_to_id[label]
        ns.deleted_count += 1

    def _get_or_create_namespace(self, namespace: str) -> _Namespace:
        if namespace not in self._namespaces:
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.m,
                allow_replace_delete=True,
            )
            index.set_ef(self.ef)
            self._namespaces[namespace] = _Namespace(index=index)
        return self._namespaces[namespace]

    def _require_record(self, namespace: str, id: str) -> _Namespace:
        ns = self._namespaces.get(namespace)
        if ns is None or id not in ns.records:
            raise KeyError(f"vector ID not found in namespace {namespace!r}: {id}")
        return ns

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dim:
            raise ValueError(
                f"vector dimension must be {self.dim}, received {len(vector)}"
            )
