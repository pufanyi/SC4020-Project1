"""Milvus Lite backed embedding storage helpers.

This module wraps :class:`pymilvus.MilvusClient` so callers can treat the
embedding database as a simple Python object.  The implementation focuses on a
file-backed Milvus Lite deployment which is well suited for local development
and testing – a single ``.db`` file stores all data without needing an
external Milvus service.  The wrapper still mirrors the previous API surface so
callers can insert, search, and delete vectors without learning the Milvus
client surface area.

Example
-------

```python
from img_search.database.embeddings import EmbeddingDatabase

db = EmbeddingDatabase(
    {
        "collection_name": "img_embeddings",
        "dimension": 512,
        "uri": "./milvus_lite.db",
    }
)
db.add_embeddings(
    ids=["cat-1", "cat-2"],
    embeddings=[cat_vector, another_cat_vector],
    model_name="siglip2",
    dataset_name="cats",
)

results = db.search(cat_query_vector, top_k=5)
```

The class lazily initialises the Milvus collection, creates an AUTOINDEX index
compatible with Milvus Lite, and exposes helpers for common CRUD operations.
Configurations are designed to be Hydra friendly – pass any mapping (or
``DictConfig``) containing the relevant keys and the wrapper resolves the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from pymilvus import DataType, MilvusClient, MilvusException

try:  # Optional dependency when Hydra isn't installed in minimal contexts
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - Hydra-less environments

    class _DictConfigFallback:  # type: ignore[too-many-ancestors]
        pass

    DictConfig = _DictConfigFallback  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

DEFAULT_INDEX_TYPE = "AUTOINDEX"
DEFAULT_METRIC_TYPE = "COSINE"
DEFAULT_ID_FIELD = "id"
DEFAULT_VECTOR_FIELD = "vector"
DEFAULT_MODEL_FIELD = "model_name"
DEFAULT_DATASET_FIELD = "dataset_name"
DEFAULT_MAX_LENGTH = 255


def _as_float_vectors(vectors: Iterable[np.ndarray], *, dim: int) -> list[list[float]]:
    data: list[list[float]] = []
    for vector in vectors:
        array = np.asarray(vector, dtype="float32")
        if array.ndim != 1:
            raise ValueError("Each embedding must be a one-dimensional vector")
        if array.shape[0] != dim:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{dim}, received {array.shape[0]}"
            )
        data.append(array.tolist())
    return data


def _config_to_dict(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
    else:
        container = cfg
    if not isinstance(container, Mapping):
        raise TypeError("Embedding database config must be a mapping")
    return {str(key): value for key, value in container.items()}


def _ensure_mapping(value: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping when provided")
    return {str(key): val for key, val in value.items()}


def _resolve_dimension(config: Mapping[str, Any]) -> int:
    dim = config.get("dimension", config.get("dim"))
    if dim is None:
        raise ValueError(
            "Embedding database config must specify a `dimension`/`dim` value"
        )
    try:
        return int(dim)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError("Embedding dimension must be an integer") from exc


def _field_config(
    fields_cfg: Mapping[str, Any],
    key: str,
    *,
    default_name: str,
    default_max_length: int | None,
) -> tuple[str, int | None]:
    value = fields_cfg.get(key, {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"database.fields.{key} must be a mapping when provided")
    name = str(value.get("name", default_name))
    max_length = value.get("max_length", default_max_length)
    if max_length is None:
        return name, None
    try:
        return name, int(max_length)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            f"database.fields.{key}.max_length must be an integer when provided"
        ) from exc


def _normalize_uri(uri: Any) -> str:
    uri_str = str(uri)
    parsed = urlparse(uri_str)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return uri_str

    path_str = parsed.path if parsed.scheme == "file" else uri_str
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def create_embedding_database(
    cfg: DictConfig | Mapping[str, Any],
    *,
    dim: int | None = None,
    client: MilvusClient | None = None,
) -> EmbeddingDatabase:
    """Instantiate :class:`EmbeddingDatabase` from a Hydra/OmegaConf config."""

    config = _config_to_dict(cfg)
    if dim is not None:
        config.setdefault("dimension", dim)
    return EmbeddingDatabase(config, client=client)


class EmbeddingDatabase:
    """Milvus Lite wrapper for storing and querying embeddings."""

    __slots__ = (
        "collection_name",
        "collection_description",
        "consistency_level",
        "dim",
        "metric_type",
        "index_type",
        "index_name",
        "index_params",
        "load_on_init",
        "default_search_params",
        "id_field",
        "id_max_length",
        "model_field",
        "model_max_length",
        "dataset_field",
        "dataset_max_length",
        "vector_field",
        "uri",
        "client",
        "_loaded",
    )

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        client: MilvusClient | None = None,
    ) -> None:
        config_dict = _config_to_dict(config)
        self.dim = _resolve_dimension(config_dict)
        self.collection_name = str(config_dict.get("collection_name", "img_embeddings"))

        uri = config_dict.get("uri")
        if uri is None:
            storage_cfg = config_dict.get("storage")
            if isinstance(storage_cfg, Mapping):
                uri = storage_cfg.get("path")
        if uri is None:
            uri = config_dict.get("path", "./milvus_lite.db")
        self.uri = _normalize_uri(uri)

        client_cfg = _ensure_mapping(
            config_dict.get("client", {}), label="database.client"
        )
        self.metric_type = str(config_dict.get("metric_type", DEFAULT_METRIC_TYPE))

        index_cfg = _ensure_mapping(
            config_dict.get("index", {}), label="database.index"
        )
        self.index_type = str(index_cfg.get("type", DEFAULT_INDEX_TYPE))
        self.index_name = index_cfg.get("name")
        self.index_params = (
            dict(index_cfg.get("params", {}))
            if isinstance(index_cfg.get("params", {}), Mapping)
            else {}
        )
        if index_cfg.get("params") is not None and not isinstance(
            index_cfg.get("params"), Mapping
        ):
            raise TypeError("database.index.params must be a mapping when provided")

        search_cfg = _ensure_mapping(
            config_dict.get("search", {}), label="database.search"
        )
        params_cfg = search_cfg.get("params", {})
        if params_cfg is not None and not isinstance(params_cfg, Mapping):
            raise TypeError("database.search.params must be a mapping when provided")
        self.default_search_params = dict(params_cfg or {})

        fields_cfg = _ensure_mapping(
            config_dict.get("fields", {}), label="database.fields"
        )
        self.id_field, self.id_max_length = _field_config(
            fields_cfg,
            "id",
            default_name=config_dict.get("id_field", DEFAULT_ID_FIELD),
            default_max_length=config_dict.get("id_max_length", DEFAULT_MAX_LENGTH),
        )
        self.vector_field, _ = _field_config(
            fields_cfg,
            "vector",
            default_name=config_dict.get("vector_field", DEFAULT_VECTOR_FIELD),
            default_max_length=None,
        )
        self.model_field, self.model_max_length = _field_config(
            fields_cfg,
            "model",
            default_name=config_dict.get("model_field", DEFAULT_MODEL_FIELD),
            default_max_length=config_dict.get("model_max_length", DEFAULT_MAX_LENGTH),
        )
        self.dataset_field, self.dataset_max_length = _field_config(
            fields_cfg,
            "dataset",
            default_name=config_dict.get("dataset_field", DEFAULT_DATASET_FIELD),
            default_max_length=config_dict.get(
                "dataset_max_length", DEFAULT_MAX_LENGTH
            ),
        )

        collection_cfg = _ensure_mapping(
            config_dict.get("collection", {}), label="database.collection"
        )
        self.collection_description = str(
            collection_cfg.get("description", "Image embedding store")
        )
        self.consistency_level = collection_cfg.get("consistency_level")

        self.load_on_init = bool(config_dict.get("load_on_init", True))

        self.client = client or MilvusClient(uri=self.uri, **client_cfg)
        self._ensure_collection()
        self._loaded = False
        if self.load_on_init:
            self._load_collection()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def add_embeddings(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[np.ndarray],
        model_name: str,
        dataset_name: str,
    ) -> list[str]:
        if len(ids) != len(embeddings):
            raise ValueError("`ids` and `embeddings` must have the same length")

        vectors = _as_float_vectors(embeddings, dim=self.dim)
        self._load_collection()

        payload = []
        for identifier, vector in zip(ids, vectors, strict=True):
            payload.append(
                {
                    self.id_field: identifier,
                    self.model_field: model_name,
                    self.dataset_field: dataset_name,
                    self.vector_field: vector,
                }
            )

        result = self.client.upsert(self.collection_name, payload)
        primary_keys = result.get("primary_keys") or []
        return [str(pk) for pk in primary_keys]

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self.client.delete(self.collection_name, ids=list(ids))

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 10,
        filter_expression: str | None = None,
        output_fields: Sequence[str] | None = ("model_name", "dataset_name"),
        search_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vector = _as_float_vectors([query], dim=self.dim)[0]
        params = dict(self.default_search_params)
        if search_params:
            params.update(search_params)

        self._load_collection()
        results = self.client.search(
            self.collection_name,
            data=[vector],
            limit=top_k,
            filter=filter_expression or "",
            output_fields=list(output_fields) if output_fields else None,
            search_params=params or None,
            anns_field=self.vector_field,
        )

        hits: list[dict[str, Any]] = []
        if not results:
            return hits

        for hit in results[0]:
            payload = {
                "id": hit.get("id"),
                "distance": hit.get("distance"),
            }
            entity = hit.get("entity") or {}
            if output_fields:
                for field in output_fields:
                    payload[field] = entity.get(field)
            hits.append(payload)
        return hits

    def drop(self) -> None:
        try:
            self.client.drop_collection(self.collection_name)
        except MilvusException:
            # Collection might have already been removed; ignore.
            pass

    def flush(self) -> None:
        self.client.flush(self.collection_name)

    def close(self) -> None:
        try:
            try:
                self.client.release_collection(self.collection_name)
            except MilvusException:
                pass
        finally:
            self.client.close()

    @contextmanager
    def session(self) -> Iterator[EmbeddingDatabase]:
        try:
            yield self
        finally:
            self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(
            auto_id=False, description=self.collection_description
        )
        schema.add_field(
            self.id_field,
            DataType.VARCHAR,
            is_primary=True,
            max_length=self.id_max_length,
        )
        schema.add_field(
            self.model_field,
            DataType.VARCHAR,
            max_length=self.model_max_length,
        )
        schema.add_field(
            self.dataset_field,
            DataType.VARCHAR,
            max_length=self.dataset_max_length,
        )
        schema.add_field(
            self.vector_field,
            DataType.FLOAT_VECTOR,
            dim=self.dim,
        )

        index_params = self.client.prepare_index_params()
        index_kwargs: dict[str, Any] = {}
        if self.index_name:
            index_kwargs["index_name"] = str(self.index_name)
        if self.index_params:
            index_kwargs["params"] = dict(self.index_params)
        index_params.add_index(
            field_name=self.vector_field,
            metric_type=self.metric_type,
            index_type=self.index_type,
            **index_kwargs,
        )

        create_kwargs: dict[str, Any] = {}
        if self.consistency_level is not None:
            create_kwargs["consistency_level"] = self.consistency_level

        self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params,
            **create_kwargs,
        )

    def _load_collection(self) -> None:
        if self._loaded:
            return
        self.client.load_collection(self.collection_name)
        self._loaded = True


__all__ = ["EmbeddingDatabase", "create_embedding_database"]
