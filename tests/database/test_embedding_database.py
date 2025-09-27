from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from img_search.database import embeddings as embeddings_module
from img_search.database.embeddings import (
    EmbeddingDatabase,
    _as_float_vectors,
    create_embedding_database,
)

np = pytest.importorskip("numpy")


class DummyDataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class DummySchema:
    def __init__(self, *, auto_id: bool, description: str | None = None) -> None:
        self.auto_id = auto_id
        self.description = description
        self.fields: list[tuple[str, Any, dict[str, Any]]] = []
        self.primary_field: str | None = None

    def add_field(self, field_name: str, datatype: Any, **kwargs: Any) -> None:
        self.fields.append((field_name, datatype, kwargs))
        if kwargs.get("is_primary"):
            self.primary_field = field_name

    def verify(self) -> None:  # pragma: no cover - compatibility shim
        return


class DummyIndexParams(list[dict[str, Any]]):
    def add_index(self, **kwargs: Any) -> None:
        self.append(kwargs)


class FakeMilvusClient:
    def __init__(self, uri: str | None = None, **_: Any) -> None:  # noqa: D401
        self.uri = uri
        self.collections: dict[str, dict[str, Any]] = {}
        self.closed = False

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_schema(
        self, *, auto_id: bool, description: str | None = None
    ) -> DummySchema:
        return DummySchema(auto_id=auto_id, description=description)

    def prepare_index_params(self, field_name: str | None = None) -> DummyIndexParams:  # noqa: D401
        params = DummyIndexParams()
        if field_name is not None:  # pragma: no cover - defensive branch
            params.add_index(field_name=field_name)
        return params

    def create_collection(
        self,
        collection_name: str,
        *,
        schema: DummySchema,
        index_params: DummyIndexParams,
        **kwargs: Any,
    ) -> None:
        self.collections[collection_name] = {
            "schema": schema,
            "index_params": list(index_params),
            "loaded": False,
            "data": [],
            "search_calls": [],
            "search_results": [[]],
            "deleted_ids": [],
            "flushed": False,
            "released": False,
            "dropped": False,
            "consistency_level": kwargs.get("consistency_level"),
            "primary_field": schema.primary_field,
        }

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def load_collection(self, collection_name: str, **_: Any) -> None:
        self.collections[collection_name]["loaded"] = True

    def release_collection(self, collection_name: str, **_: Any) -> None:
        if collection_name in self.collections:
            self.collections[collection_name]["released"] = True

    def upsert(
        self, collection_name: str, data: list[Mapping[str, Any]], **_: Any
    ) -> dict[str, Any]:
        collection = self.collections[collection_name]
        collection.setdefault("upserts", []).append(list(data))
        collection["data"].extend(data)
        primary_field = collection["primary_field"]
        keys = [row[primary_field] for row in data]
        return {"upsert_count": len(data), "primary_keys": keys}

    def delete(self, collection_name: str, ids: list[str], **_: Any) -> dict[str, int]:
        collection = self.collections[collection_name]
        collection["deleted_ids"].append(list(ids))
        return {"deleted_count": len(ids)}

    def search(
        self,
        collection_name: str,
        *,
        data: list[list[float]],
        limit: int,
        filter: str,  # noqa: A002 - match Milvus client signature
        output_fields: list[str] | None,
        search_params: Mapping[str, Any] | None,
        anns_field: str | None,
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        collection = self.collections[collection_name]
        collection["search_calls"].append(
            {
                "data": data,
                "limit": limit,
                "filter": filter,
                "output_fields": output_fields,
                "search_params": dict(search_params or {}),
                "anns_field": anns_field,
            }
        )
        return collection["search_results"]

    def flush(self, collection_name: str, **_: Any) -> None:
        if collection_name in self.collections:
            self.collections[collection_name]["flushed"] = True

    def drop_collection(self, collection_name: str, **_: Any) -> None:
        if collection_name in self.collections:
            self.collections[collection_name]["dropped"] = True
            self.collections.pop(collection_name)

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def mock_milvus(monkeypatch: pytest.MonkeyPatch) -> FakeMilvusClient:
    monkeypatch.setattr(embeddings_module, "DataType", DummyDataType)
    monkeypatch.setattr(embeddings_module, "MilvusClient", FakeMilvusClient)
    monkeypatch.setattr(embeddings_module, "MilvusException", RuntimeError)
    client = FakeMilvusClient(uri="./test.db")
    return client


def make_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "collection_name": "test_collection",
        "dimension": 4,
        "uri": "./test.db",
    }
    cfg.update(overrides)
    return cfg


def test_initialization_creates_parent_dir_for_relative_uri(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    db = EmbeddingDatabase(make_config(uri="nested/milvus.db"))

    assert Path("nested").is_dir()
    assert db.uri.endswith("nested/milvus.db")
    assert isinstance(db.client, FakeMilvusClient)


def test_initialization_strips_file_scheme(tmp_path: Path) -> None:
    target = tmp_path / "store" / "embeddings.db"
    db = EmbeddingDatabase(make_config(uri=f"file:{target}"))

    assert target.parent.is_dir()
    assert db.uri == str(target)
    assert isinstance(db.client, FakeMilvusClient)
    assert db.client.uri == str(target)


def test_collection_initialization_creates_index(
    monkeypatch: MonkeyPatch,
) -> None:
    client = FakeMilvusClient(uri="./test.db")
    db = EmbeddingDatabase(make_config(), client=client)

    collection = client.collections["test_collection"]
    assert collection["loaded"] is True
    assert collection["schema"].fields == [
        ("id", DummyDataType.VARCHAR, {"is_primary": True, "max_length": 255}),
        ("model_name", DummyDataType.VARCHAR, {"max_length": 255}),
        ("dataset_name", DummyDataType.VARCHAR, {"max_length": 255}),
        ("vector", DummyDataType.FLOAT_VECTOR, {"dim": 4}),
    ]
    assert collection["index_params"] == [
        {
            "field_name": "vector",
            "metric_type": db.metric_type,
            "index_type": db.index_type,
        }
    ]


def test_create_embedding_database_from_config(monkeypatch: MonkeyPatch) -> None:
    client = FakeMilvusClient(uri="./alt.db")

    cfg = {
        "collection_name": "cfg_collection",
        "dimension": 8,
        "uri": "./alt.db",
        "metric_type": "L2",
        "index": {"type": "FLAT", "params": {"nlist": 64}},
        "fields": {
            "id": {"max_length": 128},
            "model": {"name": "model", "max_length": 64},
            "dataset": {"name": "dataset", "max_length": 64},
            "vector": {"name": "embedding"},
        },
        "collection": {"description": "Configured collection"},
        "search": {"params": {"ef": 32}},
        "load_on_init": False,
    }

    db = create_embedding_database(cfg, client=client)

    assert db.collection_name == "cfg_collection"
    assert db.metric_type == "L2"
    assert db.index_type == "FLAT"
    assert db.dim == 8
    assert db.id_field == "id"
    assert db.model_field == "model"
    assert db.dataset_field == "dataset"
    assert db.vector_field == "embedding"
    assert db.default_search_params == {"ef": 32}
    collection = client.collections["cfg_collection"]
    assert collection["loaded"] is False
    assert collection["schema"].description == "Configured collection"
    assert collection["index_params"] == [
        {
            "field_name": "embedding",
            "metric_type": "L2",
            "index_type": "FLAT",
            "params": {"nlist": 64},
        }
    ]


def test_add_embeddings_uses_upsert_and_returns_ids() -> None:
    client = FakeMilvusClient(uri="./test.db")
    db = EmbeddingDatabase(make_config(), client=client)
    vectors = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
    ]

    ids = db.add_embeddings(
        ids=["a", "b"],
        embeddings=vectors,
        model_name="model",
        dataset_name="dataset",
    )

    assert ids == ["a", "b"]
    collection = client.collections["test_collection"]
    assert collection["upserts"] == [
        [
            {
                "id": "a",
                "model_name": "model",
                "dataset_name": "dataset",
                "vector": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "id": "b",
                "model_name": "model",
                "dataset_name": "dataset",
                "vector": [0.0, 1.0, 0.0, 0.0],
            },
        ]
    ]


def test_search_returns_hit_payload() -> None:
    client = FakeMilvusClient(uri="./test.db")
    db = EmbeddingDatabase(make_config(), client=client)
    collection = client.collections["test_collection"]
    collection["search_results"] = [
        [
            {
                "id": "item-1",
                "distance": 0.1,
                "entity": {"model_name": "m1", "dataset_name": "d1"},
            },
            {
                "id": "item-2",
                "distance": 0.2,
                "entity": {"model_name": "m1", "dataset_name": "d2"},
            },
        ]
    ]

    results = db.search(np.ones(4, dtype="float32"), top_k=2)

    assert results == [
        {"id": "item-1", "distance": 0.1, "model_name": "m1", "dataset_name": "d1"},
        {"id": "item-2", "distance": 0.2, "model_name": "m1", "dataset_name": "d2"},
    ]
    assert collection["search_calls"][0]["limit"] == 2
    assert collection["search_calls"][0]["anns_field"] == "vector"


def test_delete_passes_ids_to_client() -> None:
    client = FakeMilvusClient(uri="./test.db")
    db = EmbeddingDatabase(make_config(), client=client)

    db.delete(["x", "y"])

    collection = client.collections["test_collection"]
    assert collection["deleted_ids"] == [["x", "y"]]


def test_drop_and_close_release_resources() -> None:
    client = FakeMilvusClient(uri="./test.db")
    db = EmbeddingDatabase(make_config(), client=client)

    db.drop()
    assert "test_collection" not in client.collections

    db.close()
    assert client.closed is True


def test_as_float_vectors_validates_shape() -> None:
    with pytest.raises(ValueError):
        _as_float_vectors([np.zeros((2, 2))], dim=4)

    with pytest.raises(ValueError):
        _as_float_vectors([np.zeros(3)], dim=4)
