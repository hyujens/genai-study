import pytest

from rag.vector_db_lighter import VectorDB


def test_insert_and_get_record_within_namespace():
    db = VectorDB(dim=3, max_elements=10)

    db.insert("articles", "doc_1", [1.0, 0.0, 0.0], {"topic": "python"})

    assert db.get("articles", "doc_1") == {
        "id": "doc_1",
        "metadata": {"topic": "python"},
    }
    assert db.get("other", "doc_1") is None


def test_search_returns_nearest_records_and_cosine_similarity():
    db = VectorDB(dim=3, max_elements=10)
    db.insert("memory", "same", [1.0, 0.0, 0.0], {"content": "same"})
    db.insert("memory", "near", [0.8, 0.2, 0.0], {"content": "near"})
    db.insert("memory", "far", [0.0, 1.0, 0.0], {"content": "far"})

    results = db.search("memory", [1.0, 0.0, 0.0], k=2)

    assert [result["id"] for result in results] == ["same", "near"]
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert results[0]["metadata"] == {"content": "same"}
    assert results[1]["similarity"] < results[0]["similarity"]


def test_search_caps_k_at_number_of_records():
    db = VectorDB(dim=2, max_elements=10)
    db.insert("memory", "doc_1", [1.0, 0.0], {})

    assert len(db.search("memory", [1.0, 0.0], k=5)) == 1
    assert db.search("missing", [1.0, 0.0], k=5) == []
    assert db.search("memory", [1.0, 0.0], k=0) == []


def test_update_replaces_vector_and_metadata():
    db = VectorDB(dim=2, max_elements=10)
    db.insert("memory", "doc_1", [1.0, 0.0], {"version": 1})
    db.insert("memory", "doc_2", [0.0, 1.0], {"version": 1})

    db.update("memory", "doc_1", [0.0, 1.0], {"version": 2})

    result = db.search("memory", [0.0, 1.0], k=2)
    assert {item["id"] for item in result} == {"doc_1", "doc_2"}
    assert db.get("memory", "doc_1") == {
        "id": "doc_1",
        "metadata": {"version": 2},
    }


def test_delete_removes_record_from_get_and_search():
    db = VectorDB(dim=2, max_elements=2)
    db.insert("memory", "doc_1", [1.0, 0.0], {})
    db.insert("memory", "doc_2", [0.0, 1.0], {})

    db.delete("memory", "doc_1")

    assert db.get("memory", "doc_1") is None
    assert [item["id"] for item in db.search("memory", [1.0, 0.0], k=2)] == [
        "doc_2"
    ]

    db.insert("memory", "doc_3", [1.0, 0.0], {})
    assert db.get("memory", "doc_3") == {"id": "doc_3", "metadata": {}}


def test_invalid_operations_raise_clear_errors():
    db = VectorDB(dim=2, max_elements=1)

    with pytest.raises(ValueError, match="dimension"):
        db.insert("memory", "doc_1", [1.0], {})

    db.insert("memory", "doc_1", [1.0, 0.0], {})

    with pytest.raises(ValueError, match="already exists"):
        db.insert("memory", "doc_1", [1.0, 0.0], {})
    with pytest.raises(ValueError, match="max_elements"):
        db.insert("memory", "doc_2", [0.0, 1.0], {})
    with pytest.raises(KeyError, match="not found"):
        db.update("memory", "missing", [1.0, 0.0], {})
    with pytest.raises(KeyError, match="not found"):
        db.delete("memory", "missing")
